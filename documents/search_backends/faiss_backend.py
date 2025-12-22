from pathlib import Path
from typing import Sequence
import os
import threading
import faiss
import numpy as np
import logging

from .base import SearchBackend, SearchResult
from documents.models import Chunk
from documents.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class FaissSearchBackend(SearchBackend):
    def __init__(
        self,
        index_path: str | Path,
        embedding_service: EmbeddingService | None = None,
        dimension: int | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.embedding_service = embedding_service or EmbeddingService()

        if dimension is None:
            sample_vec = self.embedding_service.embed_chunks(["__probe__"])[0]
            dimension = len(sample_vec)

        self.dimension = dimension
        self._lock = threading.RLock()

        self.index = self._load_or_create_index()
        self._index_mtime = self._get_file_mtime_or_none()

    # --- index file helpers ---

    def _get_file_mtime_or_none(self) -> float | None:
        try:
            if self.index_path.exists():
                return self.index_path.stat().st_mtime
        except OSError:
            return None
        return None

    def _maybe_reload_index(self) -> None:
        """
        index ファイルが更新されていれば reload する。
        reload 失敗時は既存 index を維持（サービス継続優先）。
        """
        current_mtime = self._get_file_mtime_or_none()
        if current_mtime is None:
            return

        # 変更がなければ何もしない
        if self._index_mtime is not None and current_mtime <= self._index_mtime:
            return

        try:
            new_index = faiss.read_index(str(self.index_path))
            # dimension の整合性チェック
            d = getattr(new_index, "d", None)
            if d is not None and int(d) != int(self.dimension):
                raise RuntimeError(f"FAISS dimension mismatch: file_d={d} expected={self.dimension}")

            self.index = new_index  # type: ignore[assignment]
            self._index_mtime = current_mtime
            logger.warning(
                "faiss:index reloaded path=%s ntotal=%d mtime=%.3f",
                str(self.index_path), int(self.index.ntotal), current_mtime
            )
        except Exception:
            logger.exception("faiss:index reload failed (keeping existing index) path=%s", str(self.index_path))

    # --- index create/load/save ---

    def _create_empty_index(self) -> faiss.IndexIDMap2:
        base_index = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap2(base_index)

    def _load_or_create_index(self) -> faiss.IndexIDMap2:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            return index  # type: ignore[return-value]
        index = self._create_empty_index()
        self._save_index(index)
        return index

    def _save_index(self, index: faiss.Index | None = None) -> None:
        """
        atomic write（tmp→replace）
        読み手が書き込み途中のファイルを掴まないようにする。
        """
        index = index or self.index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        final_path = str(self.index_path)
        tmp_path = final_path + ".tmp"

        faiss.write_index(index, tmp_path)
        os.replace(tmp_path, final_path)

        # 保存後のmtimeを保持
        self._index_mtime = self._get_file_mtime_or_none()

    # --- index mutate ops ---

    def index_chunks(self, chunk_ids: Sequence[int]) -> None:
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return

        chunks = list(Chunk.objects.filter(id__in=chunk_ids).select_related("document"))
        if not chunks:
            return

        texts = [c.content for c in chunks]
        ids_np = np.array([c.id for c in chunks], dtype="int64")

        embeddings = self.embedding_service.embed_chunks(texts)
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        with self._lock:
            # 最新ファイルがあればリロード
            self._maybe_reload_index()

            try:
                selector = faiss.IDSelectorBatch(ids_np)
                self.index.remove_ids(selector)
            except Exception:
                pass

            self.index.add_with_ids(vectors, ids_np)
            self._save_index()

    def delete_chunks(self, chunk_ids: Sequence[int]) -> None:
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return

        ids_np = np.array(chunk_ids, dtype="int64")
        selector = faiss.IDSelectorBatch(ids_np)

        with self._lock:
            self._maybe_reload_index()
            self.index.remove_ids(selector)
            self._save_index()

    def rebuild_index(self) -> None:
        """
        新しい index をローカルで完成させてから swap する。
        途中状態（ntotal=0）を外に見せない。
        """
        total = Chunk.objects.count()
        logger.warning("faiss:rebuild_index:start chunk_count=%d", total)

        if total == 0:
            # 空で上書きしてチャンク全滅を防ぐ
            logger.error("faiss:rebuild_index:aborted chunk_count=0 (skip overwrite)")
            return

        new_index = self._create_empty_index()
        qs = Chunk.objects.all().order_by("id")

        batch_size = 256
        offset = 0
        while True:
            batch = list(qs[offset: offset + batch_size])
            if not batch:
                break

            texts = [c.content for c in batch]
            ids_np = np.array([c.id for c in batch], dtype="int64")

            embeddings = self.embedding_service.embed_chunks(texts)
            vectors = np.array(embeddings, dtype="float32")

            faiss.normalize_L2(vectors)
            if offset == 0:
                norms2 = np.linalg.norm(vectors, axis=1)
                logger.warning(
                    "faiss:rebuild_index norms(after_norm) mean=%.3f min=%.3f max=%.3f",
                    float(norms2.mean()), float(norms2.min()), float(norms2.max())
                )
            new_index.add_with_ids(vectors, ids_np)

            offset += batch_size

        with self._lock:
            # 先に保存（atomic）→ 成功したら swap
            self._save_index(new_index)
            self.index = new_index  # type: ignore[assignment]

        logger.warning("faiss:rebuild_index:finish ntotal=%d", int(self.index.ntotal))

    def search(self, query_embedding: list[float], top_k: int = 5, filters: dict | None = None) -> list[SearchResult]:
        with self._lock:
            # ファイル更新に追従
            self._maybe_reload_index()

            if self.index.ntotal == 0:
                return []

            xq = np.array([query_embedding], dtype="float32")
            faiss.normalize_L2(xq)

            # filter / search_k ロジック
            department_id = None
            department_code = None
            if filters:
                department_id = filters.get("department_id")
                department_code = filters.get("department_code")

            max_k = min(int(self.index.ntotal), top_k * 50)
            search_k = min(max_k, top_k * 5)

            while True:
                D, I = self.index.search(xq, search_k)
                ids = I[0]
                scores = D[0]

                valid_ids = [int(i) for i in ids if i != -1]
                if not valid_ids:
                    return []

                qs = Chunk.objects.filter(id__in=valid_ids).select_related("document__department")
                if department_id is not None:
                    qs = qs.filter(document__department_id=department_id)
                if department_code is not None:
                    qs = qs.filter(document__department__code=department_code)

                chunks = list(qs)
                chunk_by_id = {c.id: c for c in chunks}

                results: list[SearchResult] = []
                for chunk_id, score in zip(ids, scores):
                    if chunk_id == -1:
                        continue
                    cid = int(chunk_id)
                    chunk = chunk_by_id.get(cid)
                    if chunk is None:
                        continue
                    results.append(SearchResult(chunk=chunk, score=float(score)))
                    if len(results) >= top_k:
                        return results

                if search_k >= max_k:
                    return results

                search_k = min(max_k, search_k * 2)
