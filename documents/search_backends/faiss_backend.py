from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .base import SearchBackend, SearchResult  # あなたの ABC を置いている場所
from documents.models import Chunk
from documents.services.embedding_service import EmbeddingService  # さっきのクラス

class FaissSearchBackend(SearchBackend):
    def __init__(
        self,
        index_path: str | Path,
        embedding_service: EmbeddingService | None = None,
        dimension: int | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.embedding_service = embedding_service or EmbeddingService()

        # dimension が指定されていなければ、ダミー文字列をベクトル化したものをEmbeddingService から推定
        if dimension is None:
            sample_vec = self.embedding_service.embed_chunks(["__probe__"])[0]
            dimension = len(sample_vec)

        self.dimension = dimension
        self.index = self._load_or_create_index()

    # --- インデックス生成周り ---

    def _create_empty_index(self) -> faiss.IndexIDMap2:
        """
        空の IndexIDMap2 を生成する。
        Chunk.id をそのまま ID に使う前提。
        """
        base_index = faiss.IndexFlatIP(self.dimension)  # 内積
        index = faiss.IndexIDMap2(base_index)
        return index

    def _load_or_create_index(self) -> faiss.IndexIDMap2:
        """
        既存のインデックスがあれば読み込み、なければ新規作成。
        """
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            # 読み込んだ index が IDMap2 でない場合は注意（ここでは前提として省略）
            return index  # type: ignore[return-value]
        else:
            index = self._create_empty_index()
            self._save_index(index)
            return index

    def _save_index(self, index: faiss.Index | None = None) -> None:
        """
        インデックスをディスクに保存。
        """
        index = index or self.index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

    # ---インデックスの登録---
    def index_chunks(self, chunk_ids: Sequence[int]) -> None:
        """指定された Chunk をインデックスに追加・更新する"""
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return
        
        # 対象チャンクを登録
        chunks = list(
            Chunk.objects.filter(id__in=chunk_ids).select_related("document")
        )
        if not chunks:
            return
        
        # テキスト & ID 抽出
        texts: list[str] = [c.content for c in chunks]
        ids_np = np.array([c.id for c in chunks],dtype="int64")

        # 埋め込み生成
        embeddings: list[list[float]] = self.embedding_service.embed_chunks(texts)
        vectors = np.array(embeddings,dtype="float32")

        # 既存の同一IDがあれば削除(更新に対応するため)
        try:
            selector = faiss.IDSelectorBatch(ids_np)
            self.index.remove_ids(selector)
        except Exception:
            # TODO:後でここに例外処理を追加する
            # 初期状態ではIDがない場合があるので保険をかけておく
            pass

        # 新しいベクトルを登録
        self.index.add_with_ids(vectors,ids_np)

        # インデックスを保存
        self._save_index()
    # ---インデックスの削除---
    def delete_chunks(self, chunk_ids: Sequence[int]) -> None:
        """指定された Chunk をインデックスから削除"""
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return
        
        ids_np = np.array(chunk_ids,dtype="int64")
        selector = faiss.IDSelectorBatch(ids_np)
        self.index.remove_ids(selector)
        self._save_index()

    # ---インデックスの全再構築---
    def rebuild_index(self) -> None:
        """PostgreSQL の Chunk テーブルを元にインデックス全再構築"""
        # まず空のインデックスを作り直す
        self.index = self._create_empty_index()
        qs = Chunk.objects.all().order_by("id")

        batch_size = 256
        offset = 0
        while True:
            batch = list(qs[offset : offset + batch_size])
            if not batch:
                break

            texts = [c.content for c in batch]
            ids_np = np.array([c.id for c in batch], dtype="int64")

            embeddings = self.embedding_service.embed_chunks(texts)
            vectors = np.array(embeddings, dtype="float32")

            self.index.add_with_ids(vectors,ids_np)

            offset += batch_size
        
        self._save_index()

    # ---インデックスの検索---
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,  # department_id など
    ) -> list[SearchResult]:
        """類似チャンク検索"""
        if self.index.ntotal == 0:
            return[]
        
        xq = np.array([query_embedding], dtype="float32")

        # 絞り込みがあることを考えて、少し多めにとる
        search_k = top_k * 5
        # 上位５件を取り出す
        D, I = self.index.search(xq, search_k)
        
        # Dは類似度スコア配列、IはID配列(各クエリに対する候補IDの配列)[0]で最上位のものだけを取り出す。
        ids = I[0]
        scores = D[0]

        # ヒットしない部分の -1を除外
        valid_ids: list[int] = [int(i) for i in ids if i != -1]
        if not valid_ids:
            return []
        
        # 対応するChunkを取得
        chunks = list(
            Chunk.objects.filter(id__in=valid_ids)
            .select_related("document__department")
        )
        chunk_by_id = {c.id: c for c in chunks}

        department_id = None
        if filters:
            department_id = filters.get("department_id")

        results: list[SearchResult] = []

        for chunk_id, score in zip(ids, scores):
            if chunk_id == -1:
                continue

            cid = int(chunk_id)
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                continue

            # 部門フィルタ
            if department_id is not None:
                if chunk.document.department_id != department_id:
                    continue

            results.append(
                {
                    "chunk_id": cid,
                    "score": float(score),
                }
            )
            if len(results) >= top_k:
                break

        return results