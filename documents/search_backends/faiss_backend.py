from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from django.conf import settings

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

        # dimension が指定されていなければ、EmbeddingService から推定
        if dimension is None:
            sample_vec = self.embedding_service.embed_chunks(["__probe__"])[0]
            dimension = len(sample_vec)

        self.dimension = dimension
        self.index = self._load_or_create_index()

    # --- インデックス生成まわり ---

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

    # ---インデックスの削除---
    def delete_chunks(self, chunk_ids: Sequence[int]) -> None:
        """指定された Chunk をインデックスから削除"""

    # ---インデックスの全再構築---
    def rebuild_index(self) -> None:
        """PostgreSQL の Chunk テーブルを元にインデックス全再構築"""

    # ---インデックスの検索---
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,  # department_id など
    ) -> list[SearchResult]:
        """類似チャンク検索"""