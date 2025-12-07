from abc import ABC, abstractmethod
from typing import Sequence, TypedDict

class SearchResult(TypedDict):
    chunk_id: int
    score: float

class SearchBackend(ABC):
    @abstractmethod
    def index_chunks(self, chunk_ids: Sequence[int]) -> None:
        """指定された Chunk をインデックスに追加・更新する"""

    @abstractmethod
    def delete_chunks(self, chunk_ids: Sequence[int]) -> None:
        """指定された Chunk をインデックスから削除"""

    @abstractmethod
    def rebuild_index(self) -> None:
        """PostgreSQL の Chunk テーブルを元にインデックス全再構築"""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,  # department_id など
    ) -> list[SearchResult]:
        """類似チャンク検索"""