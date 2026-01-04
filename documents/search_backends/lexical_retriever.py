from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from django.contrib.postgres.search import TrigramSimilarity

from documents.models import Chunk
from .base import SearchResult  


@dataclass
class LexicalConfig:
    # 最初は緩め（弾きすぎ防止）。必要なら後で調整
    similarity_threshold: float = 0.05
    use_threshold: bool = True


class LexicalRetriever:
    """
    PostgreSQL(pg_trgm) を使った trigram ベースの文字検索。

    - SearchBackend は継承しない（embedding前提の契約と不整合）
    - DB側インデックスで高速化されるため、Faissのような index 運用APIも基本不要
    """

    def __init__(self, config: Optional[LexicalConfig] = None) -> None:
        self.config = config or LexicalConfig()

    def search_text(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        qs = Chunk.objects.select_related("document__department")

        department_id = None
        department_code = None
        if filters:
            department_id = filters.get("department_id")
            department_code = filters.get("department_code")

        if department_id is not None:
            qs = qs.filter(document__department_id=department_id)
        if department_code is not None:
            qs = qs.filter(document__department__code=department_code)

        # trigram similarity を付与して降順ソート
        qs = qs.annotate(similarity=TrigramSimilarity("content", query_text)).order_by("-similarity", "id")

        if self.config.use_threshold:
            qs = qs.filter(similarity__gte=self.config.similarity_threshold)

        chunks = list(qs[:top_k]) # ここで評価クエリ実行

        results: list[SearchResult] = []
        for c in chunks:
            sim = getattr(c, "similarity", None)
            score = float(sim) if sim is not None else 0.0
            results.append(SearchResult(chunk=c, score=score))
        return results
