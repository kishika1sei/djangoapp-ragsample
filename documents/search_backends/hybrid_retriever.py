from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import SearchBackend, SearchResult
from .lexical_retriever import LexicalRetriever


@dataclass
class RRFConfig:
    # RRFの定数（一般に 50〜60 あたりから開始）
    k: int = 60
    # 融合前に多めに拾う（上位だけだと融合の旨味が減る）
    vector_fetch_mult: int = 8
    lexical_fetch_mult: int = 8


class HybridRetriever:
    """
    Vector（FAISS） + Lexical（pg_trgm）を統合し、RRFで最終ランキングを作る。
    """

    def __init__(
        self,
        *,
        vector_backend: SearchBackend,
        lexical: LexicalRetriever,
        config: Optional[RRFConfig] = None,
    ) -> None:
        self.vector_backend = vector_backend
        self.lexical = lexical
        self.config = config or RRFConfig()

    def retrieve(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[SearchResult], dict]:
        vector_k = max(top_k, top_k * self.config.vector_fetch_mult)
        lexical_k = max(top_k, top_k * self.config.lexical_fetch_mult)

        vector_results = self.vector_backend.search(
            query_embedding=query_embedding,
            top_k=vector_k,
            filters=filters,
        )
        lexical_results = self.lexical.search_text(
            query_text,
            top_k=lexical_k,
            filters=filters,
        )

        # --- RRF融合（順位ベース） ---
        # chunk_id -> accumulator
        # aggregate
        agg: dict[int, dict] = {}

        def add_rrf(results: list[SearchResult], source: str) -> None:
            for rank, r in enumerate(results, start=1):
                cid = int(r.chunk.id)
                rrf_score = 1.0 / (self.config.k + rank)
                if cid not in agg:
                    agg[cid] = {
                        "chunk": r.chunk,
                        "rrf": 0.0,
                        "vector_rank": None,
                        "lexical_rank": None,
                        "vector_score": None,
                        "lexical_score": None,
                    }
                agg[cid]["rrf"] += rrf_score
                if source == "vector":
                    agg[cid]["vector_rank"] = rank
                    agg[cid]["vector_score"] = float(r.score)
                else:
                    agg[cid]["lexical_rank"] = rank
                    agg[cid]["lexical_score"] = float(r.score)

        add_rrf(vector_results, "vector")
        add_rrf(lexical_results, "lexical")

        if not agg:
            return [], {
                "engine": "hybrid",
                "fusion": {"method": "rrf", "k": self.config.k},
                "hit_count": 0,
                "top_score": None,
                "vector_hit_count": len(vector_results),
                "lexical_hit_count": len(lexical_results),
            }

        merged = sorted(
            agg.values(),
            key=lambda x: (float(x["rrf"]), -(x["vector_rank"] or 10**9), -(x["lexical_rank"] or 10**9)),
            reverse=True,
        )[:top_k]

        final_results = [SearchResult(chunk=m["chunk"], score=float(m["rrf"])) for m in merged]

        retrieval_meta = {
            "engine": "hybrid",
            "fusion": {"method": "rrf", "k": self.config.k},
            "hit_count": len(final_results),
            # ここはRRFスコア（FAISSの0.55閾値とは無関係）
            "top_score": float(final_results[0].score) if final_results else None,
            "vector_hit_count": len(vector_results),
            "lexical_hit_count": len(lexical_results),
            "vector_top_score": float(vector_results[0].score) if vector_results else None,
            "lexical_top_score": float(lexical_results[0].score) if lexical_results else None,
            "vector_hit_ids": [int(r.chunk.id) for r in vector_results[: min(10, len(vector_results))]],
            "lexical_hit_ids": [int(r.chunk.id) for r in lexical_results[: min(10, len(lexical_results))]],
            "final_hit_ids": [int(r.chunk.id) for r in final_results],
        }

        return final_results, retrieval_meta
