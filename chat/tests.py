# chat/tests.py
from __future__ import annotations

from django.test import TestCase
from unittest.mock import Mock

from chat.services.rag_chat import RAGChatService  
from documents.search_backends.base import SearchResult


class DummyChunk:
    def __init__(self, dept_code: str):
        self.document = type("Doc", (), {"department": type("Dept", (), {"code": dept_code})()})()
        self.content = "dummy"


class DummyRoute:
    primary_department = "finance"
    secondary_departments = ["hr"]


class RAGChatFallbackTests(TestCase):
    def test_primary_secondary_company_fallback_order(self):
        """
        primaryが閾値未満 → secondaryが閾値以上 → secondaryで確定、を確認
        """
        # search_backend.search の戻りを department_code に応じて変える
        search_backend = Mock()

        # financeは弱い（0.3）→ 閾値未満
        weak_results = [SearchResult(chunk=DummyChunk("finance"), score=0.3)]
        # hrは強い（0.9）→ 閾値以上
        strong_results = [SearchResult(chunk=DummyChunk("hr"), score=0.9)]

        def side_effect(*, query_embedding, top_k, filters=None):
            if filters and filters.get("department_code") == "finance":
                return weak_results
            if filters and filters.get("department_code") == "hr":
                return strong_results
            return []

        search_backend.search.side_effect = side_effect

        embedding_service = Mock()
        llm_client = Mock()
        router = Mock()

        svc = RAGChatService(search_backend, embedding_service, llm_client, router=router)

        # 直接 _search_with_fallback を叩く（chat全体よりテストが単純）
        results, meta = svc._search_with_fallback(query_embedding=[0.0], route=DummyRoute(), top_k=5)

        self.assertEqual(meta["scope_used"], "hr")
        self.assertFalse(meta["fallback_triggered"])
        self.assertAlmostEqual(meta["top_score"], 0.9, places=5)


class _DummyResult:
    def __init__(self, score: float):
        self.score = score
        self.chunk = type("Chunk", (), {"content": "x", "document": None})()


class _DummyRoute:
    def __init__(self, *, needs_clarification=False, is_business=True,
                 primary="finance", secondary=None, clarifying=""):
        self.needs_clarification = needs_clarification
        self.is_business = is_business
        self.primary_department = primary
        self.secondary_departments = secondary or []
        self.clarifying_question = clarifying

    def model_dump(self):
        return {
            "needs_clarification": self.needs_clarification,
            "is_business": self.is_business,
            "primary_department": self.primary_department,
            "secondary_departments": self.secondary_departments,
            "clarifying_question": self.clarifying_question,
        }


class RAGChatBranchingTests(TestCase):
    def setUp(self):
        self.search_backend = Mock()
        self.embedding_service = Mock()
        self.llm_client = Mock()
        self.router = Mock()

        self.svc = RAGChatService(
            self.search_backend, self.embedding_service, self.llm_client, router=self.router
        )

        self.session = Mock()

    def test_needs_clarification_short_circuits(self):
        """needs_clarificationなら検索・LLMに進まない"""
        self.router.route.return_value = _DummyRoute(
            needs_clarification=True,
            clarifying="確認です。対象の制度名は何ですか？",
        )

        answer, meta = self.svc.chat(session=self.session, user_message="休暇について教えて")
        self.assertIn("確認です", answer)
        self.assertEqual(meta["reason"], "needs_clarification")

        self.embedding_service.embed_text.assert_not_called()
        self.search_backend.search.assert_not_called()
        self.llm_client.complete.assert_not_called()

    def test_not_business_short_circuits(self):
        """業務外判定なら検索・LLMに進まない"""
        self.router.route.return_value = _DummyRoute(is_business=False)

        answer, meta = self.svc.chat(session=self.session, user_message="おすすめのラーメンは？")
        self.assertEqual(meta["reason"], "not_business")

        self.embedding_service.embed_text.assert_not_called()
        self.search_backend.search.assert_not_called()
        self.llm_client.complete.assert_not_called()


class RAGChatFallbackMoreTests(TestCase):
    def test_fallback_to_company_when_primary_and_secondary_fail(self):
        """primary/secondaryが弱いときcompanyに落ちる"""
        search_backend = Mock()

        def side_effect(*, query_embedding, top_k, filters=None):
            # finance, hr は空
            if filters and filters.get("department_code") in ("finance", "hr"):
                return []
            # companyでヒット
            if filters is None:
                return [_DummyResult(0.9)]
            return []

        search_backend.search.side_effect = side_effect

        svc = RAGChatService(search_backend, Mock(), Mock(), router=Mock())
        route = _DummyRoute(primary="finance", secondary=["hr"])

        results, meta = svc._search_with_fallback(query_embedding=[0.0], route=route, top_k=5)
        self.assertEqual(meta["scope_used"], "company")
        self.assertTrue(meta["fallback_triggered"])
        self.assertEqual(len(results), 1)

    def test_threshold_boundary_primary_equals_threshold(self):
        """閾値ちょうど(0.55)は primary で確定する（>= 判定の契約）"""
        search_backend = Mock()

        def side_effect(*, query_embedding, top_k, filters=None):
            if filters and filters.get("department_code") == "finance":
                return [_DummyResult(0.55)]  # ちょうど
            if filters and filters.get("department_code") == "hr":
                return [_DummyResult(0.9)]  # secondaryは強いが呼ばれないのが理想
            if filters is None:
                return [_DummyResult(0.9)]
            return []

        search_backend.search.side_effect = side_effect

        svc = RAGChatService(search_backend, Mock(), Mock(), router=Mock())
        route = _DummyRoute(primary="finance", secondary=["hr"])

        results, meta = svc._search_with_fallback(query_embedding=[0.0], route=route, top_k=5)
        self.assertEqual(meta["scope_used"], "finance")
        self.assertFalse(meta["fallback_triggered"])
        self.assertAlmostEqual(meta["top_score"], 0.55, places=6)


class PromptBuildTests(TestCase):
    def test_prompt_ends_with_question(self):
        """末尾がQuestionで終わる（アンカーが壊れない）"""
        svc = RAGChatService(Mock(), Mock(), Mock(), router=Mock())
        prompt = svc._build_prompt(
            system_prompt="SYS",
            history=[],
            context="CTX",
            user_message="経費精算の締め日は？",
        )
        self.assertIn("[Question]", prompt)
        self.assertTrue(prompt.strip().endswith("経費精算の締め日は？"))