# documents/tests.py
from __future__ import annotations

from django.test import TestCase
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from django.contrib.auth import get_user_model
from documents.models import Department, Document, Chunk
from documents.search_backends.faiss_backend import FaissSearchBackend

User = get_user_model()

class FaissSearchBackendDepartmentFilterTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # 部門
        cls.dep_fin = Department.objects.create(name="経理", code="finance")
        cls.dep_hr = Department.objects.create(name="人事総務", code="hr")
        cls.user = User.objects.create_user(
            username="testuser1",
            password="pass12345",
        )

        # ダミーでドキュメント作成
        cls.doc_fin = Document.objects.create(
            title="経理規程",
            file_path="dummy/fin.pdf",
            department=cls.dep_fin,
            uploaded_by = cls.user,
        )
        cls.doc_hr = Document.objects.create(
            title="人事規程",
            file_path="dummy/hr.pdf",
            department=cls.dep_hr,
            uploaded_by = cls.user,
        )

        # Chunk
        cls.chunk_fin = Chunk.objects.create(
            document=cls.doc_fin,
            content="経費精算のルール",
            chunk_index=0,
            page=1,
        )
        cls.chunk_hr = Chunk.objects.create(
            document=cls.doc_hr,
            content="有給休暇のルール",
            chunk_index=0,
            page=1,
        )

    def test_search_filters_by_department_code(self):
        """
        financeフィルタ時にhrが混ざらないこと
        """
        with TemporaryDirectory() as d:
            index_path = Path(d) / "index.faiss"
            backend = FaissSearchBackend(index_path=index_path, dimension=3)

            # indexへ登録（テストではEmbeddingを使わず、ベクトルを手で入れるのが安定）
            # query=[1,0,0] に対して hr=1.0, finance=0.8 のスコアになるようにする
            vectors = np.array([
                [0.8, 0.0, 0.0],  # finance
                [1.0, 0.0, 0.0],  # hr
            ], dtype="float32")
            ids = np.array([self.chunk_fin.id, self.chunk_hr.id], dtype="int64")
            backend.index.add_with_ids(vectors, ids)

            query = [1.0, 0.0, 0.0]

            # 全社（フィルタなし）: 2件返る想定
            all_results = backend.search(query_embedding=query, top_k=2, filters=None)
            self.assertEqual(len(all_results), 2)

            # financeフィルタ: financeだけ返る想定
            fin_results = backend.search(
                query_embedding=query,
                top_k=2,
                filters={"department_code": "finance"},
            )
            self.assertGreaterEqual(len(fin_results), 1)
            self.assertTrue(all(r.chunk.document.department.code == "finance" for r in fin_results))

    def test_search_expands_candidates_until_it_finds_filtered_hits(self):
        """
        上位が他部門で埋まっても、search_kを増やしてフィルタ部門を拾えること
        """
        with TemporaryDirectory() as d:
            index_path = Path(d) / "index.faiss"
            backend = FaissSearchBackend(index_path=index_path, dimension=3)

            # hrの高スコアchunkを10個作る（financeをランキング下位へ追いやる）
            hr_chunks = []
            for i in range(10):
                hr_chunks.append(Chunk.objects.create(
                    document=self.doc_hr,
                    content=f"人事ダミー{i}",
                    chunk_index=i + 1,
                    page=1,
                ))

            # ベクトル: hr=1.0, finance=0.6
            vecs = []
            ids = []

            # hrが上位を埋める
            for c in hr_chunks:
                vecs.append([1.0, 0.0, 0.0])
                ids.append(c.id)

            # financeを下位に置く
            vecs.append([0.6, 0.0, 0.0])
            ids.append(self.chunk_fin.id)

            backend.index.add_with_ids(np.array(vecs, dtype="float32"), np.array(ids, dtype="int64"))

            query = [1.0, 0.0, 0.0]

            # top_k=1 だと最初の search_k=5 では hrしか見えず、フィルタ後は空になりがちなので確認。
            fin_results = backend.search(
                query_embedding=query,
                top_k=1,
                filters={"department_code": "finance"},
            )
            self.assertEqual(len(fin_results), 1)
            self.assertEqual(fin_results[0].chunk.document.department.code, "finance")

class FaissBackendMoreTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="testuser", password="pass12345")
        cls.dep_hr = Department.objects.create(name="人事総務", code="hr")

        cls.doc_hr = Document.objects.create(
            title="人事規程",
            file_path="dummy/hr.pdf",
            department=cls.dep_hr,
            uploaded_by=cls.user,
        )

        cls.chunk_hr = Chunk.objects.create(
            document=cls.doc_hr,
            content="有給休暇のルール",
            chunk_index=0,
            page=1,
        )

    def test_search_returns_empty_when_index_empty(self):
        """インデックス空なら常に[]"""
        with TemporaryDirectory() as d:
            backend = FaissSearchBackend(index_path=Path(d) / "index.faiss", dimension=3)
            res = backend.search(query_embedding=[1.0, 0.0, 0.0], top_k=5, filters=None)
            self.assertEqual(res, [])

    def test_search_unknown_department_code_returns_empty(self):
        """存在しないdepartment_codeを指定したら混ざらず0件（安全側）"""
        with TemporaryDirectory() as d:
            backend = FaissSearchBackend(index_path=Path(d) / "index.faiss", dimension=3)
            backend.index.add_with_ids(
                np.array([[1.0, 0.0, 0.0]], dtype="float32"),
                np.array([self.chunk_hr.id], dtype="int64"),
            )

            res = backend.search(
                query_embedding=[1.0, 0.0, 0.0],
                top_k=5,
                filters={"department_code": "finance"},
            )
            self.assertEqual(len(res), 0)

    def test_search_never_exceeds_top_k(self):
        """top_k以上を返さない（契約）"""
        with TemporaryDirectory() as d:
            backend = FaissSearchBackend(index_path=Path(d) / "index.faiss", dimension=3)

            # hrチャンクを追加で作成
            extra_ids = []
            for i in range(10):
                c = Chunk.objects.create(
                    document=self.doc_hr,
                    content=f"ダミー{i}",
                    chunk_index=i + 1,
                    page=1,
                )
                extra_ids.append(c.id)

            vecs = np.array([[1.0, 0.0, 0.0]] * len(extra_ids), dtype="float32")
            backend.index.add_with_ids(vecs, np.array(extra_ids, dtype="int64"))

            res = backend.search(query_embedding=[1.0, 0.0, 0.0], top_k=3, filters={"department_code": "hr"})
            self.assertLessEqual(len(res), 3)