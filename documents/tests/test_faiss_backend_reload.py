import hashlib
import os
import time
from tempfile import TemporaryDirectory

from django.test import TestCase

from documents.models import Document, Chunk, Department
from documents.search_backends.faiss_backend import FaissSearchBackend
from django.contrib.auth import get_user_model

# python manage.py test documents.tests.test_faiss_backend_reload
class DummyEmbeddingService:
    """
    外部APIに依存しない決定論埋め込み。
    - 同じテキスト => 同じベクトル
    - dimension は小さめでOK（FAISS動作確認が目的）
    """
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 0..255 を 0..1 にスケールして dim 個取り出す
        return [h[i] / 255.0 for i in range(self.dim)]

    def embed_chunks(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_text(self, text: str) -> list[float]:
        return self._vec(text)


class FaissSearchBackendReloadTests(TestCase):
    def setUp(self) -> None:
        self.dept = Department.objects.create(code="it", name="IT")

        User = get_user_model()
        # username/email 等はあなたの User モデル要件に合わせて調整
        self.user = User.objects.create_user(
            username="tester",
            password="pass1234",
        )

        self.doc = Document.objects.create(
            title="IT Guide",
            department=self.dept,
            uploaded_by=self.user,   # ★これが必須
        )

    def test_search_auto_reload_after_external_rebuild(self):
        """
        別インスタンス（別プロセス相当）が rebuild して index ファイルを書き換えたとき、
        既存インスタンスが search() 時に mtime 変更を検知して自動リロードできることを検証する。
        """
        emb = DummyEmbeddingService(dim=8)

        with TemporaryDirectory() as td:
            index_path = os.path.join(td, "chunks.index")

            # 初期データ：VPN
            Chunk.objects.create(document=self.doc, chunk_index=0, page=0, content="VPN接続方法の手順")
            Chunk.objects.create(document=self.doc, chunk_index=1, page=0, content="VPNトラブルシュート")

            backend_a = FaissSearchBackend(index_path=index_path, embedding_service=emb, dimension=8)
            backend_a.rebuild_index()

            # AでVPN検索がヒットすること
            q_vpn = emb.embed_text("VPN 接続")
            r1 = backend_a.search(q_vpn, top_k=3, filters=None)
            self.assertTrue(len(r1) > 0)
            self.assertIn("VPN", r1[0].chunk.content)

            # mtime解像度（Windows等）対策：1秒またぐ
            time.sleep(1.1)

            # データ更新：有給
            Chunk.objects.create(document=self.doc, chunk_index=2, page=0, content="有給休暇の申請手順")
            Chunk.objects.create(document=self.doc, chunk_index=3, page=0, content="有給休暇の付与日数")

            # 別インスタンスBがrebuildしてディスクのindexを更新（別プロセス相当）
            backend_b = FaissSearchBackend(index_path=index_path, embedding_service=emb, dimension=8)
            backend_b.rebuild_index()

            # Aが自動リロードできていれば、有給検索がヒットする
            q_leave = emb.embed_text("有給休暇 申請")
            r2 = backend_a.search(q_leave, top_k=5, filters=None)
            self.assertTrue(len(r2) > 0)
            # 期待：有給系のチャンクが含まれる
            self.assertTrue(any("有給" in x.chunk.content for x in r2))

            # 追従できているなら ntotal も DB件数と一致するのが自然
            self.assertEqual(backend_a.index.ntotal, Chunk.objects.count())

    def test_rebuild_does_not_overwrite_when_no_chunks(self):
        """
        （任意・強い）
        chunk_count==0 の場合に rebuild を abort して空index上書きを防ぐガードがある前提で、
        既存indexが破壊されないことを検証する。

        ※もしあなたが rebuild_index() に 'chunk_count==0ならreturn/raise' を
          まだ入れていない場合、このテストは失敗します。
        """
        emb = DummyEmbeddingService(dim=8)

        with TemporaryDirectory() as td:
            index_path = os.path.join(td, "chunks.index")

            # 初期データ：VPN
            Chunk.objects.create(document=self.doc, chunk_index=0, page=0, content="VPN接続方法の手順")

            backend_a = FaissSearchBackend(index_path=index_path, embedding_service=emb, dimension=8)
            backend_a.rebuild_index()

            size_before = os.path.getsize(index_path)
            self.assertTrue(size_before > 0)

            # 全チャンク削除（異常系）
            Chunk.objects.all().delete()

            time.sleep(1.1)
            backend_b = FaissSearchBackend(index_path=index_path, embedding_service=emb, dimension=8)
            backend_b.rebuild_index()

            # 空上書きをしないなら、ファイルサイズが極端に小さくならない（簡易判定）
            size_after = os.path.getsize(index_path)
            self.assertGreaterEqual(size_after, size_before)

            # backend_a は（少なくとも）落ちずに検索できる
            q = emb.embed_text("VPN")
            r = backend_a.search(q, top_k=3, filters=None)
            # データ自体は消えているので DB一致を求めない（ここでは“破壊しない”が目的）
            self.assertIsInstance(r, list)
