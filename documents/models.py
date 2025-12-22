from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import ArrayField

from accounts.models import Department

class Document(models.Model):
    title = models.CharField("タイトル(アップロード時のファイル名)", max_length=255)
    file_path = models.CharField("ファイルパス", max_length=500)
    num_page = models.IntegerField("ページ数", null=True, blank=True)
    department = models.ForeignKey(
        Department,
        verbose_name="部門",
        on_delete=models.PROTECT,
        related_name="documents",
    )
    uploaded_by =models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name="アップロードユーザ",
        on_delete=models.PROTECT,
        related_name="uploaded_documents",
    )
    created_at = models.DateTimeField("作成日時", auto_now_add=True)

    class Meta:
        verbose_name = "ドキュメント"
        verbose_name_plural = "ドキュメント"
    
    def __str__(self) -> str:
        return self.title

class Chunk(models.Model):
    document = models.ForeignKey(
        Document,
        verbose_name="ドキュメント",
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_index = models.IntegerField("チャンク番号")
    page = models.IntegerField("元ページ番号", null=True, blank=True)
    content = models.TextField("テキスト")
    # Embedding
    embedding = ArrayField(
        base_field=models.FloatField(),
        verbose_name="埋め込みベクトル",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField("作成日時", auto_now_add=True)

    class Meta:
        verbose_name = "チャンク"
        verbose_name_plural = "チャンク"
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
        ]


    
    def __str__(self) -> str:
        return f"{self.document_id}-{self.chunk_index}"
    

class AuditLog(models.Model):
    class Action(models.TextChoices):
        UPLOAD = "UPLOAD", "アップロード"
        DELETE = "DELETE", "削除"
        REINDEX = "REINDEX", "再インデックス"
        REINDEX_ALL = "REINDEX_ALL", "全件再インデックス"

    class Status(models.TextChoices):
        SUCCESS = "SUCCESS", "成功"
        FAILED = "FAILED", "失敗"

    action = models.CharField("操作", max_length=20, choices=Action.choices)
    status = models.CharField("結果", max_length=10, choices=Status.choices, default=Status.SUCCESS)

    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name="実行ユーザ",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_logs",
    )

    # 対象はまず Document に限定（必要になったら後で汎用化）
    document = models.ForeignKey(
        Document,
        verbose_name="対象ドキュメント",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_logs",
    )

    # 操作時点の部門の証跡（後で Document が消えても追える）
    department = models.ForeignKey(
        Department,
        verbose_name="部門",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_logs",
    )

    message = models.CharField("メッセージ", max_length=255, blank=True, default="")

    # JSONで拡張（chunk_count、duration_ms、filename、error 等）
    meta = models.JSONField("詳細", default=dict, blank=True)

    created_at = models.DateTimeField("作成日時", auto_now_add=True)

    class Meta:
        verbose_name = "操作ログ"
        verbose_name_plural = "操作ログ"
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["action", "created_at"]),
            models.Index(fields=["department", "created_at"]),
            models.Index(fields=["document", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.created_at} {self.action} {self.status}"