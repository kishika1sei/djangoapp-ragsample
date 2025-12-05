from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import ArrayField

from accounts.models import Department
# Create your models here.

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