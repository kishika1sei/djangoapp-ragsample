from django.db import models
from django.conf import settings

class ChatSession(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name="ユーザー",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_sessions",
    )
    answer_department = models.ForeignKey(
        "accounts.Department",
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_sessions",
        verbose_name="最後に回答した部門",
    )
    title = models.CharField("タイトル",max_length=30,blank=True)
    created_at = models.DateTimeField("作成日時", auto_now_add=True)
    updated_at = models.DateTimeField("更新日時", auto_now=True)
    ended_at = models.DateTimeField("終了日時", null=True, blank=True)

    class Meta:
        verbose_name = "チャットセッション"
        verbose_name_plural = "チャットセッション"
    
    def __str__(self) -> str:
        return self.title or f"Session {self.id}"

class ChatMessage(models.Model):
    class Role(models.TextChoices):
        SYSTEM = "system", "system"
        USER = "user", "user"
        ASSISTANT = "assistant", "assistant"

    session = models.ForeignKey(
        ChatSession,
        verbose_name="チャットセッション",
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField("ロール",max_length=20, choices=Role.choices)
    content = models.TextField("内容")
    created_at = models.DateTimeField("作成日時", auto_now_add=True)

    routing_meta = models.JSONField(
        "ルーティングメタデータ",
        null=True,
        blank=True,
        help_text="LLMルーティング結果(is_business等)をJSONで保存",
    )

    retrieval_meta = models.JSONField(
        "検索メタデータ",
        null=True,
        blank=True,
        help_text="検索スコアやフォールバック状況等をJSONで保存",
    )

    citations = models.JSONField(
        "出典(citations)",
        default=list,
        blank=True,
        help_text="RAGが参照したドキュメント情報(document単位で集約)",

    )

    class Meta:
        verbose_name = "チャットメッセージ"
        verbose_name_plural = "チャットメッセージ"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:20]}"