# documents/admin.py
from django.contrib import admin

from .models import Document, Chunk, AuditLog


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "department", "uploaded_by", "num_page", "created_at")
    list_filter = ("department", "uploaded_by")
    search_fields = ("title",)
    ordering = ("-created_at",)


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "chunk_index", "page", "created_at")
    list_filter = ("document",)
    search_fields = ("content",)
    ordering = ("-created_at",)

@admin.register(AuditLog)
class AuditlogAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "action", "status", "department", "actor", "document", "message")
    list_filter = ("action", "status", "department", "actor")
    search_fields = ("message", "document__title", "actor__username", "actor__email")
    ordering = ("-created_at",)
    date_hierarchy = "created_at"

    # 監査ログは“改ざんできない”前提に寄せる（閲覧中心）
    readonly_fields = ("created_at",)
    actions = None

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False