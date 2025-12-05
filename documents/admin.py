# documents/admin.py
from django.contrib import admin

from .models import Document, Chunk


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "department", "uploaded_by", "num_page", "created_at")
    list_filter = ("department", "uploaded_by")
    search_fields = ("title",)


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "chunk_index", "page", "created_at")
    list_filter = ("document",)
    search_fields = ("content",)
