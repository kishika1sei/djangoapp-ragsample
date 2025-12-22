# documents/services/document_service.py
from __future__ import annotations

from collections import Counter

from django.core.files.storage import default_storage
from django.db import transaction
from django.conf import settings

from documents.models import Document, AuditLog
from documents.services.document_ingestion import DocumentIngestionService, IngestionResult
from documents.search_backends.faiss_backend import FaissSearchBackend

from pathlib import Path


def _get_faiss_backend() -> FaissSearchBackend:
    return FaissSearchBackend(index_path=settings.FAISS_INDEX_PATH)


def upload_document(*, actor, uploaded_file, department) -> Document:
    """
    アップロード → Document作成 → ingest → FAISS登録 → 監査ログ
    （同期インジェスト）
    """
    subdir = f"documents/{getattr(department, 'code', department.id)}"
    relative_path = f"{subdir}/{uploaded_file.name}"
    file_path = None
    document = None
    try:
        # 1. ファイル保存
        file_path = default_storage.save(relative_path, uploaded_file)

        # 2. Document作成
        document = Document.objects.create(
            title=uploaded_file.name,
            file_path=file_path,
            department=department,
            uploaded_by=actor,
        )

        # 3. ingest（Chunk + embedding をDBへ）
        result: IngestionResult = DocumentIngestionService.ingest_document(document)

        # 4. FAISSへ即登録
        chunk_ids = list(document.chunks.values_list("id", flat=True))
        faiss = _get_faiss_backend()
        if chunk_ids:
            faiss.index_chunks(chunk_ids)

        # 5. 監査ログ（engine/warningsも残す）
        AuditLog.objects.create(
            actor=actor,
            action=AuditLog.Action.UPLOAD,
            status=AuditLog.Status.SUCCESS,
            document=document,
            department=department,
            message="アップロード時に即インジェスト・即インデックス",
            meta={
                "file_path": file_path,
                "file_ext": Path(file_path).suffix.lower(),
                "chunk_count": result.chunk_count,
                "extract_engine": result.extractor_engine,
                "extract_warnings": result.warnings,
                # ログ肥大化を避けるために必要最小限キーのみ抽出
                "extract_meta": {
                    k: result.extractor_metadata.get(k)
                    for k in ["type", "engine", "warnings", "encoding", "delimiter", "fallback", "csv_header"]
                    if k in (result.extractor_metadata or {})
                },
            },
        )

        return document

    except Exception as e:
        extract_meta = getattr(e, "extract_meta", None)
        # 補償削除（ベストエフォート）
        if document is not None:
            try:
                document.delete()
            except Exception:
                pass
        if file_path:
            try:
                default_storage.delete(file_path)
            except Exception:
                pass

        AuditLog.objects.create(
            actor=actor,
            action=AuditLog.Action.UPLOAD,
            status=AuditLog.Status.FAILED,
            document=None,
            department=department,
            message="アップロード処理失敗",
            meta={
                "filename": getattr(uploaded_file, "name", ""),
                "file_ext": Path(getattr(uploaded_file, "name", "")).suffix.lower(),
                "error": str(e),
                "extract_meta": extract_meta,
                "extract_engine": (extract_meta or {}).get("engine"),
                "extract_warnings": (extract_meta or {}).get("warnings"),
            },
        )
        raise


def delete_document(*, actor, document: Document) -> None:
    """
    ドキュメント削除（Chunk + FAISS + ファイル）→ 監査ログ
    """
    snapshot = {
        "document_id": document.id,
        "title": document.title,
        "file_path": document.file_path,
        "department_id": document.department_id,
    }

    try:
        with transaction.atomic():
            # 1. FAISSからChunk削除
            chunk_ids = list(document.chunks.values_list("id", flat=True))
            if chunk_ids:
                faiss = _get_faiss_backend()
                faiss.delete_chunks(chunk_ids)

            # 2. 物理ファイル削除
            if document.file_path:
                default_storage.delete(document.file_path)

            # 3. Document削除（ChunkはCASCADE）
            document.delete()

        AuditLog.objects.create(
            actor=actor,
            action=AuditLog.Action.DELETE,
            status=AuditLog.Status.SUCCESS,
            document=None,
            department=document.department,
            message="ドキュメント削除",
            meta=snapshot,
        )

    except Exception as e:
        AuditLog.objects.create(
            actor=actor,
            action=AuditLog.Action.DELETE,
            status=AuditLog.Status.FAILED,
            document=document,
            department=document.department,
            message="ドキュメント削除失敗",
            meta=snapshot | {"error": str(e)},
        )
        raise


def reindex_all_documents(*, actor) -> dict:
    total = Document.objects.count()
    success = 0
    failed = 0
    failures: list[dict] = []

    engine_counts = Counter()
    warning_counts = Counter()

    qs = Document.objects.all().order_by("id")

    for doc in qs:
        try:
            with transaction.atomic():
                doc.chunks.all().delete()
                result: IngestionResult = DocumentIngestionService.ingest_document(doc)

            success += 1
            engine_counts[result.extractor_engine] += 1
            for w in (result.warnings or []):
                warning_counts[w] += 1

        except Exception as e:
            failed += 1
            failures.append(
                {"document_id": doc.id, "title": doc.title, "error": str(e)}
            )

    faiss = _get_faiss_backend()
    faiss.rebuild_index()

    
    meta = {
        "scope": "all",
        "total_documents": total,
        "success_documents": success,
        "failed_documents": failed,
        "failures": failures[:50],
        "engine_counts": dict(engine_counts),
        "warning_counts": dict(warning_counts),
    }

    AuditLog.objects.create(
        actor=actor,
        action=AuditLog.Action.REINDEX_ALL,
        status=AuditLog.Status.SUCCESS if failed == 0 else AuditLog.Status.FAILED,
        document=None,
        department=getattr(actor, "department", None),
        message="全件洗い替え（全件再インデックス）を実行",
        meta=meta,
    )

    return meta