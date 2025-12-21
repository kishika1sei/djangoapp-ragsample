from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, Tuple

from django.core.files.storage import default_storage

from documents.models import Document, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documents.services.embedding_service import EmbeddingService
from documents.services.content_extractor import get_extractor, ExtractedContent


# Embeddingサービスの初期化（現状踏襲）
embedding_service = EmbeddingService()


@dataclass
class IngestionResult:
    chunk_count: int
    extractor_engine: str
    warnings: List[str]
    extractor_metadata: dict[str, Any]
    num_pages: Optional[int] = None


class DocumentIngestionService:
    # 既存値を踏襲しつつ、種別で splitter を使い分け
    pdf_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "、", " ", ""],
        chunk_size=300,
        chunk_overlap=80,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "# ", "## ", "### ", "。", "、", " ", ""],
        chunk_size=300,
        chunk_overlap=80,
    )
    generic_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "、", " ", ""],
        chunk_size=300,
        chunk_overlap=80,
    )

    CSV_LINES_PER_CHUNK_DEFAULT = 20

    @classmethod
    def ingest_document(cls, document: Document) -> IngestionResult:
        """一つのドキュメントに対して、抽出→チャンク分割→Embedding→チャンク保存を行う"""
        # 1. storage上のpath解決
        fs_path_str = default_storage.path(document.file_path)
        file_path = Path(fs_path_str)

        # 2. Extractor選択して抽出
        extractor = get_extractor(file_path)
        content: ExtractedContent = extractor.extract_content(file_path)
        extract_meta = content.metadata or {}
        engine = str(extract_meta.get("engine") or "unknown")
        warnings = list(extract_meta.get("warnings") or [])

        # スキャンPDF疑いは取り込み対象外とする（現状はOCR未対応のため）
        if extract_meta.get("type") == "pdf":
            if ("no_text_extracted" in warnings) or ("image_pdf_suspected" in warnings):
                raise IngestionError(
                    "スキャンPDF（OCR未対応）の可能性が高いため取り込み不可です。",
                    extract_meta=extract_meta,
                )

        doc_type = str(extract_meta.get("type") or "unknown")

        # PDFページ数は Document.num_page に反映（既存の字段を活用）
        if content.num_pages is not None:
            document.num_page = content.num_pages
            document.save(update_fields=["num_page"])

        # 3. チャンク化（pagesがあればページ単位、なければfull_text）
        page_chunk_pairs: List[Tuple[Optional[int], str]] = []

        if content.pages:
            for page_idx, page_text in enumerate(content.pages, start=1):
                page_text = (page_text or "").strip()
                if not page_text:
                    continue
                for chunk_text in cls.pdf_splitter.split_text(page_text):
                    ct = chunk_text.strip()
                    if ct:
                        page_chunk_pairs.append((page_idx, ct))
        else:
            full_text = (content.full_text or "").strip()
            if not full_text:
                raise IngestionError(
                    "抽出テキストが空のため、チャンクを生成できません。",
                    extract_meta=extract_meta,
                )

            if doc_type == "csv":
                page_chunk_pairs.extend(cls._chunk_csv(full_text, extract_meta))
            elif doc_type == "text":
                for chunk_text in cls.text_splitter.split_text(full_text):
                    ct = chunk_text.strip()
                    if ct:
                        page_chunk_pairs.append((None, ct))
            else:
                for chunk_text in cls.generic_splitter.split_text(full_text):
                    ct = chunk_text.strip()
                    if ct:
                        page_chunk_pairs.append((None, ct))

        if not page_chunk_pairs:
            raise IngestionError(
                "チャンクが生成されませんでした（抽出結果が空/分割不能）。",
                extract_meta=extract_meta,
            )
        

        # 4. 埋め込み
        chunk_texts = [text for (_, text) in page_chunk_pairs]
        vectors = embedding_service.embed_chunks(chunk_texts)

        # 5. Chunkをbulk_create
        chunk_objs: list[Chunk] = []
        for idx, ((page_idx, chunk_text), vec) in enumerate(zip(page_chunk_pairs, vectors)):
            chunk_objs.append(
                Chunk(
                    document=document,
                    chunk_index=idx,
                    page=page_idx,  # Noneも許容される前提（DBがnull可でなければ0等に寄せてください）
                    content=chunk_text,
                    embedding=vec,
                )
            )

        Chunk.objects.bulk_create(chunk_objs)

        return IngestionResult(
            chunk_count=len(chunk_objs),
            extractor_engine=engine,
            warnings=warnings,
            extractor_metadata=extract_meta,
            num_pages=content.num_pages,
        )

    @classmethod
    def _chunk_csv(cls, full_text: str, meta: dict[str, Any]) -> List[Tuple[Optional[int], str]]:
        """
        full_text: 1行=1レコードの key=value 正規化済み想定
        - ヘッダを各チャンクに付与
        - N行ずつまとめてチャンク化（表構造を壊しにくい）
        """
        header = meta.get("csv_header") or []
        header_line = f"CSVヘッダ: {', '.join(header)}" if header else ""

        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        if not lines:
            return []

        n = int(meta.get("rows_per_chunk_hint") or cls.CSV_LINES_PER_CHUNK_DEFAULT)

        pairs: List[Tuple[Optional[int], str]] = []
        for i in range(0, len(lines), n):
            block = "\n".join(lines[i : i + n])
            chunk_text = f"{header_line}\n{block}".strip() if header_line else block
            pairs.append((None, chunk_text))
        return pairs
# -----------------------------
# 例外クラス
# -----------------------------
class IngestionError(Exception):
    def __init__(self, message: str, *, extract_meta: dict[str, Any] | None = None):
        super().__init__(message)
        self.extract_meta = extract_meta or {}