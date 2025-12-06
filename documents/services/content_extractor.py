from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from PyPDF2 import PdfReader


@dataclass
class ExtractedContent:
    full_text: str
    pages: Optional[List[str]] = None # PDFなどのページ概念があるもので使用
    num_pages: Optional[int] = None
    metadata: Optional[dict[str,Any]] = None

class ContentExtractor(ABC):
    @abstractmethod
    def can_handle(self,path: Path) -> bool:
        """このExtractorが対象とするファイルかどうかを判定する(拡張子やMIMEタイプなど)"""
        raise NotImplementedError()
    @abstractmethod
    def extract_content(self, path: Path) -> ExtractedContent:
        """ファイルからテキストを抽出する"""
        raise NotImplementedError()

class PDFContentExtractor(ContentExtractor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == '.pdf'
    # TODO: エラーハンドリングを追加する
    # TODO: 画像ベースのPDFには対応していないので現時点では対象外とする処理を追加する
    def extract_content(self, path: Path) -> ExtractedContent:
        reader = PdfReader(str(path))
        meta = getattr(reader, "metadata", None)
        pages_text: List[str] = []
        full_text = ""
        
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        full_text = "\n".join(pages_text)
        num_pages = len(reader.pages)

        # メタデータ取得用のヘルパー関数(PDF専用)
        def safe_get(attr_name: str) -> str:
            if meta is None:
                return ""
            # 属性が無い場合に備えて getattr(..., "") にしておく
            return getattr(meta, attr_name, "") or ""

        return ExtractedContent(
            full_text=full_text.strip(),
            pages=pages_text,
            num_pages=num_pages,
             metadata={
                "author": safe_get("author"),
                "creator": safe_get("creator"),
                "producer": safe_get("producer"),
                "subject": safe_get("subject"),
                "title": safe_get("title"),
                "type": "pdf",
            }
        )
    
# TODO:テキストやcsv、マークダウンのExtractorもここに実装していく予定