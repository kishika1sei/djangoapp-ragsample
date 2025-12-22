from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple
import csv

from PyPDF2 import PdfReader
import warnings as pywarnings
from PyPDF2.errors import PdfReadWarning


try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


REPLACEMENT_CHAR = "\uFFFD"  # �文字化け


@dataclass
class ExtractedContent:
    full_text: str
    pages: Optional[List[str]] = None  # PDFなどのページ概念があるもので使用
    num_pages: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Noneを避ける


class ContentExtractor(ABC):
    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """このExtractorが対象とするファイルかどうかを判定する(拡張子やMIMEタイプなど)"""
        raise NotImplementedError()

    @abstractmethod
    def extract_content(self, path: Path) -> ExtractedContent:
        """ファイルからテキストを抽出する"""
        raise NotImplementedError()


# -----------------------------
# 共通ヘルパ
# -----------------------------
def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _replacement_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count(REPLACEMENT_CHAR) / max(len(text), 1)

def _c1_control_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if 0x80 <= ord(ch) <= 0x9F)
    return n / max(len(text), 1)


def _latin1_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if 0xA0 <= ord(ch) <= 0xFF)
    return n / max(len(text), 1)


def _japanese_ratio(text: str) -> float:
    if not text:
        return 0.0

    def is_jp(ch: str) -> bool:
        o = ord(ch)
        return (
            (0x3040 <= o <= 0x309F)  # Hiragana
            or (0x30A0 <= o <= 0x30FF)  # Katakana
            or (0x4E00 <= o <= 0x9FFF)  # CJK Unified Ideographs
            or (0xFF66 <= o <= 0xFF9D)  # Halfwidth Katakana
        )

    n = sum(1 for ch in text if is_jp(ch))
    return n / max(len(text), 1)


def _decode_bytes_with_fallback(
    data: bytes,
    encodings: List[str],
) -> Tuple[str, str, List[str]]:
    """
    Returns: (decoded_text, used_encoding, warnings)
    """
    warnings: List[str] = []
    for enc in encodings:
        try:
            return _normalize_newlines(data.decode(enc)), enc, warnings
        except UnicodeDecodeError:
            continue

    # last resort
    text = data.decode(encodings[0], errors="replace")
    warnings.append("decode_errors_replaced")
    return _normalize_newlines(text), encodings[0], warnings


# -----------------------------
# PDF Extractor (PyPDF2 -> PyMuPDF fallback)
# -----------------------------
class PDFContentExtractor(ContentExtractor):
    def __init__(
        self,
        min_text_len: int = 100,
        max_replacement_ratio: float = 0.01,  # 1%
        empty_page_ratio_threshold: float = 0.60,
        enable_pymupdf_fallback: bool = True,
    ) -> None:
        self.min_text_len = min_text_len
        self.max_replacement_ratio = max_replacement_ratio
        self.empty_page_ratio_threshold = empty_page_ratio_threshold
        self.enable_pymupdf_fallback = enable_pymupdf_fallback

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract_content(self, path: Path) -> ExtractedContent:
        # 1) PyPDF2
        pypdf2_content = self._extract_with_pypdf2(path)
        pypdf2_warnings = self._assess_pdf_quality(pypdf2_content)

        # フォールバック不可ならそのまま返す
        if not self.enable_pymupdf_fallback or fitz is None:
            md = pypdf2_content.metadata
            md["engine"] = "pypdf2"
            md["warnings"] = sorted(set(pypdf2_warnings + (["pymupdf_not_installed"] if (self.enable_pymupdf_fallback and fitz is None) else [])))
            return pypdf2_content

        # フォールバック判定
        if not self._should_fallback(pypdf2_warnings):
            md = pypdf2_content.metadata
            md["engine"] = "pypdf2"
            md["warnings"] = pypdf2_warnings
            return pypdf2_content

        # 2) PyMuPDF
        try:
            pymupdf_content = self._extract_with_pymupdf(path)
            pymupdf_warnings = self._assess_pdf_quality(pymupdf_content)
        except Exception as e:
            md = pypdf2_content.metadata
            md["engine"] = "pypdf2"
            md["warnings"] = sorted(set(pypdf2_warnings + ["pymupdf_extract_failed"]))
            md["fallback"] = {"from": "pypdf2", "to": "pypdf2", "error": str(e), "trigger_warnings": pypdf2_warnings}
            return pypdf2_content

        # 3) どちらを採用するか決める
        chosen, engine, warnings = self._choose_better_result(
            pypdf2_content, pypdf2_warnings,
            pymupdf_content, pymupdf_warnings,
        )
        chosen.metadata["engine"] = engine
        chosen.metadata["warnings"] = warnings
        chosen.metadata["fallback"] = {
            "from": "pypdf2",
            "to": engine,
            "trigger_warnings": pypdf2_warnings,
            "metrics": {
                "pypdf2": self._quality_metrics(pypdf2_content, pypdf2_warnings),
                "pymupdf": self._quality_metrics(pymupdf_content, pymupdf_warnings),
            },
        }
        return chosen

    def _extract_with_pypdf2(self, path: Path) -> ExtractedContent:
        with pywarnings.catch_warnings(record=True) as w:
            pywarnings.simplefilter("always", PdfReadWarning)

            reader = PdfReader(str(path))
            meta = getattr(reader, "metadata", None)

            pages_text: List[str] = []
            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                pages_text.append(_normalize_newlines(text))

            full_text = "\n".join(pages_text).strip()
            num_pages = len(reader.pages)

        pdfread_warning_msgs = [str(x.message) for x in w]

        def safe_get(attr_name: str) -> str:
            if meta is None:
                return ""
            return getattr(meta, attr_name, "") or ""

        return ExtractedContent(
            full_text=full_text,
            pages=pages_text,
            num_pages=num_pages,
            metadata={
                "type": "pdf",
                "pdf_meta": {
                    "author": safe_get("author"),
                    "creator": safe_get("creator"),
                    "producer": safe_get("producer"),
                    "subject": safe_get("subject"),
                    "title": safe_get("title"),
                },
                "pypdf2_pdfread_warnings": pdfread_warning_msgs,
            },
        )

    def _extract_with_pymupdf(self, path: Path) -> ExtractedContent:
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is not installed.")

        doc = fitz.open(str(path))
        pages_text: List[str] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages_text.append(_normalize_newlines(text))

        full_text = "\n".join(pages_text).strip()
        return ExtractedContent(
            full_text=full_text,
            pages=pages_text,
            num_pages=doc.page_count,
            metadata={
                "type": "pdf",
            },
        )

    def _assess_pdf_quality(self, content: ExtractedContent) -> List[str]:
        warnings: List[str] = []

        text = content.full_text or ""

        # 基本ヒューリスティック
        if len(text) < self.min_text_len:
            warnings.append("low_text_volume")

        if _replacement_ratio(text) > self.max_replacement_ratio:
            warnings.append("replacement_characters_many")
        
        if len(text.strip()) == 0:
            warnings.append("no_text_extracted")

        pages = content.pages or []
        if pages:
            empty_pages = sum(1 for p in pages if len((p or "").strip()) == 0)
            empty_ratio = empty_pages / max(len(pages), 1)
            if empty_ratio >= self.empty_page_ratio_threshold:
                warnings.append("image_pdf_suspected")

        # mojibake（あなたの提示例に対応）
        c1 = _c1_control_ratio(text)
        l1 = _latin1_ratio(text)
        jp = _japanese_ratio(text)

        # 推奨判定：
        # - C1制御文字は強いシグナルなので単体でも疑う
        # - Latin-1は補助（日本語比率が低い時に強く疑う）
        if c1 > 0.003:
            warnings.append("mojibake_suspected")
        elif (l1 > 0.02) and (jp < 0.10):
            warnings.append("mojibake_suspected")

        msgs = content.metadata.get("pypdf2_pdfread_warnings") or []
        if any("Advanced encoding" in m and "not implemented" in m for m in msgs):
            warnings.append("pypdf2_advanced_encoding_unimplemented")
        # warningsの重複を抑える（監査ログを綺麗に）
        return sorted(set(warnings))


    def _should_fallback(self, warnings: List[str]) -> bool:
        critical = {"low_text_volume", "replacement_characters_many", "image_pdf_suspected", "mojibake_suspected", "pypdf2_advanced_encoding_unimplemented",}
        return any(w in critical for w in warnings)

    def _choose_better_result(
        self,
        p_content: ExtractedContent,
        p_warn: List[str],
        m_content: ExtractedContent,
        m_warn: List[str],
    ) -> Tuple[ExtractedContent, str, List[str]]:
        # 1) pypdf2固有の致命的警告があるなら pymupdf 優先（長さより強い）
        if "pypdf2_advanced_encoding_unimplemented" in p_warn:
            return m_content, "pymupdf", m_warn

        # 2) mojibake の片側優位も長さより優先（中身が壊れている可能性が高い）
        if ("mojibake_suspected" in p_warn) and ("mojibake_suspected" not in m_warn):
            return m_content, "pymupdf", m_warn
        if ("mojibake_suspected" in m_warn) and ("mojibake_suspected" not in p_warn):
            return p_content, "pypdf2", p_warn

        # 3) テキスト量(大差がある場合)
        p_len = len(p_content.full_text or "")
        m_len = len(m_content.full_text or "")
        if m_len > p_len * 1.10:
            return m_content, "pymupdf", m_warn
        if p_len > m_len * 1.10:
            return p_content, "pypdf2", p_warn

        # 4) 置換文字比率
        p_rr = _replacement_ratio(p_content.full_text or "")
        m_rr = _replacement_ratio(m_content.full_text or "")
        if m_rr < p_rr:
            return m_content, "pymupdf", m_warn
        if p_rr < m_rr:
            return p_content, "pypdf2", p_warn
        
        # 5) warnings 少ない方
        if len(m_warn) < len(p_warn):
            return m_content, "pymupdf", m_warn

        return p_content, "pypdf2", p_warn
    
    def _quality_metrics(self, content: ExtractedContent, warns: list[str]) -> dict[str, Any]:
        text = content.full_text or ""
        pages = content.pages or []
        empty_ratio = None
        if pages:
            empty_pages = sum(1 for p in pages if len((p or "").strip()) == 0)
            empty_ratio = empty_pages / max(len(pages), 1)

        return {
            "len": len(text),
            "replacement_ratio": _replacement_ratio(text),
            "c1_ratio": _c1_control_ratio(text),
            "latin1_ratio": _latin1_ratio(text),
            "jp_ratio": _japanese_ratio(text),
            "empty_page_ratio": empty_ratio,
            "warnings": warns,
        }


# -----------------------------
# Text / Markdown Extractor
# -----------------------------
class TextContentExtractor(ContentExtractor):
    def __init__(
        self,
        encodings: Optional[List[str]] = None,
        max_replacement_ratio: float = 0.01,
    ) -> None:
        self.encodings = encodings or ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp"]
        self.max_replacement_ratio = max_replacement_ratio

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".txt", ".md", ".markdown"}

    def extract_content(self, path: Path) -> ExtractedContent:
        data = path.read_bytes()
        text, used_enc, warnings = _decode_bytes_with_fallback(data, self.encodings)

        if _replacement_ratio(text) > self.max_replacement_ratio:
            warnings.append("replacement_characters_many")

        warnings = sorted(set(warnings))
        # engineは "text" で統一（MDもここに含める）
        return ExtractedContent(
            full_text=text.strip(),
            pages=None,
            num_pages=None,
            metadata={
                "type": "text",
                "engine": "text",
                "encoding": used_enc,
                "warnings": warnings,
            },
        )


# -----------------------------
# CSV Extractor（表構造を壊さないために key=value 正規化まで実施）
# -----------------------------
class CSVContentExtractor(ContentExtractor):
    def __init__(
        self,
        encodings: Optional[List[str]] = None,
        rows_per_chunk_hint: int = 20,
    ) -> None:
        self.encodings = encodings or ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp", "iso2022_jp"]
        self.rows_per_chunk_hint = rows_per_chunk_hint

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def extract_content(self, path: Path) -> ExtractedContent:
        data = path.read_bytes()
        text, used_enc, warnings = _decode_bytes_with_fallback(data, self.encodings)

        delimiter = ","
        try:
            sample = text[:4096]
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            warnings.append("csv_dialect_sniff_failed")

        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        rows = list(reader)
        warnings = sorted(set(warnings))
        if not rows:
            warnings.append("csv_empty")
            return ExtractedContent(
                full_text="",
                metadata={
                    "type": "csv",
                    "engine": "csv",
                    "encoding": used_enc,
                    "delimiter": delimiter,
                    "warnings": warnings,
                    "csv_header": [],
                    "rows_per_chunk_hint": self.rows_per_chunk_hint,
                },
            )

        header = [h.strip() for h in rows[0]]
        body = rows[1:]

        normalized_lines: List[str] = []
        for row in body:
            if len(row) != len(header):
                warnings.append("csv_inconsistent_columns")
            row2 = (row + [""] * len(header))[: len(header)]
            kv = [f"{k}={v}".strip() for k, v in zip(header, row2)]
            normalized_lines.append(" / ".join(kv))

        warnings = sorted(set(warnings))
        return ExtractedContent(
            full_text="\n".join(normalized_lines).strip(),
            pages=None,
            num_pages=None,
            metadata={
                "type": "csv",
                "engine": "csv",
                "encoding": used_enc,
                "delimiter": delimiter,
                "warnings": warnings,
                "csv_header": header,
                "rows_per_chunk_hint": self.rows_per_chunk_hint,
            },
        )



# -----------------------------
# Factory（既存構造に馴染む形）
# -----------------------------
DEFAULT_EXTRACTORS: List[ContentExtractor] = [
    PDFContentExtractor(),
    CSVContentExtractor(),
    TextContentExtractor(),
]


def get_extractor(path: Path, extractors: Optional[List[ContentExtractor]] = None) -> ContentExtractor:
    extractors = extractors or DEFAULT_EXTRACTORS
    for ex in extractors:
        if ex.can_handle(path):
            return ex
    raise ValueError(f"Unsupported file type: {path.suffix.lower()}")
