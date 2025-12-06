from pathlib import Path
from django.core.files.storage import default_storage
from documents.models import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents.services.content_extractor import PDFContentExtractor
from documents.services.embedding_service import EmbeddingService
from documents.models import Chunk

# TextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "、", " ", ""],
    chunk_size = 300,
    chunk_overlap = 80,
)
# Embeddingサービスの初期化
embedding_service = EmbeddingService()

class DocumentIngestionService:
    @classmethod
    def ingest_document(cls, document:Document):
        """一つのドキュメントに対して、抽出→チャンク分割→Embedding→チャンク保存を行う"""

        # 1. PDFからテキストを抽出
        pdf_fs_path_str = default_storage.path(document.file_path)
        pdf_path = Path(pdf_fs_path_str)
        extractor = PDFContentExtractor()
        content = extractor.extract_content(pdf_path)

        # 2. ページごとに抽出したテキストをチャンク分割
        page_chunk_pairs: list[tuple[int,str]] = []
        for page_idx,page_text in enumerate(content.pages,start=1):
            page_chunks = text_splitter.split_text(page_text)
            for chunk_text in page_chunks:
                page_chunk_pairs.append((page_idx,chunk_text))
                
        # 3. 埋め込み用にテキストだけのリストを作る
        chunk_texts = [text for (_,text) in page_chunk_pairs]

        # 4. チャンクごとに埋め込みを一括生成
        vectors = embedding_service.embed_chunks(chunk_texts)

        # 5. チャンクモデルのレコードをまとめて作成
        chunk_objs: list[Chunk] = []
        for idx, ((page_idx, chunk_text), vec) in enumerate(zip(page_chunk_pairs, vectors)):
            chunk_objs.append(
                Chunk(
                    document=document,
                    chunk_index=idx,
                    page=page_idx,
                    content=chunk_text,
                    embedding=vec,
                )
            )

        Chunk.objects.bulk_create(chunk_objs)

        # 作成したチャンク数を返す
        return len(chunk_objs)