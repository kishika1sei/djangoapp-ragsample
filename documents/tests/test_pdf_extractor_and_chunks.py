# python manage.py test documents.tests.test_pdf_extractor_and_chunks

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
# TextSplitter の設定（本番と同じもの想定→変わるかも）
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "、", " ", ""],
    chunk_size=300,
    chunk_overlap=80,
)


# ① プロジェクトルート（scripts の 1つ上）を sys.path に追加
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

# ② これで documents モジュールが import できるようになる
from documents.services.content_extractor import PDFContentExtractor


def main():
    # ③ テスト用の PDF ファイルを指定
    #   例: プロジェクト直下に sample.pdf を置く想定
    pdf_path = BASE_DIR / "sample.pdf"

    extractor = PDFContentExtractor()

    assert extractor.can_handle(pdf_path), "can_handle が False になっている"

    content = extractor.extract_content(pdf_path)

    print("=== full_text (先頭500文字) ===")
    print(content.full_text[:500])
    print()
    print("num_pages:", content.num_pages)
    print("metadata:", content.metadata)
    chunks = text_splitter.split_text(content.full_text)
    print()
    print(f"=== チャンク数: {len(chunks)} ===")

    # 先頭数チャンクだけ確認
    for i, chunk in enumerate(chunks[:5]):
        print("-" * 40)
        print(f"[chunk {i}] length={len(chunk)}")
        print(chunk)

    # 長さチェック
    too_long = [len(c) for c in chunks if len(c) > 300]
    print()
    print(f"max chunk length: {max(len(c) for c in chunks)}")
    print(f"over 300 chars chunks: {len(too_long)}")


if __name__ == "__main__":
    main()
