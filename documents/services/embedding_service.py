from langchain_openai import OpenAIEmbeddings
from django.conf import settings
class EmbeddingService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            )

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        複数テキスト(チャンク)用。(ドキュメント登録時に使います。)
        戻り値：「ベクトル、ベクトル、…」のlist[list[float]]
        """
        return self.embeddings.embed_documents(chunks)
    
    def embed_text(self, text: str) -> list[float]:
        """
        単一テキストをベクトル化します。
        ユーザの質問用に使用
        戻り値:１本のベクトル
        """
        return self.embeddings.embed_query(text)