from langchain_openai import OpenAIEmbeddings
from django.conf import settings
class EmbeddingService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            )

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(chunks)