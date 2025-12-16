from django.conf import settings
from documents.services.embedding_service import EmbeddingService
from documents.search_backends.faiss_backend import FaissSearchBackend
from chat.services.rag_chat import RAGChatService
from chat.services.llm_client import OpenAILlmClient

embedding_service = EmbeddingService()
search_backend = FaissSearchBackend(
    index_path=settings.FAISS_INDEX_PATH,
    embedding_service=embedding_service,
)
llm_client = OpenAILlmClient(api_key=settings.OPENAI_API_KEY)

rag_service = RAGChatService(
    search_backend=search_backend,
    embedding_service=embedding_service,
    llm_client=llm_client,
)