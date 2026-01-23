from django.conf import settings
from documents.services.embedding_service import EmbeddingService
from documents.search_backends.faiss_backend import FaissSearchBackend
from documents.search_backends.lexical_retriever import LexicalRetriever, LexicalConfig
from documents.search_backends.hybrid_retriever import HybridRetriever, RRFConfig
from chat.services.rag_chat import RAGChatService
from chat.services.llm_client import OpenAILlmClient
from chat.services.routing_service import RoutingService

# --- 共通コンポーネント ---
embedding_service = EmbeddingService()

# --- Vector backend（既存） ---
search_backend = FaissSearchBackend(
    index_path=settings.FAISS_INDEX_PATH,
    embedding_service=embedding_service,
)

# --- Lexical（pg_trgm） ---
lexical = LexicalRetriever(
    config=LexicalConfig(
        similarity_threshold=0.05,  # 最初は緩め（弾きすぎ防止）
        use_threshold=True,
    )
)

# --- Hybrid（RRF融合） ---
hybrid_retriever = HybridRetriever(
    vector_backend=search_backend,
    lexical=lexical,
    config=RRFConfig(
        k=60,
        vector_fetch_mult=8,
        lexical_fetch_mult=8,
    ),
)

# --- LLM ---
llm_client = OpenAILlmClient(api_key=settings.OPENAI_API_KEY)

# --- Routing service ---
router = RoutingService(model="gpt-4.1-nano")

# --- RAG service（hybrid固定） ---
rag_service = RAGChatService(
    search_backend=search_backend,
    embedding_service=embedding_service,
    llm_client=llm_client,
    router=router,
    retriever=hybrid_retriever, 
)