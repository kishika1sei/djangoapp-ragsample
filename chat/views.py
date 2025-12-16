from django.shortcuts import render,redirect
from django.views.decorators.http import require_POST
from .models import ChatMessage
from .services import rag_service
from .services.session_manager import ChatSessionService
RECENT_MESSAGE_LIMIT = 30

import logging
logger = logging.getLogger(__name__)

def index(request):
    # 1.セッションを特定(なければ作る)
    session = ChatSessionService.get_or_create_session(request)

    if request.method == "POST":
        # 1.チャット内容を受け取る
        user_text = request.POST.get("message", "").strip()
        # TODO: 会話履歴表示用にタイトル自動生成するならこの辺に追加予定

        if not user_text:
            return redirect("chat:index")
        
        # 2.ユーザメッセージを保存
        user_msg = ChatMessage.objects.create(
            session=session,
            role=ChatMessage.Role.USER,
            content=user_text,
        )
        # 3.結果をRAGServiceに「このセッションでチャットして」と依頼
            # 1.FAISSにクエリを渡して検索を依頼し、LLMも呼び出す
        try:
            answer, meta = rag_service.chat(session=session,user_message=user_text)
        except Exception:
            logger.exception("rag_service.chat failed")
            answer = "申し訳ありません。回答生成中にエラーが発生しました。もう一度お試しください。"
            meta = {}

        # routing_metaをユーザメッセージに保存
        routing = meta.get("routing")
        if routing is not None:
            user_msg.routing_meta = routing
            user_msg.save(update_fields=["routing_meta"])
        
        # 4.ユーザに回答を返す(画面表示)
        assistant_msg = ChatMessage.objects.create(
            session=session,
            role=ChatMessage.Role.ASSISTANT,
            content=answer,
        )

        # retrieval_metaを assistant側に保存
        retrieval = meta.get("retrieval")
        if retrieval is not None:
            assistant_msg.retrieval_meta = retrieval
            assistant_msg.save(update_fields=["retrieval_meta"])

        # POST-redirect-GET パターンで再読み込み時の二重送信を防ぐ
        return redirect("chat:index")
    
    qs = (
        ChatMessage.objects
        .filter(session=session)
        .order_by("-created_at")[:RECENT_MESSAGE_LIMIT]
    )
    messages = list(qs)[::-1] #古い→新しい順に並び替え

    context = {
        "messages":messages,
        "answer_department": getattr(session, "answer_department", None),
        # TODO: meta情報も必要に応じて追加予定
    }
    return render(request, 'chat/index.html', context)

@require_POST
def reset_view(request):
    ChatSessionService.reset_session(request)
    return redirect("chat:index")

