from django.shortcuts import render,redirect
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from .models import ChatMessage
from .services import rag_service
from .services.session_manager import ChatSessionService
RECENT_MESSAGE_LIMIT = 30
from accounts.models import Department

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
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JsonResponse({"error": "empty_message"}, status=400)
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
        finally:
            update_session_answer_department_from_meta(session, meta)
            session.refresh_from_db(fields=["answer_department"])

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
        
        # citationsをassistant側に保存
        assistant_msg.citations = meta.get("citations", []) or []

        # まとめて保存
        assistant_msg.save(update_fields=["retrieval_meta", "citations"])

        # AJAX（fetch）ならJSONを返す（ページ遷移しない）
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({
                "assistant": answer,
                "meta": meta, # meta肥大化するなら将来はused_document_idsを返すだけ等調整する
                "answer_department": display_answer_department(session),  # 日本語名
                # 必要ならコードも返せる
                # "answer_department_code": session.answer_department.code if session.answer_department else None,
            })
        # POST-redirect-GET パターンで再読み込み時の二重送信を防ぐ
        return redirect("chat:index")
    
    qs = (
        ChatMessage.objects
        .filter(session=session)
        .order_by("-created_at")[:RECENT_MESSAGE_LIMIT]
    )
    messages = list(qs)[::-1] #古い→新しい順に並び替え
    
    latest_assistant = (
        ChatMessage.objects
        .filter(session=session, role=ChatMessage.Role.ASSISTANT)
        .order_by("-created_at")
        .first()
    )
    initial_citations = (latest_assistant.citations if latest_assistant else []) or []

    context = {
        "messages":messages,
        "answer_department": display_answer_department(session),
        # TODO: meta情報も必要に応じて追加予定
        "initial_citations": initial_citations,
    }
    return render(request, 'chat/index.html', context)

@require_POST
def reset_view(request):
    ChatSessionService.reset_session(request)
    return redirect("chat:index")


# --- ヘルパー関数群 ---
def extract_department_code_from_meta(meta) -> str | None:
    routing = (meta or {}).get("routing")
    if not isinstance(routing, dict):
        return None
    code = (routing.get("primary_department") or "").strip()
    if not code or code == "unknown":
        return None
    return code

def update_session_answer_department_from_meta(session, meta) -> None:
    """
    metaから部門コードを抽出し、解決できた場合のみ session.answer_department を更新する。
    metaに部門が無い / unknown / DBに存在しない場合は更新しない（既存値維持）。
    """
    dept_code = extract_department_code_from_meta(meta)
    if not dept_code:
        return

    dept = Department.objects.filter(code=dept_code).first()
    if not dept:
        return

    # 既に同じなら無駄なUPDATEを避ける
    if getattr(session, "answer_department_id", None) != dept.id:
        session.answer_department = dept
        session.save(update_fields=["answer_department"])

# 日本語名表示の補助関数
def display_answer_department(session) -> str | None:
    dept = getattr(session, "answer_department", None)
    return dept.name if dept else None