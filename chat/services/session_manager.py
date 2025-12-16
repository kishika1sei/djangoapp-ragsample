from django.db import transaction
from django.utils import timezone

from chat.models import  ChatSession


class ChatSessionService:
    @staticmethod
    def get_or_create_session(request):
        """
        まだチャットセッションがない場合は新規作成
        すでにセッションIDをsession / cookie / URL paramで持っているならそれを使う
        """
        session_id = request.session.get("chat_session_id")

        if session_id:
            qs = ChatSession.objects.filter(id=session_id, ended_at__isnull=True)

            # ログイン中であれば自分のセッションのみ
            if request.user.is_authenticated:
                qs = qs.filter(user=request.user)
            else:
                qs = qs.filter(user__isnull=True)

            session = qs.first()
            if session:
                return session
        
        session = ChatSession.objects.create(
            user=request.user if request.user.is_authenticated else None,
        )
        request.session["chat_session_id"] = session.id
        return session
    
    @staticmethod
    def reset_session(request):
        """
        現在のセッションを「終了扱い」にして、次回アクセス時に新規セッションが作られる状態に戻す。
        """
        session_id = request.session.get("chat_session_id")
        if not session_id:
            return
        
        qs = ChatSession.objects.filter(id=session_id, ended_at__isnull=True)

        # ログイン中は自分のセッションのみ
        if request.user.is_authenticated:
            qs = qs.filter(user=request.user)
        else:
            qs = qs.filter(user__isnull=True)

        session = qs.first()
        if not session:
            request.session.pop("chat_session_id", None)
            return
        with transaction.atomic():
            session.ended_at = timezone.now()
            session.save(update_fields=["ended_at"])

        # 次回は新規セッションを作らせる
        request.session.pop("chat_session_id", None)