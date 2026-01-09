
import re

from chat.models import ChatMessage,ChatSession
from chat.services.routing_service import RoutingService
from accounts.models import Department

from documents.search_backends.hybrid_retriever import HybridRetriever

class RAGChatService:
    def __init__(self,search_backend,embedding_service,llm_client, router: RoutingService | None = None, retriever: HybridRetriever | None = None,):
        self.search_backend = search_backend
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.router = router or RoutingService(model="gpt-4.1-nano")
        # 本番想定：ハイブリッド固定で実装を前提のため、必須化
        if retriever is None:
            raise ValueError("HybridRetriever is required (hybrid mode).")
        self.retriever = retriever

    _AFFIRMATIONS = {"はい", "うん", "そう", "そうです", "ok", "okay", "了解", "りょうかい", "OK"}
    _NEGATIONS = {"いいえ", "ちがう", "違う", "違います", "no", "ノー", "違います。"}

    def _normalize_short(self, text: str) -> str:
        return (text or "").strip().lower().replace("。", "").replace("！", "").replace("?", "").replace("？", "")

    def _is_affirmation(self, text: str) -> bool:
        t = self._normalize_short(text)
        return t in {s.lower() for s in self._AFFIRMATIONS}

    def _is_negation(self, text: str) -> bool:
        t = self._normalize_short(text)
        return t in {s.lower() for s in self._NEGATIONS}

    def _is_too_short_to_standalone(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return self._is_affirmation(t) or self._is_negation(t)

    def _last_assistant_text(self, history_messages: list[ChatMessage], current_user_message: str) -> str | None:
        """
        views.py が user_msg を保存してから rag_service.chat を呼ぶため、
        history_messages に「今回のuser_message」が含まれる可能性がある。
        なので current_user_message と同一の USER は除外しつつ、直近の ASSISTANT を拾う。
        """
        cur = (current_user_message or "").strip()
        for m in reversed(history_messages):
            role = getattr(m, "role", None)
            txt = (getattr(m, "content", "") or "").strip()
            if not txt:
                continue
            if role == ChatMessage.Role.USER and txt == cur:
                continue
            if role == ChatMessage.Role.ASSISTANT:
                return txt
        return None

    def _looks_like_question(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        # UI上は「ですか」で終わることも多いので ? だけに依存しない
        return ("？" in t) or ("?" in t) or t.endswith("ですか") or ("教えてください" in t) or ("知りたいですか" in t)

    def _needs_date_slot(self, text: str) -> bool:
        """
        「いつまで？」「具体的な日付？」など、日付が必要なパターンを検知
        """
        t = (text or "")
        keywords = ["いつまで", "いつ", "日付", "期限", "締切", "締め切り", "何日", "何時まで", "まで"]
        return any(k in t for k in keywords)

    def _build_short_reply_clarification(self, *, last_assistant_text: str, dept_code: str) -> str:
        """
        短文返答が来たときの、次に聞くべきことを固定化してループを止める。
        """
        # 直前が「期限/日付」系なら、計算に必要な“取得予定日”を聞く（汎用で強い）
        if self._needs_date_slot(last_assistant_text):
            return (
                "了解しました。締切日を具体的に出すには「取得予定日（休暇を取る日）」が必要です。\n"
                "取得予定日を教えてください（例：2026-01-15）。そこから「2営業日前 17:00」の期限を計算します。"
            )

        # それ以外は部門メニューに落とす（Hard clarification を選択肢化）
        return self._menu_clarification(dept_code)
    # --------------------------
    # 追い質問用のクエリ拡張
    # --------------------------
    _FOLLOWUP_REGEXES = [
        r"^(それ|これ|その|あれ)$",
        r"^(詳細|規程|規則|ルール|手続き|申請|条件|要件|期間|締め日|期限|様式|フォーム|場所|どこ|どれ)$",
        r".*(について|に関して)(教えて|知りたい).*",
        r".*(規程|規則|ルール|手続き|申請)(について|に関して).*(教えて|知りたい).*",
    ]
    def _is_followup(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if len(t) <= 12:
            return any(re.match(p, t) for p in self._FOLLOWUP_REGEXES)
        return any(re.match(p, t) for p in self._FOLLOWUP_REGEXES)

    def _pick_last_topic_hint(self, history_messages: list[ChatMessage], current_user_message: str) -> str | None:
        """
        直近のユーザ発話から「話題のヒント」になりそうな文を拾う。
        """
        cur = (current_user_message or "").strip()

        # 古い→新しい順のhistoryを前提に、後ろから辿る（新しい順）
        for m in reversed(history_messages):
            if getattr(m, "role", None) != ChatMessage.Role.USER:
                continue
            txt = (getattr(m, "content", "") or "").strip()
            if not txt:
                continue
            # 同文（今回の入力が履歴に混ざる可能性）を除外
            if txt == cur:
                continue
            # “薄い追質問”はヒントにしない
            if self._is_followup(txt):
                continue
            return txt

        return None

    def _build_effective_query_text(self, user_message: str, history_messages: list[ChatMessage]) -> str:
        """
        検索用のクエリを組み立てる。
        フォローアップなら直近の具体トピックを付与して retrieval を安定させる。
        """
        user_message = (user_message or "").strip()
        if not self._is_followup(user_message):
            return user_message

        hint = self._pick_last_topic_hint(history_messages, current_user_message=user_message)
        if hint and hint != user_message:
            return f"{hint} {user_message}"

        return user_message
    
    # --------------------------
    # Hard clarification を選択肢化（ループ防止）
    # --------------------------
    def _is_generic_clarifying_question(self, q: str) -> bool:
        q = (q or "").strip()
        if not q:
            return True
        if "具体的に" in q and ("どのような" in q or "何を" in q):
            return True
        return False

    def _menu_clarification(self, dept_code: str) -> str:
        if dept_code == "hr":
            return (
                "人事のどのトピックについて知りたいですか？番号で教えてください。\n"
                "1) 年次有給休暇  2) 育児休業  3) 病気休暇  4) 残業/休日出勤  5) その他"
            )
        if dept_code == "finance":
            return (
                "経理のどのトピックについて知りたいですか？番号で教えてください。\n"
                "1) 経費精算  2) 交通費  3) 請求/支払  4) その他"
            )
        if dept_code == "legal":
            return (
                "法務のどのトピックについて知りたいですか？番号で教えてください。\n"
                "1) 契約書  2) 稟議/承認フロー  3) コンプライアンス  4) その他"
            )
        if dept_code == "it":
            return (
                "情シスのどのトピックについて知りたいですか？番号で教えてください。\n"
                "1) VPN  2) パスワードリセット  3) アカウント/端末  4) その他"
            )
        return "どの制度・手続きについて知りたいですか？制度名を1つ教えてください。"

    def _choose_hard_clarification_question(self, *, router_question: str, dept_code: str, default_q: str) -> str:
        if self._is_generic_clarifying_question(router_question):
            return self._menu_clarification(dept_code)
        return router_question or default_q
    

    # --------------------------
    # メイン処理
    # --------------------------
    def chat(self, session: ChatSession, user_message: str) -> tuple[str, dict]:
        """
        1回分のチャット処理を行う
        - 検索バックエンド(FAISSなど)から関連コンテキストを取得
        - LLMに投げて回答を生成
        - 回答本文とメタ情報(出典など)を返す
        """
        dept_codes = list(Department.objects.values_list("code", flat=True))

        # 過去メッセージを取得してプロンプトやルート判定に含める
        # 直近10往復分だけの履歴を使う
        HISTORY_LIMIT = 20
        history_qs = ChatMessage.objects.filter(
            session=session,
            role__in=[ChatMessage.Role.USER, ChatMessage.Role.ASSISTANT],
        ).order_by("-created_at")[:HISTORY_LIMIT] # 新しい順に取り出す
        history_messages = list(history_qs)[::-1] # LLMに渡すために古い順に戻す
        
        MAX_CHARS = 1000  # コンテキストが大きくならないように調整
        SNIP = 200 # 1メッセージ当たりの上限
        session_context = ""
        for m in reversed(history_messages):
            line = f"{m.role}: {m.content[:SNIP]}\n"
            if len(session_context) + len(line) > MAX_CHARS:
                break
            session_context = line + session_context
        # 分類器に業務判定と部門判定を委託する
        route = self.router.route(
            user_text=user_message,
            department_codes=dept_codes,
            # 直近の会話を挿入してルーティングの精度を上げる
            session_context=session_context or None,
        )

        # ルーティング結果をmetaに載せる
        route_meta = route.model_dump() if hasattr(route, "model_dump") else dict(route)

        # 業務外なら、RAG処理に進まず返す
        if not route.is_business:
            return "本件は社内業務に関する問い合わせではない可能性が高いです。業務に関する内容であれば目的や対象手続きを具体的に教えてください。",{
                "routing": route_meta,
                "reason": "not_business",
            }
        
        router_needs_clarification = bool(getattr(route, "needs_clarification", False))
        router_question = (getattr(route, "clarifying_question", "") or "").strip()
        if self._is_too_short_to_standalone(user_message):
            last_a = self._last_assistant_text(history_messages, current_user_message=user_message)
            if last_a and self._looks_like_question(last_a):
                q = self._build_short_reply_clarification(
                    last_assistant_text=last_a,
                    dept_code=route.primary_department,
                )
                return (
                    q,
                    {
                        "routing": route_meta,
                        "reason": "short_reply_needs_context",
                        "clarification": {
                            "mode": "hard",
                            "question": q,
                            "kind": "short_reply",
                        },
                    },
                )

        # 検索用クエリを文脈で拡張
        effective_query_text = self._build_effective_query_text(user_message, history_messages)    
    
        # ユーザのクエリをベクトル化する(embeddingservice)
        query_embedding = self.embedding_service.embed_text(effective_query_text)
        
        # Hybridにクエリとクエリ埋め込みを投げて似ているチャンクをtop_k件分頂戴と聞く
        search_results, retrieval_meta = self._search_with_fallback(
            query_text=effective_query_text,
            query_embedding=query_embedding,
            route=route,
            top_k= 5,
        )
        retrieval_meta["effective_query_text"] = effective_query_text
        # top_score = retrieval_meta.get("top_score")
        hit_count = int(retrieval_meta.get("hit_count", 0) or 0)
        # --- hybridはスコア閾値で弾かない---
        search_weak = (hit_count == 0)

        # # デバッグ用(TODO:後で消すかコメントアウト)
        # print('#検索結果ここから')
        # print(search_results)
        # print('#検索結果ここまで')

        # --- evidence strength（UXと安全性の折衷） ---
        vector_top = retrieval_meta.get("vector_top_score")
        try:
            vector_top_f = float(vector_top) if vector_top is not None else None
        except Exception:
            vector_top_f = None

        lexical_hit_count = int(retrieval_meta.get("lexical_hit_count", 0) or 0)

        used_documents = set()
        for r in (search_results or []):
            c = getattr(r, "chunk", None)
            d = getattr(c, "document", None) if c else None
            if d is not None:
                used_documents.add(d)
        num_unique_docs = len(used_documents)

        evidence_strong = (
            # キーワード検索でヒットがある
            (lexical_hit_count > 0)
            # ベクトル検索のトップスコアが十分高い
            or (vector_top_f is not None and vector_top_f >= 0.60)
            # 根拠が１ドキュメントに収束している
            or (num_unique_docs == 1)
        )

         # --- 検索弱い場合は hard clarification（router質問があれば優先） ---
        if search_weak:
            q = self._choose_hard_clarification_question(
                router_question=router_question,
                dept_code=route.primary_department,
                default_q="関連資料を特定できませんでした。対象の制度・手続き名（または担当部署の心当たり）を教えてください。",
            )
            return (
                q,
                {
                    "routing": route_meta,
                    "retrieval": retrieval_meta,
                    "reason": "search_weak",
                    "clarification": {
                        "mode": "hard",
                        "question": q,
                        "router_needs_clarification": router_needs_clarification,
                        "evidence_strong": evidence_strong,
                    },
                    "effective_query_text": effective_query_text,
                },
            )

        # --- routerがclarification要求 & 根拠も弱い → hard clarification ---
        # （“答えられるのに答えない”を避けるため、ここは evidence_strong を条件にしている）
        if router_needs_clarification and not evidence_strong:
            q = self._choose_hard_clarification_question(
                router_question=router_question,
                dept_code=route.primary_department,
                default_q="確認のため、対象の制度・手続き名（または前提条件）を1つだけ教えてください。",
            )
            return (
                q,
                {
                    "routing": route_meta,
                    "retrieval": retrieval_meta,
                    "reason": "router_clarification_and_weak_evidence",
                    "clarification": {
                        "mode": "hard",
                        "question": q,
                        "router_needs_clarification": True,
                        "evidence_strong": False,
                    },
                    "effective_query_text": effective_query_text,
                },
            )

            
        
        # チャンク内容をもとにコンテキストを組み立てる
        context_texts = []

        for result in search_results:
            chunk = result.chunk
            context_texts.append(chunk.content)
            if chunk.document:
                used_documents.add(chunk.document)
        
        context_block = "\n\n".join(context_texts)
    
        # システムプロンプトを第一候補の部門から作成する
        system_prompt =self._select_system_prompt(route.primary_department)

        # 最終的にLLMに渡すプロンプトを構築する
        prompt = self._build_prompt(
            system_prompt=system_prompt,
            history=history_messages,
            context=context_block,
            user_message=user_message,
        )

        # LMを呼んで、回答を生成
        answer_text = self.llm_client.complete(prompt)

        # meta 情報を (どのドキュメントを使ったか等) を組み立てて返す
        # --- routerがclarification要求 & 根拠が強い → soft clarification（末尾に1問だけ） ---
        clarification_meta = {"mode": "none"}
        if router_needs_clarification and router_question and not self._is_generic_clarifying_question(router_question):
            answer_text = f"{answer_text}\n\n（念のため確認）{router_question}"
            clarification_meta = {
                "mode": "soft",
                "question": router_question,
                "router_needs_clarification": True,
                "evidence_strong": True,
            }

        meta = {
            "routing": route_meta,
            "retrieval": retrieval_meta,
            "used_document_ids": [doc.id for doc in used_documents],
            "num_context_chunks": len(search_results),
            "citations": self._build_citations(search_results),
            "clarification": clarification_meta,
            "effective_query_text": effective_query_text,
        }

        return answer_text, meta
    
    def _build_prompt(self, system_prompt, history, context, user_message) -> str:
        """
        LLMに渡す入力文字列を組み立てるヘルパー
        最初はシンプルで。あとで ChatCompletion 形式に変えるなり拡張。
        """
        history_lines = []
        for msg in history:
            role = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        # 履歴の最後がUserでないなら今回の発話を追加する
        if not history or history[-1].role != "user" or history[-1].content != user_message:
            history_lines.append(f"User: {user_message}")

        history_block = "\n".join(history_lines)

        prompt = f"""[system]
        {system_prompt}

        [Conversation history]
        {history_block}

        [Retrieved context]
        {context}

        [Instruction]
        - 必ず「Question」に対しての回答をしてください。
        - 根拠は「Retrieved context」と「Conversation history」のみです。
        - 根拠が不足して断定できない場合は「手元の資料からは判断できません」と答えてください。
        - 推測で事実を作らないでください。

        [Question]
        {user_message}
        """
        return prompt

    def _search_with_fallback(self, *, query_text: str, query_embedding, route, top_k: int):
        scopes = []
        if route.primary_department and route.primary_department != "unknown":
            scopes.append(route.primary_department)
        for d in route.secondary_departments:
            if d and d != "unknown" and d not in scopes:
                scopes.append(d)

        for scope in scopes:
            results, meta = self.retriever.retrieve(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                filters={"department_code": scope},
            )
            if meta.get("hit_count", 0) > 0:
                meta.setdefault("scope_used", scope)
                meta.setdefault("fallback_triggered", False)
                meta.setdefault("k", top_k)
                return results, meta

        results, meta = self.retriever.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=None,
        )
        meta.setdefault("scope_used", "company")
        meta.setdefault("fallback_triggered", True)
        meta.setdefault("k", top_k)
        return results, meta

    def _select_system_prompt(self, dept_code: str) -> str:
        base = (
            "あなたは社内問合せ専用のアシスタントです。"
            "以下の社内資料（検索で取得したコンテキスト）を根拠に、日本語で簡潔かつ丁寧に回答してください。"
            "根拠が不足している場合は推測で断定せず、「手元の資料からは判断できません」と答えてください。"
        )

        roles = {
            "hr": "あなたは人事総務の担当者です。",
            "finance": "あなたは経理の担当者です。",
            "legal": "あなたは法務の担当者です。",
            "it": "あなたは情シスの担当者です。",
        }

        role = roles.get(dept_code, "あなたは総合窓口の担当者です。")
        return f"{base}\n{role}"
    
    # 引用構築
    def _build_citations(self, search_results) -> list[dict]:
        by_doc: dict[int, dict] = {}

        for r in (search_results or []):
            chunk = getattr(r, "chunk", None)
            if chunk is None or chunk.document_id is None:
                continue

            doc_id = int(chunk.document_id)
            doc = chunk.document

            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    "document_id": doc_id,
                    "title": (getattr(doc, "title", "") or f"Document#{doc_id}"),
                    "has_page": False,
                    "pages": set(),
                    "chunks": set(),
                }

            if chunk.page is not None:
                by_doc[doc_id]["has_page"] = True
                by_doc[doc_id]["pages"].add(int(chunk.page))
            else:
                if chunk.chunk_index is not None:
                    # ingestionは0-basedなので、表示は1-based推奨
                    by_doc[doc_id]["chunks"].add(int(chunk.chunk_index) + 1)

        citations: list[dict] = []
        for doc_id, acc in by_doc.items():
            if acc["has_page"] and acc["pages"]:
                locator = {"type": "page_set", "pages": sorted(acc["pages"])}
            else:
                locator = {"type": "chunk_set", "chunks": sorted(acc["chunks"])}

            citations.append({
                "document_id": acc["document_id"],
                "title": acc["title"],
                "locator": locator,
            })

        citations.sort(key=lambda x: (x.get("title") or "", x.get("document_id") or 0))
        return citations

