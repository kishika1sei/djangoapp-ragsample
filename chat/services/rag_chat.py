from chat.models import ChatMessage,ChatSession
from chat.services.routing_service import RoutingService
from accounts.models import Department


class RAGChatService:
    def __init__(self,search_backend,embedding_service,llm_client, router: RoutingService | None = None):
        self.search_backend = search_backend
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.router = router or RoutingService(model="gpt-4.1-nano")
        
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
        SNIP = 200 # １メッセージ当たりの上限
        session_context = ""
        for m in reversed(history_messages):
            line = f"{m.role}: {m.content[:SNIP]}\n"
            if len(session_context) + len(line) > MAX_CHARS:
                break
            session_context = line + session_context
        # 0-1.分類器に業務判定と部門判定を委託する
        route = self.router.route(
            user_text=user_message,
            department_codes=dept_codes,
            # 直近の会話を挿入してルーティングの精度を上げる
            session_context=session_context or None,
        )

        # ルーティング結果をmetaに載せる
        route_meta = route.model_dump() if hasattr(route, "model_dump") else dict(route)

        # 0-2. 業務外なら、RAG処理に進まず返す
        if not route.is_business:
            return "本件は社内業務に関する問い合わせではない可能性が高いです。業務に関する内容であれば目的や対象手続きを具体的に教えてください。",{
                "routing": route_meta,
                "reason": "not_business",
            }

        # 1. ユーザのクエリをベクトル化する(embeddingservice)
        query_embedding = self.embedding_service.embed_text(user_message)
        
        # 2. FAISSにクエリを投げて似ているチャンクをtop_k件頂戴と聞く(search_backend)
        search_results, retrieval_meta = self._search_with_fallback(
            query_embedding=query_embedding,
            route=route,
            top_k= 5,
        )

        top_score = retrieval_meta.get("top_score")
        hit_count = retrieval_meta.get("hit_count", 0)
        
        threshold = retrieval_meta.get("score_threshold", 0.55) # 検索の閾値

        retrieval_meta.setdefault("score_threshold", threshold)
        retrieval_meta.setdefault("top_score", top_score)

        search_weak = (top_score is None or hit_count == 0 or top_score < threshold) # 検索弱いの定義
        # デバッグ用(TODO:後で消す)
        print('#検索結果ここから')
        print(search_results)
        print('#検索結果ここまで')

        # 検索結果が弱いならclarificationを返す
        if search_weak:
            return (
            "関連資料を特定できませんでした。対象の制度・手続き名（または担当部署の心当たり）を教えてください。",
                {
                    "routing" : route_meta,
                    "retrieval": retrieval_meta,
                    "reason": "search_weak",
                },
            )
            
        
        # 3. チャンク内容をもとにコンテキストを組み立てる
        context_texts = []
        used_documents = set()

        for result in search_results:
            chunk = result.chunk
            context_texts.append(chunk.content)
            if chunk.document:
                used_documents.add(chunk.document)
        
        context_block = "\n\n".join(context_texts)
    
        # 4. システムプロンプトを第一候補の部門から作成する
        system_prompt =self._select_system_prompt(route.primary_department)

        # 5. 最終的にLLMに渡すプロンプトを構築する
        prompt = self._build_prompt(
            system_prompt=system_prompt,
            history=history_messages,
            context=context_block,
            user_message=user_message,
        )

        # 6. LLMを読んで、回答を生成
        answer_text = self.llm_client.complete(prompt)

        # 7. meta 情報を (どのドキュメントを使ったか等) を組み立てて返す
        meta = {
            "routing": route_meta,
            "retrieval":retrieval_meta,
            "used_document_ids": [doc.id for doc in used_documents],
            "num_context_chunks": len(search_results),
            "citations": self._build_citations(search_results),
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

    
    def _search_with_fallback(self, *, query_embedding, route, top_k: int):
        # 閾値は仮置き。分布を見て後で調整する。
        SCORE_THRESHOLD = 0.55

        scopes = []
        if route.primary_department and route.primary_department != "unknown":
            scopes.append(route.primary_department)
        for d in route.secondary_departments:
            if d and d != "unknown" and d not in scopes:
                scopes.append(d)

        # 1) primary → secondary
        for scope in scopes:
            results = self.search_backend.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters={"department_code": scope},
            )
            top_score = results[0].score if results else None

            if top_score is not None and top_score >= SCORE_THRESHOLD:
                return results, {
                    "engine": "vector",
                    "scope_used": scope,
                    "fallback_triggered": False,
                    "top_score": float(top_score),
                    "hit_count": len(results),
                    "k": top_k,
                    "score_threshold": SCORE_THRESHOLD,
                }

        # 2) 全社フォールバック
        results = self.search_backend.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=None,
        )
        top_score = results[0].score if results else None

        return results, {
            "engine": "vector",
            "scope_used": "company",
            "fallback_triggered": True,
            "top_score": float(top_score) if top_score is not None else None,
            "hit_count": len(results),
            "k": top_k,
            "score_threshold": SCORE_THRESHOLD,
        }

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

