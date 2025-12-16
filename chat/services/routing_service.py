from __future__ import annotations

from typing import Iterable, Optional

from openai import OpenAI
from pydantic import ValidationError

from chat.schemas.routing import RoutingResult

import logging
logger = logging.getLogger(__name__)

class RoutingService:
    """
    - 1回のLLM呼び出しで、業務判定 + 部門判定 + clarification判定を返す
    - Structured Outputs で「JSONスキーマ準拠」を強制
    - その上で、Pydanticで整合性チェック
    - 失敗時は安全側（業務扱い + clarification）へ倒す
    """

    ROUTING_MODEL = "gpt-4.1-nano"

    def __init__(self, model: str | None = None, client: OpenAI | None = None):
        self.model = model or self.ROUTING_MODEL
        self.client = client or OpenAI()

    def route(
        self,
        user_text: str,
        *,
        department_codes: Iterable[str],
        session_context: Optional[str] = None,
    ) -> RoutingResult:
        dept_codes = sorted(set([c.strip() for c in department_codes if c and c.strip()]))

        # ここで「DBにある部門コード」だけを候補としてモデルに提示する
        # （部門追加にも追従できる）
        dept_hint = ", ".join(dept_codes) if dept_codes else "(none)"

        instructions = (
            "あなたは社内問い合わせ回答アシスタントのルーティング担当です。\n"
            "次のJSONスキーマに厳密に従って出力してください。\n"
            "判定の方針:\n"
            "- 業務かどうか曖昧なら is_business は true 寄りにする\n"
            "- ただし曖昧で誤回答リスクが高い場合は needs_clarification=true にし、clarifying_question を1つだけ作る\n"
            "- primary_department は必ず部門コードで返す（不明なら unknown）\n"
            "- secondary_departments は最大2つ程度まで（不要なら空配列）\n"
            "\n"
            f"利用可能な部門コード一覧: {dept_hint}\n"
        )

        # session_context はあれば「補助情報」として入れる（ないなら省略）
        user_payload = (
            f"ユーザの質問:\n{user_text}\n\n"
            + (f"直近文脈(要約):\n{session_context}\n" if session_context else "")
        )

        try:
            # Structured Outputs: Pydanticモデルを text_format に渡して parse する
            resp = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_payload},
                ],
                text_format=RoutingResult,
                temperature=0,  # 分類なので低温度
            )

            result: RoutingResult = resp.output_parsed

            # 追加の“業務ロジック側”検証（DBにない部門コードが来た場合など）
            return self._post_validate(result, dept_codes)

        except ValidationError as e:
            # 失敗時は安全側に倒す（業務扱い + clarification）
            # 想定失敗例:Pydantic起因の失敗(True→yesが入ってる)
            logger.warning("パース/バリデーション経路が失敗しました。", exc_info=e)
            return RoutingResult(
                is_business=True,
                business_confidence=0.0,
                primary_department="unknown",
                department_confidence=0.0,
                secondary_departments=[],
                needs_clarification=True,
                clarifying_question="どの手続き・制度・トピックに関する問い合わせか、具体名を1つ教えてください。",
            )
        except Exception as e:
            # LLMのAPI呼び出し失敗など
            logger.exception("APIコールが失敗しました。", exc_info=e)
            return RoutingResult(
                is_business=True,
                business_confidence=0.0,
                primary_department="unknown",
                department_confidence=0.0,
                secondary_departments=[],
                needs_clarification=True,
                clarifying_question="通信/内部エラーが発生しました。もう一度お試しください。",
            )


    def _post_validate(self, result: RoutingResult, dept_codes: list[str]) -> RoutingResult:
        # unknown は許容
        if result.primary_department != "unknown" and dept_codes and result.primary_department not in dept_codes:
            # DBにないコード → ルーティング結果としては使えないので clarification へ
            result.needs_clarification = True
            result.clarifying_question = (
                "どの部門の内容に近いですか？次から選んでください: "
                + ", ".join(dept_codes)
            )
            result.primary_department = "unknown"
            result.department_confidence = 0.0
            result.secondary_departments = []

        # secondary も同様にフィルタ
        if dept_codes:
            result.secondary_departments = [
                d for d in result.secondary_departments
                if d in dept_codes and d != result.primary_department
            ][:2]

        return result
