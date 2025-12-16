import os
from typing import Optional
from openai import OpenAI

_client_singleton: Optional[OpenAI] = None

# OpenAIとの接続処理を書く
def get_client() -> OpenAI:
    """
    OpenAIクライアントのシングルトンを返す。
    最初の１回だけインスタンスを作り、以降は再利用する。
    """
    global _client_singleton
    if _client_singleton is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が未設定です（.env を確認）")
        _client_singleton = OpenAI(api_key=api_key)  # envから拾う場合は OpenAI() でもOK
    return _client_singleton

class OpenAILlmClient():
    """
    OpenAI ChatAPIをたたくための薄いラッパークラス
    RAGChatServiceからは、complete(prompt: str)だけを意識
    """
    def __init__(self,api_key: str, model: str = "gpt-4.1-nano", temperature: float = 0.2) -> None:
        self.api_key = api_key
        self.model = model
        self.temparture = temperature
        self.client = get_client()

    def complete(self, prompt: str) -> str:
        """
        既に組み立てられたプロンプト文字列をそのまま userメッセージとして投げて、
        かえってきたテキストだけを返す。
        """
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt, 
                    }
                ],
                temperature=self.temparture,
        )
        choice = response.choices[0]
        answer_text = (choice.message.content or "").strip()
        return answer_text