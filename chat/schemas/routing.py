from __future__ import annotations

from typing import List,Annotated
from pydantic import BaseModel, Field, model_validator


class RoutingResult(BaseModel):
    is_business: bool
    business_confidence: Annotated[float,Field(ge=0.0, le=1.0)]

    primary_department: str # 例： "finance" / "hr" / "legal" / "it" / "unknown"
    department_confidence: Annotated[float,Field(ge=0.0, le=1.0)] 
    secondary_departments: List[str] = Field(default_factory=list)

    needs_clarification: bool
    clarifying_question: str = ""

    @model_validator(mode="after")
    def _validate_consistency(self) -> "RoutingResult":
        # clarification が必要なら、質問は必須なので制御
        if self.needs_clarification and not self.clarifying_question.strip():
            raise ValueError("needs_clarification=true の場合は、clarifying_question は必須です。")
        
        # secondaryの重複除去  + primary除外(安全側)
        sec = []
        for d in self.secondary_departments:
            if not d:
                continue
            if d == self.primary_department:
                continue
            if d not in sec:
                sec.append(d)
        self.secondary_departments = sec

        # primary が空は不可(unknownに寄せたい場合はunknownを返却させる)
        if not self.primary_department.strip():
            raise ValueError("primary_department は空にできません。（unknownを使用してください。）")
        
        return self