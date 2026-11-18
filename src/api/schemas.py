from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConsultRequest(BaseModel):
    """Схема запиту для завантаження правил або текстів у сесію PrologMCP."""

    source_code: str = Field(
        ..., description="Декларативні правила або факти у синтаксисі SWI-Prolog."
    )
    dialect: str = Field(
        default="swi", description="Діалект Prolog (за замовчуванням swi)."
    )


class ConsultResponse(BaseModel):
    """Схема відповіді при успішному завантаженні правил."""

    status: str
    load_id: str
    message: str


class GoalExecutionRequest(BaseModel):
    """Схема запиту для виконання логічної цілі[cite: 4]."""

    load_id: str = Field(..., description="Ідентифікатор активної сесії PrologMCP.")
    goal: str = Field(
        ..., description="Цільовий запит для перевірки логічного виведення."
    )


class GoalExecutionResponse(BaseModel):
    """Схема відповіді із результатами виконання цілі[cite: 4]."""

    status: str
    result: Dict[str, Any]
    error: Optional[str] = None


class GraphQueryRequest(BaseModel):
    """Схема запиту до графової бази даних FalkorDB."""

    query: str = Field(..., description="Cypher-запит до графа.")
    graph_name: str = Field(default="copter", description="Назва графа в FalkorDB.")
