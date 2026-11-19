from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=12_000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value

class GraphSearchRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank.")
        return value


class GraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    relations: list[dict[str, Any]]


class RuleSelectionRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class RuleSelectionResponse(BaseModel):
    selected_rules: list[dict[str, Any]]
    signatures: list[dict[str, Any]]


class FactGenerationRequest(BaseModel):
    question: str = Field(min_length=1)
    graph_state: dict[str, Any]


class PlanRequest(BaseModel):
    question: str
    signatures: list[dict[str, Any]]
    graph_facts: str = ""


class PrologExecutionRequest(BaseModel):
    goal: str = Field(min_length=1)


class RuleArtifactResponse(BaseModel):
    artifact_id: str
    domain: str
    intents: list[str] = Field(default_factory=list)
    source_path: str = ""


class ExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question: str
    selected_rules: list[str]
    generated_facts: str
    plan: dict[str, Any]
    goal: str
    prolog_result: dict[str, Any]
    answer: str | None = None
