
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)


class GraphSearchRequest(BaseModel):
    query: str = Field(min_length=1)


class FactGenerationRequest(BaseModel):
    question: str = Field(min_length=1)
    graph_state: dict[str, Any]


class PlanRequest(BaseModel):
    question: str
    signatures: list[dict[str, Any]]
    graph_facts: str = ""


class PrologExecutionRequest(BaseModel):
    goal: str = Field(min_length=1)


class ExecutionResponse(BaseModel):
    status: str
    result: dict[str, Any]
    answer: str | None = None


class GraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    relations: list[dict[str, Any]]


class RuleArtifactResponse(BaseModel):
    artifact_id: str
    domain: str
    intents: list[str]
    source_path: str
