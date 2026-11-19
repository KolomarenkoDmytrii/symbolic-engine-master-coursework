
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuleArtifact:
    artifact_id: str
    prolog_path: Path
    provenance_path: Path
    domain: str
    intents: tuple[str, ...]
    source_path: str


@dataclass
class PipelineContext:
    graph_result: dict[str, Any]
    graph_facts: str
    rule_files: list[RuleArtifact]
    signatures: list[dict[str, Any]]
    graph_atoms: set[str]


@dataclass
class ExecutionResult:
    question: str
    selected_rules: list[str]
    generated_facts: str
    plan: dict[str, Any]
    goal: str
    prolog_result: dict[str, Any]
    answer: str | None = None

@dataclass
class AnswerRequest:
    question: str
    graph_result: dict[str, Any]
    signatures: list[dict[str, Any]]
    goal: str
    prolog_result: dict[str, Any]
    history: list[tuple[str, str]] | None = None