from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

from fastapi import HTTPException, Request

from .services import (
    AnswerService,
    ChatService,
    FactService,
    GraphService,
    PipelineService,
    PlanningService,
    PrologService,
    RuleService,
)

from ..config import api_settings, extraction_settings


@dataclass
class Services:
    graph: GraphService
    rules: RuleService
    facts: FactService
    planning: PlanningService
    prolog: PrologService
    answer: AnswerService
    pipeline: PipelineService
    chat: ChatService

    # Retain these so application shutdown can close infrastructure resources
    # when their implementations provide close() or aclose().
    llm: Any
    graph_repository: Any


def create_services() -> Services:
    llm = api_settings.create_llm()
    graph_repository = api_settings.create_graph_repository()

    graph = GraphService(graph_repository, extraction_settings)

    rules = RuleService(
        rules_dir=api_settings.rulers_directory,
        index_path=api_settings.rules_index,
        top_k=3,
    )

    facts = FactService(llm)
    planning = PlanningService(llm)
    prolog = PrologService(timeout_seconds=20)
    answer = AnswerService(llm)

    pipeline = PipelineService(
        graph_service=graph,
        rule_service=rules,
        fact_service=facts,
    )

    chat = ChatService(
        pipeline_service=pipeline,
        planning_service=planning,
        prolog_service=prolog,
        answer_service=answer,
    )

    return Services(
        graph=graph,
        rules=rules,
        facts=facts,
        planning=planning,
        prolog=prolog,
        answer=answer,
        pipeline=pipeline,
        chat=chat,
        llm=llm,
        graph_repository=graph_repository,
    )


def initialize_services(app: Any) -> None:
    """Create one service graph per application process."""
    app.state.services = create_services()


async def close_services(app: Any) -> None:
    """Close infrastructure resources when they expose a close method."""
    services: Services | None = getattr(app.state, "services", None)
    if services is None:
        return

    seen: set[int] = set()

    for resource in (services.llm, services.graph_repository):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))

        close = getattr(resource, "aclose", None)
        if close is None:
            close = getattr(resource, "close", None)
        if not callable(close):
            continue

        result = close()
        if inspect.isawaitable(result):
            await result

    del app.state.services


def _get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=503,
            detail="Application services are not initialized.",
        )
    return services


def get_chat_service(request: Request) -> ChatService:
    return _get_services(request).chat


def get_graph_service(request: Request) -> GraphService:
    return _get_services(request).graph


def get_rule_service(request: Request) -> RuleService:
    return _get_services(request).rules
