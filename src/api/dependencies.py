from dataclasses import dataclass

from src.api.services import (
    GraphService,
    RuleService,
    FactService,
    PlanningService,
    PrologService,
    AnswerService,
    PipelineService,
    ChatService,
)

from src.config import settings


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


def create_services() -> Services:

    llm = settings.create_llm()

    # Infrastructure.
    graph_repository = settings.create_graph_repository()

    # Individual services.
    graph = GraphService(graph_repository)

    rules = RuleService(
        rules_dir=settings.RULES_DIR,
        index_path=settings.INDEX_JSON,
        top_k=3,
    )

    facts = FactService(llm)

    planning = PlanningService(llm)

    prolog = PrologService(
        timeout_seconds=20,
    )

    answer = AnswerService(llm)

    # Pipeline.
    pipeline = PipelineService(
        graph_service=graph,
        rule_service=rules,
        fact_service=facts,
    )

    # Main application service.
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
    )
