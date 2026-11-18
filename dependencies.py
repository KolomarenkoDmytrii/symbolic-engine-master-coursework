from dataclasses import dataclass

from src.config import settings

from .api.services.graph_service import GraphService
from .api.services.rule_service import RuleService
from .api.services.fact_service import FactService
from .api.services.planning_service import PlanningService
from .api.services.prolog_service import PrologService
from .api.services.answer_service import AnswerService
from .api.services.pipeline_service import PipelineService
from .api.services.chat_service import ChatService


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

    # Infrastructure initialization.
    llm = settings.create_llm()
    graph_repository = settings.create_graph_repository()
    rules_repository = settings.create_rules_repository()

    # Domain services.
    graph = GraphService(graph_repository)
    rules = RuleService(rules_repository)
    facts = FactService(llm)
    planning = PlanningService(llm)
    prolog = PrologService()
    answer = AnswerService(llm)

    # Pipeline orchestration.
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
    )
