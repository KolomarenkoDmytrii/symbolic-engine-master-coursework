from __future__ import annotations

from src.data_models import PipelineContext

from src.api.services.graph_service import GraphService
from src.api.services.rule_service import RuleService
from src.api.services.fact_service import FactService


class PipelineService: # pylint: disable=too-few-public-methods

    def __init__(
        self,
        graph_service: GraphService,
        rule_service: RuleService,
        fact_service: FactService,
    ):
        self.graph_service = graph_service
        self.rule_service = rule_service
        self.fact_service = fact_service

    async def build_context(
        self,
        question: str,
    ) -> PipelineContext:

        # 1. Retrieve relevant graph information.
        graph_result = await self.graph_service.search(question)

        # 2. Select relevant rule artifacts.
        artifacts, signatures = self.rule_service.select_rules(question)

        # 3. Serialize normalized graph context.
        graph_state = self.graph_service.serialize(graph_result)

        # 4. Generate Prolog facts.
        graph_facts = await self.fact_service.generate(
            question=question,
            graph_state=graph_state,
            artifacts=artifacts,
        )

        # 5. Extract graph atoms.
        graph_atoms = self.graph_service.get_atoms(graph_result)

        return PipelineContext(
            graph_result=graph_result,
            graph_facts=graph_facts,
            rule_files=artifacts,
            signatures=signatures,
            graph_atoms=graph_atoms,
        )
