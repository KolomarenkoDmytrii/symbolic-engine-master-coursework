from __future__ import annotations

import logging

from ...data_models import PipelineContext

from .graph_service import GraphService
from .rule_service import RuleService
from .fact_service import FactService
from .prolog_syntax import extract_prolog_atoms


class PipelineService:  # pylint: disable=too-few-public-methods

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
        artifacts = []
        signatures = []
        try:
            artifacts, signatures = self.rule_service.select_rules(question)
        except LookupError:
            # SILENT RECOVERY: If no rules match the question (or index is missing),
            # we continue with empty artifacts so the LLM can still use the Graph.
            print("[PipelineService] No relevant rules found. Proceeding with Graph context only.")
            artifacts = []
            signatures = []
        except Exception as e:
            # Other errors (like FileNotFoundError for the index) should still be handled
            print(f"[PipelineService] Critical error in rule selection: {str(e)}")
            # We decide here: do we want to crash or proceed? 
            # For a robust reasoning agent, proceeding with empty rules is better.
            artifacts = []
            signatures = []


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

        # NEW: Centralized Atom Extraction for the whole pipeline
        allowed_atoms = set(graph_atoms)
        allowed_atoms.update(extract_prolog_atoms(graph_facts))
        for artifact in artifacts:
            source = artifact.prolog_path.read_text(encoding="utf-8")
            allowed_atoms.update(extract_prolog_atoms(source))

        return PipelineContext(
            graph_result=graph_result,
            graph_facts=graph_facts,
            rule_files=artifacts,
            signatures=signatures,
            graph_atoms=graph_atoms,
            allowed_atoms=allowed_atoms,  # Pass to context
        )
