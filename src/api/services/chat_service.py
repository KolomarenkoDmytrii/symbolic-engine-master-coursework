from __future__ import annotations

from typing import Any

from src.api.services.pipeline_service import PipelineService
from src.api.services.planning_service import PlanningService
from src.api.services.prolog_service import PrologService
from src.api.services.answer_service import AnswerService


class ChatService:

    def __init__(
        self,
        pipeline_service: PipelineService,
        planning_service: PlanningService,
        prolog_service: PrologService,
        answer_service: AnswerService,
    ):

        self.pipeline_service = pipeline_service
        self.planning_service = planning_service
        self.prolog_service = prolog_service
        self.answer_service = answer_service

    async def query(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        history = history or []

        # STEP 1: Build graph and rule context.
        context = await self.pipeline_service.build_context(question)

        # STEP 2: Generate constrained query plan.
        plan = await self.planning_service.create_plan(
            question=question,
            signatures=context.signatures,
            graph_facts=context.graph_facts,
            artifacts=context.rule_files,
        )

        # STEP 3: Compile and validate Prolog goal.
        goal = self.planning_service.make_goal(
            plan=plan,
            context=context,
        )

        # STEP 4: Execute Prolog query.
        prolog_result = await self.prolog_service.execute(
            context=context,
            goal=goal,
        )

        # STEP 5: Generate natural-language answer.
        answer = await self.answer_service.generate(
            question=question,
            graph_result=context.graph_result,
            signatures=context.signatures,
            goal=goal,
            prolog_result=prolog_result,
            history=history,
        )

        return {
            "question": question,
            "selected_rules": [artifact.artifact_id for artifact in context.rule_files],
            "generated_facts": context.graph_facts,
            "plan": plan,
            "goal": goal,
            "prolog_result": prolog_result,
            "answer": answer,
        }

    async def ask(
        self,
        question: str,
    ) -> str:

        result = await self.query(question)

        return result["answer"]
