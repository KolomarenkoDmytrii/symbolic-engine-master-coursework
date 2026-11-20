from __future__ import annotations

from typing import Any

from ...data_models import ExecutionResult, AnswerRequest
from .pipeline_service import PipelineService
from .planning_service import PlanningService
from .prolog_service import PrologService
from .answer_service import AnswerService


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
    ) -> ExecutionResult:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        history = history or []

        # 1. Build context
        # 1. Build graph and rule context.
        try:
            context = await self.pipeline_service.build_context(question)
        except LookupError as e:
            # If rules are missing, we don't crash. 
            # We proceed with an empty rule set so the LLM can still use the Graph.
            print(f"[ChatService] Warning: No rules found ({str(e)}). Proceeding with Graph only.")
            # We must manually construct a "dummy" context or modify PipelineService to handle this.
            # For now, let's assume we modify PipelineService to return empty artifacts on LookupError.
            raise e 
        except Exception as e:
            raise RuntimeError(f"Failed to build context: {str(e)}")

        # 2. Generate Plan
        try:
            plan = await self.planning_service.create_plan(
                question=question,
                signatures=context.signatures,
                graph_facts=context.graph_facts,
                artifacts=context.rule_files,
                allowed_atoms=context.allowed_atoms,  # Using the fix from previous task
            )
        except ValueError as e:
            # This is likely the "Invalid Atom" error.
            # We raise a specific error so the endpoint can return 422.
            raise ValueError(f"Planning Error (Possible invalid atom): {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Planning Engine failed: {str(e)}")

        # 3. Compile Goal
        try:
            goal = self.planning_service.make_goal(plan=plan, context=context)
        except Exception as e:
            raise ValueError(f"Goal Compilation failed: {str(e)}")

        # 4. Execute Prolog
        prolog_result = {
            "status": "exception",
            "error": "Execution skipped due to previous step failure.",
        }

        if goal:
            try:
                prolog_result = await self.prolog_service.execute(
                    context=context,
                    goal=goal,
                )
            except Exception as e:
                prolog_result = {
                    "status": "exception",
                    "error": f"Prolog execution error: {str(e)}",
                }

        # 5. Generate Answer
        try:
            request = AnswerRequest(
                question=question,
                graph_result=context.graph_result,
                signatures=context.signatures,
                goal=goal,
                prolog_result=prolog_result,
                history=history,
            )
            answer = await self.answer_service.generate(request)
        except Exception as e:
            answer = f"Error generating natural language answer: {str(e)}"

        return ExecutionResult(
            question=question,
            selected_rules=[a.artifact_id for a in context.rule_files],
            generated_facts=context.graph_facts,
            plan=plan,
            goal=goal,
            prolog_result=prolog_result,
            answer=answer,
        )

    async def ask(self, question: str) -> str | None:
        result = await self.query(question)
        return result.answer
