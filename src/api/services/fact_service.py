from __future__ import annotations

from typing import Any

import re
import json

from pydantic import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from ...data_models import RuleArtifact
from ..schemas import FactEntry, FactBatch
from .prolog_syntax import extract_rule_body_signatures, safe_atom

FACT_SYSTEM_PROMPT = """
Generate ground facts based on the provided vehicle state and scenario.
Return the facts as a structured JSON list.

Requirements:
1. Output ONLY a JSON object containing a list of facts.
2. Each fact must have a 'predicate' (string) and 'args' (list of strings).
3. All arguments must be ground (no variables like X, Y, or _).
4. Use only the allowed predicates and arity provided in the context.
5. Use lowercase snake_case for atoms.
6. If a value is a number, provide it as a string representation of the number.
7. Do not invent facts; only use information supported by the context.
8. Omit any facts that cannot be fully determined.
"""


def extract_body_predicates(
    prolog_text: str,
) -> set[str]:
    predicates: set[str] = set()
    prolog_text = re.sub(
        r"%[^\n]*|/\*.*?\*/",
        "",
        prolog_text,
        flags=re.DOTALL,
    )
    for clause in re.findall(
        r"(?s)(.*?\.)",
        prolog_text,
    ):
        if ":-" not in clause:
            continue
        _, body = clause.split(":-", 1)
        predicates.update(
            re.findall(
                r"(?<![A-Za-z0-9_])([a-z][A-Za-z0-9_]*)\s*\(",
                body,
            )
        )
    return predicates


def extract_body_signatures(prolog_text: str) -> set[tuple[str, int]]:
    return extract_rule_body_signatures(prolog_text)


def construct_prolog_from_json(
    batch: FactBatch,
    allowed_signatures: set[tuple[str, int]],
    allowed_atoms: set[str] | None = None,
) -> str:
    """
    Converts structured FactBatch into a validated Prolog string.
    This replaces the old 'validate_generated_facts' that relied on parsing raw text.
    """
    checked: list[str] = []

    for entry in batch.facts:
        predicate = entry.predicate
        args = entry.args
        signature = (predicate, len(args))

        # 1. Validate Predicate and Arity
        if signature not in allowed_signatures:
            raise ValueError(f"Predicate or arity not allowed: {predicate}/{len(args)}")

        # 2. Validate Atoms (if provided)
        if allowed_atoms is not None:
            for arg in args:
                # Check if it looks like an atom (not a number)
                if not re.match(r"^-?\d+(\.\d+)?$", arg):
                    if arg not in allowed_atoms:
                        raise ValueError(f"Unknown atom: {arg!r}")

        # 3. Construct the ground fact string safely
        # We use safe_atom for string arguments to ensure correct Prolog quoting
        rendered_args = []
        for arg in args:
            if re.match(r"^-?\d+(\.\d+)?$", arg):
                rendered_args.append(arg)
            else:
                rendered_args.append(safe_atom(arg))

        if rendered_args:
            checked.append(f"{predicate}({', '.join(rendered_args)}).")
        else:
            checked.append(f"{predicate}.")

    return "\n".join(checked)


class FactService:  # pylint: disable=too-few-public-methods

    def __init__(self, llm: Any):
        self.llm = llm

    async def generate(
        self,
        question: str,
        graph_state: str,
        artifacts: list[RuleArtifact],
    ) -> str:

        allowed_signatures: set[tuple[str, int]] = set()

        for artifact in artifacts:
            text = artifact.prolog_path.read_text(encoding="utf-8")
            allowed_signatures.update(extract_rule_body_signatures(text))

        if not allowed_signatures:
            raise LookupError("Selected rules contain no body predicates.")

        # Use structured output to force JSON format
        structured_llm = self.llm.with_structured_output(FactBatch)

        payload = {
            "question": question,
            "graph_state": graph_state,
            "selected_rules": [a.prolog_path.name for a in artifacts],
            "allowed_signatures": [
                {"predicate": p, "arity": a} for p, a in sorted(allowed_signatures)
            ],
        }

        response = await structured_llm.ainvoke(
            [
                SystemMessage(content=FACT_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )

        # Construct the Prolog string from the validated JSON
        return construct_prolog_from_json(
            response,
            allowed_signatures,
        )
