from __future__ import annotations

import json
import math
import re
from typing import Any

from langchain_core.messages import HumanMessage

from ...data_models import PipelineContext
from .prolog_syntax import safe_atom


VARIABLE_NAME = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


def response_text(response: Any) -> str:

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )

    return str(content)

def _validate_plan_argument(argument: dict) -> None:
    if not isinstance(argument, dict) or len(argument) != 1:
        raise ValueError("Each argument must contain exactly one kind.")

    kind, item = next(iter(argument.items()))
    if kind == "var":
        if not isinstance(item, str) or not VARIABLE_NAME.fullmatch(item):
            raise ValueError("Invalid planner variable.")
    elif kind == "atom":
        if not isinstance(item, str) or not item:
            raise ValueError("Invalid planner atom.")
    elif kind == "number":
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or (isinstance(item, float) and not math.isfinite(item))
        ):
            raise ValueError("Invalid planner number.")
    else:
        raise ValueError(f"Unsupported argument kind: {kind!r}")

def parse_plan_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Planner response must be one JSON object.") from exc

    if not isinstance(value, dict):
        raise ValueError("Planner response must be a JSON object.")

    if set(value) != {"predicate", "arguments"}:
        raise ValueError("Planner JSON has unexpected or missing fields.")

    if not isinstance(value["predicate"], str):
        raise ValueError("Planner predicate must be a string.")

    if not isinstance(value["arguments"], list):
        raise ValueError("Planner arguments must be a list.")

    for argument in value["arguments"]:
        _validate_plan_argument(argument)

    return value


# def safe_atom(value: str) -> str:

#     escaped = []

#     for char in value:

#         codepoint = ord(char)

#         if char == "\\":
#             escaped.append("\\\\")

#         elif char == "'":
#             escaped.append("''")

#         elif char == "\n":
#             escaped.append("\\n")

#         elif char == "\r":
#             escaped.append("\\r")

#         elif char == "\t":
#             escaped.append("\\t")

#         elif codepoint < 32 or codepoint == 127:
#             escaped.append(f"\\x{codepoint:x}\\")

#         else:
#             escaped.append(char)

#     return "'" + "".join(escaped) + "'"


def extract_prolog_atoms(*texts: str) -> set[str]:
    combined = "\n".join(texts)
    combined = re.sub(
        r"%[^\n]*|/\*.*?\*/",
        "",
        combined,
        flags=re.DOTALL,
    )

    # Improved regex for quoted atoms
    quoted = set(
        re.findall(
            r"'((?:\\.|[^'\\])*)'",
            combined,
        )
    )

    # Improved regex for bare atoms to be more inclusive of Prolog naming
    # Ensures we don't just get single letters but valid predicate/atom names
    bare = set(
        re.findall(
            r"\b[a-z][A-Za-z0-9_]*\b",
            combined,
        )
    )

    return quoted | bare


class PlanningService:

    def __init__(self, llm: Any):
        self.llm = llm

    async def create_plan(
        self,
        question: str,
        signatures: list[dict[str, Any]],
        graph_facts: str,
        artifacts: list[Any],
        allowed_atoms: set[str],
    ) -> dict[str, Any]:

        prompt = f"""
Choose one Prolog predicate that best answers the question.

Return exactly one JSON object.

Required format:
{{
    "predicate": "predicate_name",
    "arguments": [
        {{"atom": "autopilot"}},
        {{"var": "X"}},
        {{"number": 12}}
    ]
}}

Rules:

- Use only predicates from the supplied signatures.
- The number of arguments must match the signature arity.
- Use lowercase words as atoms, not variables.
- Variables must begin with uppercase letters.
- Use constants only if they occur in graph facts,
  generated facts or selected rules.
- Do not invent facts or constants.
- Return no explanations or Markdown.

Allowed signatures:
{json.dumps(signatures, ensure_ascii=False, indent=2)}

Generated graph facts:
{graph_facts}

Allowed atoms (constants/facts):
{", ".join(sorted(allowed_atoms))}

Selected rules:
{json.dumps(
    [
        {
            "artifact_id": item.artifact_id,
            "domain": item.domain,
        }
        for item in artifacts
    ],
    ensure_ascii=False,
)}

User question:
{question}
""".strip()

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        return parse_plan_json(response_text(response))

    @staticmethod
    def _render_argument(
        item: dict[str, Any],
        allowed_atoms: set[str],
    ) -> str:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Invalid plan argument.")

        if "atom" in item:
            atom = item["atom"]
            if not isinstance(atom, str):
                raise ValueError("Atom must be a string.")
            if atom not in allowed_atoms:
                raise ValueError(f"Unknown Prolog atom: {atom!r}")
            return safe_atom(atom)

        if "var" in item:
            variable = item["var"]
            if not isinstance(variable, str) or not VARIABLE_NAME.fullmatch(variable):
                raise ValueError("Invalid variable name.")
            return variable

        if "number" in item:
            number = item["number"]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError("Invalid number.")
            if isinstance(number, float) and not math.isfinite(number):
                raise ValueError("Invalid numeric value.")
            return repr(number)

        raise ValueError("Argument must contain atom, var or number.")

    def make_goal(
        self,
        plan: dict[str, Any],
        context: PipelineContext,
    ) -> str:

        predicate = plan.get("predicate")
        arguments = plan.get("arguments")

        if not isinstance(predicate, str):
            raise ValueError("Invalid predicate.")

        if not isinstance(arguments, list):
            raise ValueError("Invalid arguments.")

        if not re.fullmatch(
            r"[a-z][A-Za-z0-9_]*",
            predicate,
        ):
            raise ValueError("Invalid predicate name.")

        matching = [
            signature
            for signature in context.signatures
            if (
                signature["predicate"] == predicate
                and signature["arity"] == len(arguments)
            )
        ]
        if not matching:
            raise ValueError(f"No allowed signature for {predicate}/{len(arguments)}.")

        allowed_atoms = set(context.graph_atoms)
        allowed_atoms.update(extract_prolog_atoms(context.graph_facts))

        for artifact in context.rule_files:
            source = artifact.prolog_path.read_text(encoding="utf-8")
            allowed_atoms.update(extract_prolog_atoms(source))

        rendered = [self._render_argument(item, allowed_atoms) for item in arguments]

        if not rendered:
            return predicate

        return f"{predicate}({', '.join(rendered)})"
