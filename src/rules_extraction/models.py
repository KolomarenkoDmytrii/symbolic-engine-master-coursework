from __future__ import annotations

import re
from typing import Sequence
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

# Deliberately limited Prolog term syntax: lowercase atoms, uppercase/underscore variables, and integer terms.
ATOM_RE = re.compile(r"^[a-z][A-Za-z0-9_]*$")
VARIABLE_RE = re.compile(r"^[A-Z_][A-Za-z0-9_]*$")
INTEGER_RE = re.compile(r"^-?\d+$")

ENTITY_PREDICATES = {
    "failsafe_mechanism",
    "failsafe_trigger",
    "failsafe_action",
    "flight_mode",
    "parameter",
    "vehicle_component",
    "sensor",
    "estimator",
    "communication_link",
    "mission_command",
    "configuration_value",
    "threshold",
    "timeout",
    "vehicle_state",
}

RELATION_PREDICATES = {
    "triggered_by",
    "monitors",
    "configured_by",
    "takes_action",
    "switches_to",
    "falls_back_to",
    "bypassed_by",
    "applies_to",
    "requires",
    "remains_in",
    "clears_when",
    "modifies",
}

CANONICAL_ARITY = {
    **{predicate: 1 for predicate in ENTITY_PREDICATES},
    **{predicate: 2 for predicate in RELATION_PREDICATES},
    "param_value": 2,
}


class Goal(BaseModel):
    """Represents a canonical domain predicate, e.g. sensor(X) or monitors(X, Y)."""

    model_config = ConfigDict(extra="forbid")

    predicate: str
    args: list[str]

    @field_validator("predicate")
    @classmethod
    def validate_predicate(cls, value: str) -> str:
        """Validate that the predicate belongs to the canonical vocabulary."""
        if value not in CANONICAL_ARITY:
            raise ValueError(f"Predicate is not in the canonical vocabulary: {value!r}")
        return value

    @field_validator("args")
    @classmethod
    def validate_arguments(cls, values: list[str]) -> list[str]:
        """Validate that each argument conforms to supported Prolog term syntax."""
        for value in values:
            if not (
                ATOM_RE.fullmatch(value)
                or VARIABLE_RE.fullmatch(value)
                or INTEGER_RE.fullmatch(value)
            ):
                raise ValueError(f"Unsupported Prolog term: {value!r}")
        return values

    @model_validator(mode="after")
    def validate_arity(self) -> Goal:
        """Validate that the number of arguments matches the canonical arity for the predicate."""
        expected = CANONICAL_ARITY[self.predicate]
        if len(self.args) != expected:
            raise ValueError(
                f"{self.predicate}/{expected} requires {expected} argument(s); "
                f"got {len(self.args)}."
            )
        return self


class Rule(BaseModel):
    """Represents a Horn-clause rule with disjunction of conjunctions in the body."""

    model_config = ConfigDict(extra="forbid")

    head: Goal
    alternatives: list[list[Goal]]
    evidence: str

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: str) -> str:
        """Ensure that supporting source evidence is provided and non-empty."""
        value = value.strip()
        if not value:
            raise ValueError("Each rule needs supporting source evidence.")
        return value

    @model_validator(mode="after")
    def validate_rule(self) -> Rule:
        """Validate rule structure and the head-variable binding requirement."""
        if not self.alternatives:
            raise ValueError("A rule needs at least one alternative.")

        head_variables = {
            arg for arg in self.head.args if VARIABLE_RE.fullmatch(arg) and arg != "_"
        }

        for index, conjunction in enumerate(self.alternatives):
            if not conjunction:
                raise ValueError(
                    f"Alternative {index + 1} must contain at least one goal."
                )

            body_variables = {
                arg
                for goal in conjunction
                for arg in goal.args
                if VARIABLE_RE.fullmatch(arg) and arg != "_"
            }

            if not head_variables.issubset(body_variables):
                raise ValueError(
                    f"Every variable in the rule head must appear in "
                    f"alternative {index + 1}."
                )

        return self


class RuleBatch(BaseModel):
    """Container model for a batch of extracted rules and source summary."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    rules: list[Rule]


class DocumentMetadata(BaseModel):
    """Container model for document description and domain metadata."""

    model_config = ConfigDict(extra="forbid")

    description: str
    domain: str
