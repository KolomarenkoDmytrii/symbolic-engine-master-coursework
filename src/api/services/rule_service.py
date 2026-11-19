from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.data_models import RuleArtifact


def tokenize(text: str) -> set[str]:
    ignored = {
        "the",
        "and",
        "for",
        "with",
        "how",
        "can",
        "mode",
        "copter",
    }

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        if len(token) > 2 and token not in ignored
    }


class RuleService:

    def __init__(
        self,
        rules_dir: Path,
        index_path: Path,
        top_k: int = 3,
    ):
        self.rules_dir = rules_dir.resolve()
        self.index_path = index_path.resolve()
        self.top_k = top_k

    @staticmethod
    def _rank_records(
        records: list[dict[str, Any]],
        question: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize(question)
        scored = []

        for record in records:
            if not record.get("rule_count"):
                continue

            searchable = " ".join(
                str(record.get(key, ""))
                for key in ("source_path", "domain", "description")
            )
            score = len(query_tokens & tokenize(searchable))
            if score:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _build_artifact(self, record: dict[str, Any]) -> RuleArtifact:
        artifact_id = record.get("source_path")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("Rule record needs source_path.")

        intents = record.get("intents", [])
        if not isinstance(intents, list):
            intents = []

        prolog_path = self._resolve_path(record.get("prolog_path"))
        provenance_path = self._resolve_path(record.get("provenance_path"))

        assert prolog_path is not None
        assert provenance_path is not None

        return RuleArtifact(
            artifact_id=artifact_id,
            prolog_path=prolog_path,
            provenance_path=provenance_path,
            domain=str(record.get("domain", "")),
            intents=tuple(map(str, intents)),
            source_path=artifact_id,
        )

    @staticmethod
    def _merge_signature(
        signature_map: dict[tuple[str, int], dict[str, Any]],
        signature: dict[str, Any],
    ) -> None:
        key = (signature["predicate"], signature["arity"])
        existing = signature_map.get(key)

        if existing is None:
            signature_map[key] = signature
            return

        for field in ("artifacts", "domains", "evidence"):
            for value in signature[field]:
                if value not in existing[field]:
                    existing[field].append(value)

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Rule index not found: {self.index_path}")

        data = json.loads(self.index_path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            records = data

        elif isinstance(data, dict):
            records = (
                data.get("rules")
                or data.get("files")
                or data.get("artifacts")
                or data.get("items")
            )

        else:
            records = None

        if not isinstance(records, list):
            raise ValueError("Invalid rule index format.")

        return [item for item in records if isinstance(item, dict)]

    def _resolve_path(
        self,
        value: str,
        required: bool = True,
    ) -> Path | None:

        if not value:
            if required:
                raise ValueError("Missing rule artifact path.")
            return None

        candidate = Path(value)

        if not candidate.is_absolute():
            candidate = self.rules_dir / candidate

        candidate = candidate.resolve()

        try:
            candidate.relative_to(self.rules_dir)

        except ValueError as exc:
            raise ValueError(
                f"Rule path outside rulers_directory: {candidate}"
            ) from exc

        if not candidate.is_file():

            if required:
                raise FileNotFoundError(candidate)

            return None

        return candidate

    def _extract_signatures(
        self,
        artifact: RuleArtifact,
    ) -> list[dict[str, Any]]:

        provenance = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))

        if not isinstance(provenance, dict):
            raise ValueError(f"Invalid provenance: {artifact.artifact_id}")

        signatures = []

        for rule in provenance.get("rules", []):

            if not isinstance(rule, dict):
                continue

            head = rule.get("head")

            if not isinstance(head, dict):
                continue

            predicate = head.get("predicate")
            args = head.get("args")

            if (
                not isinstance(predicate, str)
                or not re.fullmatch(
                    r"[a-z][A-Za-z0-9_]*",
                    predicate,
                )
                or not isinstance(args, list)
            ):
                continue

            signatures.append(
                {
                    "predicate": predicate,
                    "args": args,
                    "arity": len(args),
                    "artifacts": [artifact.artifact_id],
                    "domains": ([artifact.domain] if artifact.domain else []),
                    "evidence": (
                        [rule["evidence"]] if rule.get("evidence") is not None else []
                    ),
                }
            )

        return signatures

    def select_rules(
        self,
        question: str,
    ) -> tuple[list[RuleArtifact], list[dict[str, Any]]]:
        selected = self._rank_records(
            self._load_index(),
            question,
            self.top_k,
        )
        if not selected:
            raise LookupError("No matching rule artifacts found.")

        artifacts: list[RuleArtifact] = []
        signature_map: dict[tuple[str, int], dict[str, Any]] = {}

        for record in selected:
            artifact = self._build_artifact(record)
            artifacts.append(artifact)

            for signature in self._extract_signatures(artifact):
                self._merge_signature(signature_map, signature)

        signatures = list(signature_map.values())
        if not signatures:
            raise LookupError("Selected rules contain no usable signatures.")

        return artifacts, signatures

    def list_rules(self) -> list[dict[str, Any]]:
        return self._load_index()

    def get_rule(
        self,
        artifact_id: str,
    ) -> RuleArtifact:

        artifacts, _ = self.select_rules(artifact_id)

        for artifact in artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact

        raise LookupError(f"Rule artifact not found: {artifact_id}")
