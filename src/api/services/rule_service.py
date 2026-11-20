from __future__ import annotations

import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

from ...data_models import RuleArtifact
from ...rules_extraction.pipeline import process_file
from ...rules_extraction.extraction import build_chains
from ...config import api_settings

logger = logging.getLogger(__name__)


def tokenize(text: str) -> Set[str]:
    """Tokenizes text for simple keyword-based ranking."""
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

    def _rank_records(
        self,
        records: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Ranks records based on token overlap with the query."""
        query_tokens = tokenize(query)
        scored: List[Tuple[int, Dict[str, Any]]] = []

        for record in records:
            if not record.get("rule_count"):
                continue

            searchable = " ".join(
                str(record.get(key, ""))
                for key in ("source_path", "domain", "description")
            )
            score = len(query_tokens & tokenize(searchable))
            if score > 0:
                scored.append((score, record))

        # Sort by score descending
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _build_artifact(self, record: Dict[str, Any]) -> RuleArtifact:
        artifact_id = record.get("source_path")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("Rule record needs source_path.")

        intents = record.get("intents", [])
        if not isinstance(intents, list):
            intents = []

        prolog_path = self._resolve_path(record.get("prolog_path"))
        provenance_path = self._resolve_path(record.get("provenance_path"))

        if prolog_path is None or provenance_path is None:
            raise ValueError(f"Missing required paths in record: {record}")

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
        signature_map: Dict[Tuple[str, int], Dict[str, Any]],
        signature: Dict[str, Any],
    ) -> None:
        """Merges signature into the map if it shares predicate and arity."""
        key = (signature["predicate"], signature["arity"])
        existing = signature_map.get(key)

        if existing is None:
            signature_map[key] = signature
            return

        # Merge sets of values to avoid duplicates
        for field in ("artifacts", "domains", "evidence"):
            new_values = signature.get(field, [])
            for val in new_values:
                if val not in existing[field]:
                    existing[field].append(val)

    def _load_index(self) -> List[Dict[str, Any]]:
        """Safely loads the rule index."""
        if not self.index_path.exists():
            logger.error(f"Rule index missing at: {self.index_path}")
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return (
                    data.get("rules")
                    or data.get("files")
                    or data.get("artifacts")
                    or []
                )
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse index JSON: {str(e)}")
            raise ValueError(f"Invalid index format: {str(e)}")

    def _resolve_path(self, value: str | None, required: bool = True) -> Path | None:
        """Safely resolves paths relative to the rules directory."""
        if not value:
            if required:
                raise ValueError("Missing required path.")
            return None

        candidate = (self.rules_dir / Path(value)).resolve()

        if not candidate.is_file():
            if required:
                logger.warning(f"Required file not found: {candidate}")
                raise FileNotFoundError(f"File not found: {candidate}")
            return None
        return candidate

    def _extract_signatures(self, artifact: RuleArtifact) -> List[Dict[str, Any]]:
        """Extracts signatures from the provenance file."""
        try:
            if not artifact.provenance_path.exists():
                logger.warning(f"Provenance file missing for {artifact.artifact_id}")
                return []

            provenance = json.loads(
                artifact.provenance_path.read_text(encoding="utf-8")
            )
            signatures = []

            for rule in provenance.get("rules", []):
                head = rule.get("head")
                if not head or not isinstance(head, dict):
                    continue

                predicate = head.get("predicate")
                args = head.get("args")

                if not predicate or not isinstance(args, list):
                    continue

                if not re.fullmatch(r"[a-z][A-Za-z0-9_]*", predicate):
                    continue

                signatures.append(
                    {
                        "predicate": predicate,
                        "args": args,
                        "arity": len(args),
                        "artifacts": [artifact.artifact_id],
                        "domains": [artifact.domain] if artifact.domain else [],
                        "evidence": (
                            [rule.get("evidence")] if rule.get("evidence") else []
                        ),
                    }
                )
            return signatures

        except Exception as e:
            logger.error(
                f"Error extracting signatures for {artifact.artifact_id}: {str(e)}"
            )
            return []

    def select_rules(
        self, question: str
    ) -> Tuple[List[RuleArtifact], List[Dict[str, Any]]]:
        """Main entry point for rule selection via keyword search."""
        records = self._load_index()
        selected_records = self._rank_records(records, question, self.top_k)

        if not selected_records:
            raise LookupError("No matching rules found for the given query.")

        artifacts: List[RuleArtifact] = []
        signature_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for record in selected_records:
            try:
                artifact = self._build_artifact(record)
                artifacts.append(artifact)

                for sig in self._extract_signatures(artifact):
                    self._merge_signature(signature_map, sig)
            except (FileNotFoundError, ValueError) as e:
                logger.error(
                    f"Skipping broken rule record {record.get('source_path')}: {str(e)}"
                )
                continue

        signatures = list(signature_map.values())
        if not signatures:
            raise LookupError("Selected rules contain no valid signatures.")

        return artifacts, signatures

    def list_rules(self) -> List[Dict[str, Any]]:
        return self._load_index()

    def get_rule(self, artifact_id: str) -> RuleArtifact:
        """Directly fetches a specific rule artifact by its ID."""
        records = self._load_index()
        for record in records:
            if record.get("source_path") == artifact_id:
                return self._build_artifact(record)

        raise LookupError(f"Rule artifact not found: {artifact_id}")

    async def run_extraction_pipeline(
        self, docs_dir: Path
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Orchestrates the rule extraction process and updates the index.json."""
        extractor, metadata_chain = build_chains(
            api_settings.base_url, api_settings.model_name
        )

        rst_files = list(docs_dir.rglob("*.rst"))
        all_extracted_rules_metadata = []
        signature_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        index_records = []

        for source in rst_files:
            try:
                result = process_file(
                    source=source,
                    input_root=docs_dir,
                    output_root=self.rules_dir,
                    extractor=extractor,
                    metadata_chain=metadata_chain,
                )
                all_extracted_rules_metadata.append(result)

                prov_path = Path(result["provenance_path"])
                if prov_path.exists():
                    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))

                    # Build Signatures
                    for rule_data in prov_data.get("rules", []):
                        head = rule_data.get("head", {})
                        predicate = head.get("predicate")
                        args = head.get("args", [])
                        if predicate and args:
                            sig = {
                                "predicate": predicate,
                                "args": args,
                                "arity": len(args),
                                "artifacts": [result["source_path"]],
                                "domains": [prov_data.get("domain", "unspecified")],
                                "evidence": (
                                    [rule_data.get("evidence", "")]
                                    if rule_data.get("evidence")
                                    else []
                                ),
                            }
                            self._merge_signature(signature_map, sig)

                    # Build Index Record
                    index_record = {
                        "source_path": result["source_path"],
                        "prolog_path": str(prov_path.with_suffix("")),
                        "provenance_path": str(prov_path),
                        "description": prov_data.get(
                            "description", "No description available."
                        ),
                        "domain": prov_data.get("domain", "N/A"),
                        "status": (
                            "complete"
                            if len(prov_data.get("rules", [])) > 0
                            else "partial"
                        ),
                        "rule_count": len(prov_data.get("rules", [])),
                        "source_sha256": "unknown",
                        "chunk_errors": prov_data.get("chunk_errors", []),
                        "generated_at": prov_data.get(
                            "generated_at", datetime.now().isoformat()
                        ),
                    }
                    index_records.append(index_record)

            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                continue

        if index_records:
            index_file_path = self.rules_dir / "index.json"
            try:
                with open(index_file_path, "w", encoding="utf-8") as f:
                    json.dump(index_records, f, indent=4)
                logger.info(f"Successfully updated index at {index_file_path}")
            except Exception as e:
                logger.error(f"Failed to write index.json: {e}")
        else:
            logger.warning("No rules were extracted. index.json was not updated.")

        return all_extracted_rules_metadata, list(signature_map.values())
