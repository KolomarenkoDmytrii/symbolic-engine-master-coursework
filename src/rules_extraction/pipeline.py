from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reusing submodules from docs module
from ..docs.text_extraction import RstBlockExtractor, TextChunker
from ..config import RstSettings

from .models import Rule
from .extractor import invoke_extractor, timed_invoke, fallback_metadata
from .compiler import render_prolog, atomic_write_text, atomic_write_json


def process_file(
    source: Path,
    input_root: Path,
    output_root: Path,
    extractor: Any,
    metadata_chain: Any,
    rst_settings: RstSettings | None = None,
) -> dict[str, Any]:
    """Process an RST document, extract rules and summaries, and compile artifacts."""
    relative_source = source.relative_to(input_root)

    settings = rst_settings or RstSettings()
    block_extractor = RstBlockExtractor(settings=settings)
    chunker = TextChunker(limit=settings.chunk_limit)

    blocks = block_extractor.extract(source)
    chunks = chunker.chunk(blocks)

    all_rules: list[Rule] = []
    summaries: list[str] = []
    chunk_errors: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        label = f"{source} extraction chunk {index}/{len(chunks)} ({len(chunk)} chars)"

        try:
            batch = invoke_extractor(extractor, chunk, label)
            all_rules.extend(batch.rules)

            if batch.summary.strip():
                summaries.append(batch.summary.strip())

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"ERROR processing chunk {label}: {error_text}", file=sys.stderr)
            chunk_errors.append(
                {
                    "chunk": index,
                    "error": error_text,
                }
            )

    unique_rules: dict[str, Rule] = {}
    for rule in all_rules:
        for alternative in rule.alternatives:
            single_alternative_rule = Rule(
                head=rule.head,
                alternatives=[alternative],
                evidence=rule.evidence,
            )
            clause = render_prolog([single_alternative_rule])
            unique_rules.setdefault(clause, single_alternative_rule)

    rules = list(unique_rules.values())
    prolog = render_prolog(rules)

    summary_text = "\n".join(f"- {summary}" for summary in summaries)
    if not summary_text:
        summary_text = "No prose summaries were produced."

    try:
        metadata = timed_invoke(
            f"{source} metadata",
            metadata_chain,
            {
                "source_path": relative_source.as_posix(),
                "summaries": summary_text,
            },
        )
    except Exception as exc:
        print(
            f"WARNING: metadata generation failed for {source}: {exc}",
            file=sys.stderr,
        )
        metadata = fallback_metadata(relative_source.as_posix())

    output_path = (output_root / relative_source).with_suffix(".pl")
    atomic_write_text(output_path, prolog + "\n")

    provenance_path = Path(f"{output_path}.json")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    status = "partial" if chunk_errors else "complete"
    if not chunks:
        status = "complete"

    provenance = {
        "source_path": relative_source.as_posix(),
        "source_sha256": source_hash,
        "description": metadata.description,
        "domain": metadata.domain,
        "status": status,
        "rule_count": len(rules),
        "summaries": summaries,
        "chunk_errors": chunk_errors,
        "rules": [rule.model_dump() for rule in rules],
        "prolog_path": output_path.as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(provenance_path, provenance)

    print(f"{source} -> {output_path} ({len(rules)} clauses, status={status})")

    return {
        "source_path": relative_source.as_posix(),
        "prolog_path": output_path.as_posix(),
        "provenance_path": provenance_path.as_posix(),
        "description": metadata.description,
        "domain": metadata.domain,
        "status": status,
        "rule_count": len(rules),
        "source_sha256": source_hash,
        "chunk_errors": chunk_errors,
        "generated_at": provenance["generated_at"],
    }
