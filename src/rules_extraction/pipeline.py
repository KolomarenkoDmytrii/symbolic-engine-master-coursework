from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import RstSettings
from ..docs.text_extraction import RstBlockExtractor, TextChunker
from .compilation import atomic_write_json, atomic_write_text, render_prolog
from .extraction import fallback_metadata, invoke_extractor, timed_invoke
from .models import Rule


def _extract_chunks(
    source: Path,
    chunks: list[str],
    extractor: Any,
) -> tuple[list[Rule], list[str], list[dict[str, Any]]]:
    """Extract rules and summaries, recording failures per chunk."""
    all_rules: list[Rule] = []
    summaries: list[str] = []
    chunk_errors: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        label = f"{source} extraction chunk {index}/{len(chunks)} ({len(chunk)} chars)"

        try:
            batch = invoke_extractor(extractor, chunk, label)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"ERROR processing chunk {label}: {error_text}", file=sys.stderr)
            chunk_errors.append({"chunk": index, "error": error_text})
            continue

        all_rules.extend(batch.rules)
        summary = batch.summary.strip()
        if summary:
            summaries.append(summary)

    return all_rules, summaries, chunk_errors


def _deduplicate_rules(all_rules: list[Rule]) -> list[Rule]:
    """Deduplicate rules by their rendered Prolog clause."""
    unique_rules: dict[str, Rule] = {}

    for rule in all_rules:
        for alternative in rule.alternatives:
            single_rule = Rule(
                head=rule.head,
                alternatives=[alternative],
                evidence=rule.evidence,
            )
            clause = render_prolog([single_rule])
            unique_rules.setdefault(clause, single_rule)

    return list(unique_rules.values())


def _generate_metadata(
    source: Path,
    relative_path: str,
    summaries: list[str],
    metadata_chain: Any,
):
    """Generate metadata, falling back if the metadata chain fails."""
    summary_text = "\n".join(f"- {summary}" for summary in summaries)
    if not summary_text:
        summary_text = "No prose summaries were produced."

    try:
        return timed_invoke(
            f"{source} metadata",
            metadata_chain,
            {
                "source_path": relative_path,
                "summaries": summary_text,
            },
        )
    except Exception as exc:
        print(
            f"WARNING: metadata generation failed for {source}: {exc}",
            file=sys.stderr,
        )
        return fallback_metadata(relative_path)


def _build_provenance(
    *,
    relative_path: str,
    source_hash: str,
    metadata: Any,
    status: str,
    rules: list[Rule],
    summaries: list[str],
    chunk_errors: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Build the JSON provenance record."""
    return {
        "source_path": relative_path,
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
    relative_path = relative_source.as_posix()

    settings = rst_settings or RstSettings()
    blocks = RstBlockExtractor(settings=settings).extract(source)
    chunks = TextChunker(limit=settings.chunk_limit).chunk(blocks)

    all_rules, summaries, chunk_errors = _extract_chunks(
        source,
        chunks,
        extractor,
    )
    rules = _deduplicate_rules(all_rules)
    metadata = _generate_metadata(
        source,
        relative_path,
        summaries,
        metadata_chain,
    )

    output_path = (output_root / relative_source).with_suffix(".pl")
    atomic_write_text(output_path, render_prolog(rules) + "\n")

    provenance_path = Path(f"{output_path}.json")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    status = "partial" if chunk_errors and chunks else "complete"

    provenance = _build_provenance(
        relative_path=relative_path,
        source_hash=source_hash,
        metadata=metadata,
        status=status,
        rules=rules,
        summaries=summaries,
        chunk_errors=chunk_errors,
        output_path=output_path,
    )
    atomic_write_json(provenance_path, provenance)

    print(f"{source} -> {output_path} ({len(rules)} clauses, status={status})")

    return {
        "source_path": relative_path,
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
