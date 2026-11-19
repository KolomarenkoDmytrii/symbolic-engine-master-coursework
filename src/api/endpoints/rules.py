from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_rule_service
from ..schemas import (
    RuleSelectionRequest,
    RuleSelectionResponse,
    RuleExtractionRequest,
    RuleExtractionResponse,
)
from ..services.rule_service import RuleService


router = APIRouter(
    prefix="/rules",
    tags=["Rules Management"],
)


def _artifact_summary(artifact: Any) -> dict[str, Any]:
    """Return only public metadata; do not expose server-side file paths."""
    return {
        "artifact_id": str(getattr(artifact, "artifact_id", "")),
        "domain": str(getattr(artifact, "domain", "")),
    }


async def _run_rule_extraction(
    service: RuleService,
    source_text: str,
    domain: str | None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    if hasattr(service, "extract_rules") and callable(service.extract_rules):
        raw_result = service.extract_rules(source_text, domain=domain)
        if asyncio.iscoroutine(raw_result):
            raw_result = await raw_result
        if isinstance(raw_result, tuple) and len(raw_result) == 2:
            rules_part, sigs_part = raw_result
            return list(rules_part), list(sigs_part)
        if isinstance(raw_result, dict):
            return list(raw_result.get("extracted_rules", [])), list(
                raw_result.get("signatures", [])
            )

    artifacts, signatures = service.select_rules(source_text)
    summaries = [_artifact_summary(item) for item in artifacts]
    return summaries, list(signatures)


@router.post("/select", response_model=RuleSelectionResponse)
def select_rules_endpoint(
    payload: RuleSelectionRequest,
    service: RuleService = Depends(get_rule_service),
) -> RuleSelectionResponse:
    try:
        artifacts, signatures = service.select_rules(payload.question)

        return RuleSelectionResponse(
            selected_rules=[_artifact_summary(item) for item in artifacts],
            signatures=signatures,
        )

    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Rule selection failed.",
        ) from exc


@router.post("/extract", response_model=RuleExtractionResponse)
async def extract_rules_endpoint(
    payload: RuleExtractionRequest,
    service: RuleService = Depends(get_rule_service),
) -> RuleExtractionResponse:
    print(
        f"[PROLOG_RULE_EXTRACTION] Starting Prolog rules extraction task (domain={payload.domain!r}, length={len(payload.source_text)})."
    )

    try:
        extracted_rules, signatures = await _run_rule_extraction(
            service=service,
            source_text=payload.source_text,
            domain=payload.domain,
        )

        return RuleExtractionResponse(
            extracted_rules=extracted_rules,
            signatures=signatures,
        )

    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prolog rule extraction failed.",
        ) from exc
