from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_rule_service
from ..schemas import (
    RuleExtractionRequest,
    RuleExtractionResponse,
    RuleSelectionRequest,
    RuleSelectionResponse,
)
from ..services.rule_service import RuleService
from ...config import DOCS_DIR

router = APIRouter(
    prefix="/rules",
    tags=["Rules Management"],
)


async def _run_rule_extraction(
    service: RuleService,
    docs_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Simplified: Delegation of responsibility to the Service layer.
    The Service handles the orchestration, loop, and standardization.
    """
    return await service.run_extraction_pipeline(docs_dir)


@router.post("/select", response_model=RuleSelectionResponse)
def select_rules_endpoint(
    payload: RuleSelectionRequest,
    service: RuleService = Depends(get_rule_service),
) -> RuleSelectionResponse:
    try:
        # The service now handles internal errors gracefully (skipping bad rules)
        artifacts, signatures = service.select_rules(payload.question)

        return RuleSelectionResponse(
            selected_rules=[
                {"artifact_id": a.artifact_id, "domain": a.domain} for a in artifacts
            ],
            signatures=signatures,
        )

    except LookupError as exc:
        # This is a legitimate "no results" case
        raise HTTPException(status_code=404, detail=str(exc))

    except FileNotFoundError as exc:
        # This happens if the index itself is missing
        raise HTTPException(status_code=404, detail=f"Resource not found: {str(exc)}")

    except ValueError as exc:
        # This happens if the input or JSON format is invalid
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        # Catch-all for unexpected crashes to provide a clean 500
        import traceback

        print(f"Unhandled exception in select_rules_endpoint: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during rule selection.",
        )


@router.post("/extract", response_model=RuleExtractionResponse)
async def extract_rules_endpoint(
    service: RuleService = Depends(get_rule_service),
) -> RuleExtractionResponse:
    docs_dir = DOCS_DIR
    print(
        f"[PROLOG_RULE_EXTRACTION] Starting Prolog rules extraction task from DOCS_DIR={docs_dir}.",
        flush=True,
    )

    try:
        # The service handles the heavy lifting of iterating through files and building chains
        extracted_rules, signatures = await _run_rule_extraction(
            service=service,
            docs_dir=docs_dir,
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
            detail=f"Prolog rule extraction failed: {str(exc)}",
        ) from exc
