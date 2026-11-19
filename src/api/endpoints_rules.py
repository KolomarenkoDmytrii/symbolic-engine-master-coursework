from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_rule_service
from src.api.schemas import RuleSelectionRequest, RuleSelectionResponse
from src.api.services.rule_service import RuleService


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
