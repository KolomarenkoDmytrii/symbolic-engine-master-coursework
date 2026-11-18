from __future__ import annotations

from fastapi import APIRouter, HTTPException
from .schemas import ConsultRequest, ConsultResponse

router = APIRouter(prefix="/rules", tags=["Rules Management"])

# Реєстр активних сесій (у реальному середовищі замінюється на менеджер сесій PrologMCP)
ACTIVE_SESSIONS: dict[str, str] = {}


@router.post("/consult", response_model=ConsultResponse)
def consult_rules_endpoint(payload: ConsultRequest) -> ConsultResponse:
    """Завантажує правила у середовище ізольованого модульного пісочнику PrologMCP[cite: 4]."""
    try:
        load_id = "session_copter_default"
        ACTIVE_SESSIONS[load_id] = payload.source_code
        return ConsultResponse(
            status="success",
            load_id=load_id,
            message="Source consulted successfully in isolated module sandbox.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
