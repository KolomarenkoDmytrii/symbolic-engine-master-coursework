
from fastapi import APIRouter, Depends, HTTPException

from .schemas import (
    QueryRequest,
    ExecutionResponse,
)
from .services.chat_service import ChatService
from ..dependencies import get_chat_service


router = APIRouter(
    prefix="/execution",
    tags=["Execution"],
)


@router.post("/query")
async def execute_query(
    request: QueryRequest,
    service: ChatService = Depends(get_chat_service),
):
    try:
        result = await service.query(
            question=request.question,
            history=[
                (item["role"], item["content"])
                for item in request.history
            ],
        )

        return result

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal execution error.",
        ) from exc
