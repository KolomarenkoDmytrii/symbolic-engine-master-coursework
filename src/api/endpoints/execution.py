from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_chat_service
from ..schemas import ExecutionResponse, QueryRequest
from ..services.chat_service import ChatService


router = APIRouter(
    prefix="/execution",
    tags=["Execution"],
)


@router.post("/query", response_model=ExecutionResponse)
async def execute_query(
    request: QueryRequest,
    service: ChatService = Depends(get_chat_service),
) -> ExecutionResponse:
    try:
        result = await service.query(
            question=request.question,
            history=[(item.role, item.content) for item in request.history],
        )
        return ExecutionResponse.model_validate(result)

    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        # Keep implementation details and tracebacks out of the HTTP response.
        raise HTTPException(
            status_code=500,
            detail="Internal execution error.",
        ) from exc
