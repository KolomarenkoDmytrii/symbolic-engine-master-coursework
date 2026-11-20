import traceback
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_chat_service
from ..schemas import ExecutionResponse, QueryRequest
from ..services.chat_service import ChatService

logger = logging.getLogger(__name__)

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

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except RuntimeError as exc:
        # Log the error and the traceback so you can see WHY it failed in the console
        logger.error(f"RuntimeError in execution/query: {str(exc)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        # For any other unexpected error
        logger.error(f"Unexpected error in execution/query: {str(exc)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="An unexpected internal error occurred."
        )
