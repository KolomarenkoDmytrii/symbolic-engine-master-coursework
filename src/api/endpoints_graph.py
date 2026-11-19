from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_graph_service
from .schemas import GraphResponse, GraphSearchRequest
from .services.graph_service import GraphService


router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)


@router.post("/query", response_model=GraphResponse)
async def query_graph_endpoint(
    payload: GraphSearchRequest,
    service: GraphService = Depends(get_graph_service),
) -> GraphResponse:
    try:
        result = await service.search(payload.query)
        return GraphResponse.model_validate(result)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Graph query failed.",
        ) from exc
