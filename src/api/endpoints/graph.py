from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_graph_service
from ..schemas import (
    GraphResponse,
    GraphSearchRequest,
    GraphExtractionRequest,
    GraphExtractionResponse,
)
from ..services.graph_service import GraphService


router = APIRouter(
    prefix="/graph-extraction",
    tags=["Knowledge Graph"],
)


async def _run_graph_extraction(
    service: GraphService,
    source_text: str,
) -> dict[str, Any]:
    if hasattr(service, "extract") and callable(service.extract):
        result = service.extract(source_text)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return await service.search(source_text)


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


@router.post("/extract", response_model=GraphExtractionResponse)
async def extract_graph_endpoint(
    payload: GraphExtractionRequest,
    service: GraphService = Depends(get_graph_service),
) -> GraphExtractionResponse:
    print(
        f"[GRAPH_EXTRACTION] Starting knowledge graph extraction task for input length={len(payload.source_text)}."
    )

    try:
        graph_data = await _run_graph_extraction(service, payload.source_text)
        atoms = sorted(service.get_atoms(graph_data))

        return GraphExtractionResponse(
            nodes=graph_data.get("nodes", []),
            relations=graph_data.get("relations", []),
            atoms=atoms,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Knowledge graph extraction failed.",
        ) from exc
