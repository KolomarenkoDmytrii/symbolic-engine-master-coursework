from __future__ import annotations

from fastapi import APIRouter, HTTPException
from .schemas import GraphQueryRequest
from typing import Any, Dict

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


@router.post("/query", response_model=Dict[str, Any])
def query_graph_endpoint(payload: GraphQueryRequest) -> Dict[str, Any]:
    """Виконує Cypher-запит до FalkorDB та повертає структуровані дані стану БПЛА."""
    try:
        # Симуляція результату виконання запиту до графа
        return {
            "status": "success",
            "graph_name": payload.graph_name,
            "nodes": [
                {
                    "id": "battery_1",
                    "labels": ["vehicle_component", "sensor"],
                    "properties": {"voltage": 10.2},
                }
            ],
            "relations": [
                {"head": "battery_1", "type": "monitors", "tail": "failsafe_trigger"}
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
