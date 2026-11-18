from __future__ import annotations

from fastapi import FastAPI
from .endpoints_rules import router as rules_router
from .endpoints_graph import router as graph_router
from .endpoints_execution import router as execution_router


def register_api_routers(app: FastAPI) -> None:
    """Реєструє всі підмодулі REST API ендпоінтів для створення MCP-інструментів."""
    app.include_router(rules_router)
    app.include_router(graph_router)
    app.include_router(execution_router)
