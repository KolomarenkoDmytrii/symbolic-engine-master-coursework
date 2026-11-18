from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import register_api_routers

app = FastAPI(
    title="ArduPilot Neuro-Symbolic Expert System API",
    description=(
        "High-performance asynchronous backend and API gateway for the ArduPilot "
        "neuro-symbolic expert system, integrating FalkorDB/Neo4j graph retrieval, "
        "SWI-Prolog symbolic reasoning, and LLM agent orchestration."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all modular API endpoints for MCP tools and symbolic inference
register_api_routers(app)


@app.get("/healthz", tags=["Health Management"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for container liveness and readiness probes.

    Supports Podman/Quadlet health check orchestration and ensures service availability.
    """
    return {
        "status": "healthy",
        "service": "ardupilot-expert-api-gateway",
    }


if __name__ == "__main__":
    uvicorn.main(["main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
