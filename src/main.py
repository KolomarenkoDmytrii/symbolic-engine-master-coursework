from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.endpoints.execution import router as execution_router
from .api.endpoints.graph import router as graph_router
from .api.endpoints.rules import router as rules_router
from .api.dependencies import close_services, initialize_services


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    initialize_services(fastapi_app)
    try:
        yield
    finally:
        await close_services(fastapi_app)


app = FastAPI(
    title="Copter Reasoning API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(execution_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")
app.include_router(rules_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
