
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.endpoints_execution import router as execution_router
from .api.endpoints_graph import router as graph_router
from .api.endpoints_rules import router as rules_router

from .dependencies import initialize_services, close_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_services(app)
    yield
    close_services(app)


app = FastAPI(
    title="Copter Reasoning API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(
    execution_router,
    prefix="/api/v1",
)

app.include_router(
    graph_router,
    prefix="/api/v1",
)

app.include_router(
    rules_router,
    prefix="/api/v1",
)


@app.get("/health")
async def health():
    return {"status": "ok"}
