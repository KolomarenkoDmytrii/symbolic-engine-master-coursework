from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

# from .. import config as app_config
from ..dependencies import get_graph_service
from ..schemas import (
    GraphExtractionRequest,
    GraphExtractionResponse,
    GraphResponse,
    GraphSearchRequest,
)
from ..services.graph_service import GraphService
from ...config import DOCS_DIR, extraction_settings
from ...facts_extraction.extraction import extract_facts
from ...docs.text_extraction import RstTextExtractor

router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)


# def _resolve_docs_dir() -> Path:
#     """Resolve DOCS_DIR from src.config with a safe fallback to ./docs."""
#     configured_dir = getattr(app_config, "DOCS_DIR", None)
#     if configured_dir is None:
#         rst_settings = getattr(app_config, "rst_settings", None)
#         configured_dir = getattr(rst_settings, "docs_dir", Path("./docs"))
#     return Path(configured_dir)


async def _call_maybe_async(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a callable that may return either a synchronous value or an awaitable."""
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _run_graph_extraction(
    service: GraphService,
    docs_dir: Path,
) -> dict[str, Any]:
    """Trigger knowledge graph extraction from DOCS_DIR via GraphService."""
    # if hasattr(service, "extract") and callable(service.extract):
    #     params = inspect.signature(service.extract).parameters
    #     if "docs_dir" in params:
    #         return await _call_maybe_async(service.extract, docs_dir=docs_dir)
    #     return await _call_maybe_async(service.extract)

    # if hasattr(service, "get_state") and callable(service.get_state):
    #     return await _call_maybe_async(service.get_state)

    # return await service.search("")
    return await _call_maybe_async(
        extract_facts,
        docs_dir=docs_dir,
        output_json=Path("./output/facts.json"),
        settings=extraction_settings,
        extractor=RstTextExtractor(),
    )


@router.post("/query", response_model=GraphResponse)
async def query_graph_endpoint(
    request: GraphSearchRequest,
    service: GraphService = Depends(get_graph_service),
) -> GraphResponse:
    try:
        # 1. Perform the search
        raw_result = await service.search(request.query)

        # 2. Standardize the output to match GraphResponse schema
        # This is crucial to avoid the 500 error caused by FastAPI validation
        standardized_result = service.serialize(raw_result)

        return standardized_result

    except ValueError as exc:
        # This handles logic errors (e.g. malformed query) as 400/422
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        # This ensures we log the actual error before sending a generic 500
        import traceback

        print(f"CRITICAL ERROR in /graph/query: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred during graph query."
        )


@router.post("/extract", response_model=GraphExtractionResponse)
async def extract_graph_endpoint(
    service: GraphService = Depends(get_graph_service),
) -> GraphExtractionResponse:
    # docs_dir = _resolve_docs_dir()
    docs_dir = DOCS_DIR
    print(
        f"[GRAPH_EXTRACTION] Starting knowledge graph extraction task from DOCS_DIR={docs_dir}.",
        flush=True,
    )

    try:
        processed_files = await _run_graph_extraction(
            service=service,
            docs_dir=docs_dir,
        )
        # atoms = sorted(service.get_atoms(graph_data))

        # return GraphExtractionResponse.model_validate(
        #     {
        #         "nodes": graph_data.get("nodes", []),
        #         "relations": graph_data.get("relations", []),
        #         "atoms": atoms,
        #     }
        # )
        return {"processed_files": processed_files}

    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Knowledge graph extraction failed.",
        ) from exc
