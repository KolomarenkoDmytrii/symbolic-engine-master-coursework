from dataclasses import dataclass
from typing import Sequence, TypedDict, Callable, Any

import importlib

import os
import json
from pathlib import Path

import retrico
from docutils import nodes
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .repositories.falkordb import FalkorDBGraphRepository

load_dotenv()


DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "google/gemma-3-27b-it:deepinfra"

DOCS_DIR = Path("./documents")

@dataclass(frozen=True)
class RstSettings:
    """Configuration shared by the RST processing classes."""

    # docs_root: Path
    max_files_per_batch: int = 20
    max_chars_per_batch: int = 100_000
    chunk_limit: int = 12_000
    excluded_nodes: tuple[type[nodes.Node], ...] = (
        nodes.literal_block,
        nodes.doctest_block,
        nodes.raw,
        nodes.comment,
        nodes.system_message,
    )


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.1


@dataclass(frozen=True)
class BatchLimits:
    max_files: int = 16
    max_chars: int = 500_000


class Labels(TypedDict):
    entity_labels: Sequence[str]
    relation_labels: Sequence[str]


@dataclass(frozen=True)
class ExtractionSettings:
    model: ModelSettings
    builder_name: str
    entity_labels: Sequence[str]
    relation_labels: Sequence[str]
    graph_store: retrico.FalkorDBLiteConfig
    limits: BatchLimits = BatchLimits()
    chunk_method: str = "paragraph"


_labels: Labels = {"entity_labels": [], "relation_labels": []}
with open(
    Path(os.environ["CONFIG_DIR"]) / "labels.json", encoding="utf-8"
) as labels_file:
    _labels = json.load(labels_file)

extraction_settings = ExtractionSettings(
    model=ModelSettings(
        base_url=os.getenv("OPENAI_API_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    ),
    builder_name=os.getenv("RETRICO_BUILDER_NAME", DEFAULT_MODEL),
    entity_labels=_labels["entity_labels"],
    relation_labels=_labels["relation_labels"],
    graph_store=retrico.FalkorDBLiteConfig(),
)


# def load_labels() -> Labels:
#     if not _labels:
#         load_dotenv()
#         with open(
#             Path(os.environ["CONFIG_DIR"]) / "labels.json", encoding="utf-8"
#         ) as labels_file:
#             _labels = json.load(labels_file)
#             return _labels
#     else:
#         return _labels


# def load_settings() -> ExtractionSettings:
#     """Read environment variables and build the settings object.

#     Called once by the entry point, so importing this module has no side effects.
#     """
#     load_dotenv()
#     labels = load_labels()

#     return ExtractionSettings(
#         model=ModelSettings(
#             base_url=DEFAULT_BASE_URL,
#             api_key=os.environ["HF_TOKEN"],
#             model=DEFAULT_MODEL,
#         ),
#         builder_name=os.environ["RETRICO_BUILDER_NAME"],
#         entity_labels=labels["entity_labels"],
#         relation_labels=labels["relation_labels"],
#         graph_store=retrico.FalkorDBLiteConfig(),
#     )


def _load_factory(spec: str) -> Callable[[], Any]:
    """Load a zero-argument factory from 'package.module:callable'."""
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Factory must be specified as 'package.module:callable'.")

    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)

    if not callable(factory):
        raise TypeError(f"Configured factory is not callable: {spec}")

    return factory


@dataclass(frozen=True)
class APISettings:
    rules_dir: Path
    index_json: Path
    model_name: str
    base_url: str | None = None
    temperature: float = 0.0

    @property
    def rulers_directory(self) -> Path:
        # Retains compatibility with the dependency code already shown.
        return self.rules_dir

    @property
    def rules_index(self) -> Path:
        return self.index_json

    def create_llm(self) -> ChatOpenAI:
        api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set HF_TOKEN or OPENAI_API_KEY to configure the API model."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return ChatOpenAI(**kwargs)

    def create_graph_repository(self) -> Any:
        # spec = os.getenv("GRAPH_REPOSITORY_FACTORY")
        # if not spec:
        #     raise RuntimeError(
        #         "Set GRAPH_REPOSITORY_FACTORY to "
        #         "'package.module:factory' for the graph repository."
        #     )

        # factory = _load_factory(spec)
        # return factory()
        return FalkorDBGraphRepository(extraction_settings.graph_store)


api_settings = APISettings(
    rules_dir=Path(os.getenv("rulers_directory", "data/rules")),
    index_json=Path(os.getenv("RULE_INDEX_JSON", "data/rules/index.json")),
    model_name=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    base_url=os.getenv("OPENAI_API_BASE_URL", DEFAULT_BASE_URL),
)
