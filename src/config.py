from dataclasses import dataclass
from typing import Sequence, TypedDict

import os
import json
from pathlib import Path

import retrico
from docutils import nodes
from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "google/gemma-3-27b-it:deepinfra"


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


@dataclass(frozen=True)
class ExtractionSettings:
    model: ModelSettings
    builder_name: str
    entity_labels: Sequence[str]
    relation_labels: Sequence[str]
    graph_store: retrico.FalkorDBLiteConfig
    limits: BatchLimits = BatchLimits()
    chunk_method: str = "paragraph"

class Labels(TypedDict):
    entity_labels: Sequence[str]
    relation_labels: Sequence[str]


def load_labels() -> Labels:
    load_dotenv()

    with open(Path(os.environ["CONFIG_DIR"]) / "labels.json") as labels_file:
        return json.load(labels_file)


def load_settings() -> ExtractionSettings:
    """Read environment variables and build the settings object.

    Called once by the entry point, so importing this module has no side effects.
    """
    load_dotenv()
    labels = load_labels()

    return ExtractionSettings(
        model=ModelSettings(
            base_url=DEFAULT_BASE_URL,
            api_key=os.environ["HF_TOKEN"],
            model=DEFAULT_MODEL,
        ),
        builder_name=os.environ["RETRICO_BUILDER_NAME"],
        entity_labels=labels["entity_labels"],
        relation_labels=labels["relation_labels"],
        graph_store=retrico.FalkorDBLiteConfig(),
    )
