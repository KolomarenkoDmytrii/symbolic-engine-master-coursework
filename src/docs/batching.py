from pathlib import Path
from typing import Iterable, Iterator

from ..config import BatchLimits
from .rst_loading import build_source_text, find_rst_files, load_rst_text
from .text_extraction import RstTextExtractor


def iter_documents(docs_dir: Path, extractor: RstTextExtractor) -> Iterator[str]:
    """Yield one extracted document at a time, skipping files that fail or are empty."""
    for path in find_rst_files(docs_dir):
        try:
            text = load_rst_text(path, extractor)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        if text:
            yield build_source_text(path, docs_dir, text)


def batch_texts(items: Iterable[str], limits: BatchLimits) -> Iterator[list[str]]:
    """Yield bounded lists of texts; a single oversized item forms its own batch."""
    batch: list[str] = []
    batch_chars = 0

    for text in items:
        if batch and (
            len(batch) >= limits.max_files
            or batch_chars + len(text) > limits.max_chars
        ):
            yield batch
            batch = []
            batch_chars = 0

        batch.append(text)
        batch_chars += len(text)

    if batch:
        yield batch
