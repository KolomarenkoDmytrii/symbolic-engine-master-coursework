from pathlib import Path
from typing import Iterator

from .text_extraction import RstTextExtractor


def find_rst_files(docs_dir: Path) -> Iterator[Path]:
    """Yield .rst paths under docs_dir without collecting the full list."""
    yield from docs_dir.rglob("*.rst")


def load_rst_text(path: Path, extractor: RstTextExtractor) -> str:
    """Extract plain text from one RST file using the given extractor."""
    return extractor.extract(path)


def build_source_text(path: Path, docs_dir: Path, text: str) -> str:
    """Prefix extracted text with its path relative to docs_dir."""
    relative_path = path.relative_to(docs_dir)
    return f"Source file: {relative_path}\n\n{text}"
