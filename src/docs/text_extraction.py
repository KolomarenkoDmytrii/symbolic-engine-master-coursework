from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import io
import sys

from docutils import nodes
from docutils.core import publish_doctree


@dataclass(frozen=True)
class RstSettings:
    """Configuration shared by the RST processing classes."""

    docs_root: Path
    max_files_per_batch: int = 20
    max_chars_per_batch: int = 100_000
    chunk_limit: int = 12_000
    excluded_nodes: tuple[type[nodes.Node], ...] = (nodes.comment,)


class RstFileFinder:
    """Locates .rst files beneath the docs root."""

    def __init__(self, settings: RstSettings):
        self.settings = settings

    def files(self) -> Iterator[Path]:
        yield from self.settings.docs_root.rglob("*.rst")


class RstParser:
    """Parses RST source into a docutils doctree."""

    @staticmethod
    def parse(path: Path, warnings: io.StringIO | None = None):
        source = path.read_text(encoding="utf-8")
        return publish_doctree(
            source,
            settings_overrides={
                "report_level": 5,
                "halt_level": 6,
                "warning_stream": warnings if warnings is not None else io.StringIO(),
            },
        )


class RstTextExtractor:
    """Produces plain text for each RST file as a single string."""

    def __init__(self, parser: RstParser = RstParser()):
        self.parser = parser

    def extract(self, path: Path) -> str:
        doctree = self.parser.parse(path)
        return doctree.astext().strip()


class RstBlockExtractor:
    """Extracts labelled, section-aware text blocks from RST files."""

    def __init__(self, settings: RstSettings, parser: RstParser = RstParser()):
        self.settings = settings
        self.parser = parser

    def extract(self, path: Path) -> list[str]:
        warnings = io.StringIO()
        doctree = self.parser.parse(path, warnings)

        if warnings.getvalue().strip():
            print(
                f"RST warnings in {path}:\n{warnings.getvalue()}",
                file=sys.stderr,
            )

        blocks: list[str] = []
        self._walk(doctree, [], path, blocks)
        return blocks

    def _walk(
        self,
        node: nodes.Node,
        headings: list[str],
        path: Path,
        blocks: list[str],
    ) -> None:
        if isinstance(node, self.settings.excluded_nodes):
            return

        if isinstance(node, nodes.section):
            self._walk_section(node, headings, path, blocks)
        elif isinstance(node, nodes.row):
            self._add_table_row(node, headings, path, blocks)
        elif isinstance(node, nodes.field):
            self._add_field(node, headings, path, blocks)
        elif isinstance(node, nodes.title):
            self._add_text(node.astext(), headings, path, blocks, label="Title")
        elif isinstance(node, (nodes.paragraph, nodes.term, nodes.caption)):
            self._add_text(node.astext(), headings, path, blocks)
        else:
            for child in getattr(node, "children", []):
                self._walk(child, headings, path, blocks)

    def _walk_section(self, node, headings, path, blocks) -> None:
        title_node = next(
            (c for c in node.children if isinstance(c, nodes.title)),
            None,
        )
        section_headings = headings
        if title_node is not None:
            title = title_node.astext().strip()
            if title:
                section_headings = [*headings, title]

        for child in node.children:
            if child is not title_node:
                self._walk(child, section_headings, path, blocks)

    def _add_table_row(self, node, headings, path, blocks) -> None:
        # Keep each row together so its label and value stay paired.
        cells = [
            entry.astext().strip()
            for entry in node.findall(nodes.entry)
            if entry.astext().strip()
        ]
        if cells:
            self._add_text(" | ".join(cells), headings, path, blocks, label="Table row")

    def _add_field(self, node, headings, path, blocks) -> None:
        name_node = next(
            (c for c in node.children if isinstance(c, nodes.field_name)),
            None,
        )
        body_node = next(
            (c for c in node.children if isinstance(c, nodes.field_body)),
            None,
        )
        name = name_node.astext().strip() if name_node else ""
        body = body_node.astext().strip() if body_node else ""
        self._add_text(
            f"{name}: {body}".strip(": "), headings, path, blocks, label="Field"
        )

    @staticmethod
    def _add_text(
        text: str,
        headings: list[str],
        path: Path,
        blocks: list[str],
        label: str = "Text",
    ) -> None:
        text = text.strip()
        if not text:
            return
        section = " > ".join(headings) if headings else path.stem
        blocks.append(f"Section: {section}\n{label}: {text}")


class TextChunker:
    """Groups blocks into bounded chunks without splitting blocks unnecessarily."""

    def __init__(self, limit: int = 12_000):
        if limit <= 0:
            raise ValueError("Chunk limit must be a positive integer.")
        self.limit = limit

    def chunk(self, blocks: Iterable[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for block in blocks:
            for piece in self._split_long_block(block):
                candidate = f"{current}\n\n{piece}".strip()
                if current and len(candidate) > self.limit:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate

        if current:
            chunks.append(current)
        return chunks

    def _split_long_block(self, block: str) -> list[str]:
        """Split an oversized block without dropping any characters."""
        if len(block) <= self.limit:
            return [block]
        return [block[i : i + self.limit] for i in range(0, len(block), self.limit)]


class DocumentBatcher:
    """Groups texts into batches bounded by file count and character count."""

    def __init__(self, max_files: int, max_chars: int):
        self.max_files = max_files
        self.max_chars = max_chars

    def batch(self, items: Iterable[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        batch_chars = 0

        for text in items:
            if batch and (
                len(batch) >= self.max_files or batch_chars + len(text) > self.max_chars
            ):
                yield batch
                batch = []
                batch_chars = 0

            batch.append(text)
            batch_chars += len(text)

        if batch:
            yield batch


class RstCorpus:
    """Coordinates the other classes to build documents, batches, and chunks."""

    def __init__(self, settings: RstSettings):
        self.settings = settings
        self.finder = RstFileFinder(settings)
        self.text_extractor = RstTextExtractor()
        self.block_extractor = RstBlockExtractor(settings)
        self.chunker = TextChunker(settings.chunk_limit)
        self.batcher = DocumentBatcher(
            settings.max_files_per_batch,
            settings.max_chars_per_batch,
        )

    def documents(self) -> Iterator[str]:
        """Yield one parsed document at a time, skipping files that fail."""
        for path in self.finder.files():
            try:
                text = self.text_extractor.extract(path)
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
                continue

            if text:
                relative_path = path.relative_to(self.settings.docs_root)
                yield f"Source file: {relative_path}\n\n{text}"

    def batches(self) -> Iterator[list[str]]:
        return self.batcher.batch(self.documents())

    def chunks_for_file(self, path: Path) -> list[str]:
        return self.chunker.chunk(self.block_extractor.extract(path))


if __name__ == "__main__":
    settings = RstSettings(docs_root=Path("docs"))
    corpus = RstCorpus(settings)

    for batch in corpus.batches():
        print(f"Batch with {len(batch)} document(s)")
