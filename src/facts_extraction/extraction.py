import logging
import tempfile
from pathlib import Path

from ..config import ExtractionSettings
from ..docs.rst_loading import build_source_text, find_rst_files, load_rst_text
from ..docs.text_extraction import RstTextExtractor
from .merge import merge_json_arrays
from .pipeline import build_pipeline

logger = logging.getLogger(__name__)


def extract_facts(
    docs_dir: Path,
    output_json: Path,
    settings: ExtractionSettings,
    extractor: RstTextExtractor,
) -> int:
    """Extract a knowledge graph from every .rst file under docs_dir."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(find_rst_files(docs_dir))
    total = len(files)
    # logger.info("Found %d .rst files under %s", total, docs_dir.resolve())
    print(f"Found {total} .rst files under {docs_dir.resolve()}")

    if not files:
        logger.error("No .rst files found under %s", docs_dir.resolve())
        raise SystemExit(f"No .rst files found under {docs_dir.resolve()}")

    processed_files = 0
    part_files: list[Path] = []

    with tempfile.TemporaryDirectory(
        prefix="retrico_parts_",
        dir=output_json.parent,
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for number, path in enumerate(files, start=1):
            relative_path = path.relative_to(docs_dir)
            # logger.info("[%d/%d] Parsing: %s", number, total, relative_path)
            logger.info(f"[{number}/{total}] Parsing: {relative_path}")

            try:
                text = load_rst_text(path, extractor)
            except Exception as exc:
                logger.warning(
                    "[%d/%d] Skipped: %s — %s", number, total, relative_path, exc
                )
                continue

            if not text:
                logger.info(
                    "[%d/%d] Skipped empty file: %s", number, total, relative_path
                )
                continue

            part_path = temp_dir / f"file_{number:06d}.json"
            source_text = build_source_text(path, docs_dir, text)

            # logger.info("[%d/%d] Extracting graph: %s", number, total, relative_path)
            print(f"[{number}/{total}] Extracting graph: {relative_path}")
            try:
                pipeline = build_pipeline(part_path, settings)
                pipeline.run(texts=[source_text])
            # except Exception as exc:
            #     logger.error(
            #         "[%d/%d] Failed: %s — %s", number, total, relative_path, exc
            #     )
            #     continue
            except Exception:
                logger.error(
                    "[%d/%d] Failed: %s — %s", number, total, relative_path
                )
                continue

            processed_files += 1
            if part_path.is_file():
                part_files.append(part_path)

            # logger.info(f"[{number}/{total}] Done: {relative_path}")
            print(f"[{number}/{total}] Done: {relative_path}")

        if not part_files:
            logger.error("No batch JSON files were produced; nothing to merge.")
            raise SystemExit("No batch JSON files were produced; nothing to merge.")

        # logger.info("Merging %d part files into %s", len(part_files), output_json)
        print(f"Merging {len(part_files)} part files into {output_json}")
        merge_json_arrays(part_files, output_json)

    # logger.info("Processed %d of %d files.", processed_files, total)
    # logger.info("Knowledge graph export: %s", output_json.resolve())
    print(f"Processed {processed_files} of {total} files.")
    print(f"Knowledge graph export: {output_json.resolve()}")
    return processed_files
