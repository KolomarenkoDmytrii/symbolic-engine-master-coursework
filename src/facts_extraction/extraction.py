from ..config import ExtractionSettings

from ..docs.text_extraction import RstTextExtractor
from ..docs.rst_loading import build_source_text, find_rst_files, load_rst_text

from .merge import merge_json_arrays
from .pipeline import build_pipeline


def extract_facts(
    docs_dir: Path,
    output_json: Path,
    settings: ExtractionSettings,
    extractor: RstTextExtractor,
) -> int:
    """Extract a knowledge graph from every .rst file under docs_dir.

    Returns the number of files that were processed successfully.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(find_rst_files(docs_dir))
    total = len(files)

    if not files:
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
            print(f"[{number}/{total}] Parsing: {relative_path}", flush=True)

            try:
                text = load_rst_text(path, extractor)
            except Exception as exc:
                print(
                    f"[{number}/{total}] Skipped: {relative_path} — {exc}", flush=True
                )
                continue

            if not text:
                print(
                    f"[{number}/{total}] Skipped empty file: {relative_path}",
                    flush=True,
                )
                continue

            part_path = temp_dir / f"file_{number:06d}.json"
            source_text = build_source_text(path, docs_dir, text)

            print(f"[{number}/{total}] Extracting graph: {relative_path}", flush=True)
            try:
                pipeline = build_pipeline(part_path, settings)
                pipeline.run(texts=[source_text])
            except Exception as exc:
                print(f"[{number}/{total}] Failed: {relative_path} — {exc}", flush=True)
                continue

            processed_files += 1
            if part_path.is_file():
                part_files.append(part_path)

            print(f"[{number}/{total}] Done: {relative_path}", flush=True)

        if not part_files:
            raise SystemExit("No batch JSON files were produced; nothing to merge.")

        merge_json_arrays(part_files, output_json)

    print(f"Processed {processed_files} of {total} files.")
    print(f"Knowledge graph export: {output_json.resolve()}")
    return processed_files
