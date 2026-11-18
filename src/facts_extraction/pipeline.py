from pathlib import Path

import retrico

from ..config import ExtractionSettings


def build_pipeline(json_path: Path, settings: ExtractionSettings):
    """Build a retrico pipeline that writes extracted graph data to FalkorDB and JSON."""
    model = settings.model
    entity_labels = list(settings.entity_labels)
    relation_labels = list(settings.relation_labels)

    builder = retrico.RetriCoBuilder(name=settings.builder_name)
    builder.graph_store(settings.graph_store)
    builder.chunker(method=settings.chunk_method)

    builder.ner_llm(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model,
        labels=entity_labels,
        temperature=model.temperature,
    )

    builder.relex_llm(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model,
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        temperature=model.temperature,
    )

    builder.graph_writer(json_output=str(json_path))
    return builder.build(verbose=True)
