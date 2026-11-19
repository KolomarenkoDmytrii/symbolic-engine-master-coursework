from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from ...config import ExtractionSettings

# from ...config import api_settings


# COPTER_FAILSAFE_ENTITY_LABELS = {
#     "failsafe_mechanism",
#     "failsafe_trigger",
#     "failsafe_action",
#     "flight_mode",
#     "parameter",
#     "vehicle_component",
#     "sensor",
#     "estimator",
#     "communication_link",
#     "mission_command",
#     "configuration_value",
#     "threshold",
#     "timeout",
#     "vehicle_state",
# }

# COPTER_FAILSAFE_RELATION_LABELS = {
#     "triggered_by",
#     "monitors",
#     "configured_by",
#     "takes_action",
#     "switches_to",
#     "falls_back_to",
#     "bypassed_by",
#     "applies_to",
#     "requires",
#     "remains_in",
#     "clears_when",
#     "modifies",
# }


def to_plain_data(value: Any) -> Any:
    result: Any = str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        result = value

    if isinstance(value, dict):
        result = {str(key): to_plain_data(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        result = [to_plain_data(item) for item in value]

    if hasattr(value, "model_dump"):
        result = to_plain_data(value.model_dump())

    if hasattr(value, "dict") and callable(value.dict):
        result = to_plain_data(value.dict())

    if hasattr(value, "__dict__"):
        result = to_plain_data(vars(value))

    return result


def first_present(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def node_key(node: dict[str, Any]) -> str:
    properties = node.get("properties")

    if not isinstance(properties, dict):
        properties = {}

    for source in (node, properties):
        for key in (
            "id",
            "element_id",
            "node_id",
            "text",
            "name",
        ):
            value = source.get(key)

            if value is not None and not isinstance(value, (dict, list)):
                return str(value)

    raise ValueError(f"Graph node has no usable identifier: {node!r}")


def node_labels(node: dict[str, Any]) -> set[str]:
    properties = node.get("properties")

    if not isinstance(properties, dict):
        properties = {}

    values = []

    labels = node.get("labels")

    if isinstance(labels, list):
        values.extend(labels)
    elif isinstance(labels, str):
        values.append(labels)

    for source in (node, properties):
        for key in ("label", "type"):
            if source.get(key) is not None:
                values.append(source[key])

    return {str(value).strip().casefold() for value in values if value is not None}


def normalize_endpoint(
    endpoint: Any,
    aliases: dict[str, str],
) -> str | None:

    if isinstance(endpoint, dict):
        endpoint = first_present(
            endpoint,
            (
                "id",
                "element_id",
                "node_id",
                "text",
                "name",
            ),
        )

    if endpoint is None:
        return None

    value = str(endpoint)
    return aliases.get(value, value)


def normalize_graph_result(raw: Any) -> dict[str, Any]:
    data = to_plain_data(raw)

    if isinstance(data, list):
        data = {
            "nodes": data,
            "relations": [],
        }

    if not isinstance(data, dict):
        raise TypeError(f"Unexpected graph result: {type(raw).__name__}")

    for key in ("data", "result", "subgraph"):
        nested = data.get(key)

        if isinstance(nested, dict) and any(
            item in nested
            for item in (
                "nodes",
                "entities",
                "relations",
                "edges",
            )
        ):
            data = nested
            break

    nodes = data.get("nodes") or data.get("entities") or []

    relations = (
        data.get("relations") or data.get("edges") or data.get("relationships") or []
    )

    if not isinstance(nodes, list):
        raise TypeError("Graph nodes must be a list.")

    if not isinstance(relations, list):
        raise TypeError("Graph relations must be a list.")

    return {
        "nodes": nodes,
        "relations": relations,
    }


def filter_graph_to_copter_schema(
    graph_result: dict[str, Any],
    entity_labels: Sequence[str],
    relation_labels: Sequence[str],
) -> dict[str, Any]:

    nodes = []
    aliases: dict[str, str] = {}
    allowed_keys: set[str] = set()

    for raw_node in graph_result["nodes"]:

        node = raw_node if isinstance(raw_node, dict) else to_plain_data(raw_node)

        if not isinstance(node, dict):
            continue

        if not node_labels(node) & set(entity_labels):
            continue

        key = node_key(node)

        allowed_keys.add(key)
        nodes.append(node)

        properties = node.get("properties")

        if not isinstance(properties, dict):
            properties = {}

        for source in (node, properties):
            for alias_key in (
                "id",
                "element_id",
                "node_id",
                "text",
                "name",
            ):
                value = source.get(alias_key)

                if value is None or isinstance(value, (dict, list)):
                    continue

                alias = str(value)
                previous = aliases.get(alias)

                if previous is not None and previous != key:
                    raise ValueError(f"Ambiguous graph alias: {alias}")

                aliases[alias] = key

    relations = []

    for raw_relation in graph_result["relations"]:

        relation = (
            raw_relation
            if isinstance(raw_relation, dict)
            else to_plain_data(raw_relation)
        )

        if not isinstance(relation, dict):
            continue

        relation_type = first_present(
            relation,
            ("type", "label", "relation", "predicate"),
        )

        if relation_type is None:
            continue

        if str(relation_type).strip().casefold() not in relation_labels:
            continue

        head = first_present(
            relation,
            ("head", "source", "from", "start"),
        )

        tail = first_present(
            relation,
            ("tail", "target", "to", "end"),
        )

        head_key = normalize_endpoint(head, aliases)
        tail_key = normalize_endpoint(tail, aliases)

        if head_key in allowed_keys and tail_key in allowed_keys:
            relations.append(relation)

    return {
        "nodes": nodes,
        "relations": relations,
    }


def graph_atom_values(
    graph_result: dict[str, Any],
) -> set[str]:

    values: set[str] = set()

    for node in graph_result["nodes"]:

        try:
            values.add(node_key(node))
        except ValueError:
            pass

        properties = node.get("properties", {})

        if isinstance(properties, dict):
            for value in properties.values():
                if isinstance(value, str):
                    values.add(value)

        values.update(str(label) for label in node.get("labels", []))

    for relation in graph_result["relations"]:
        for key in (
            "type",
            "label",
            "relation",
            "predicate",
        ):
            value = relation.get(key)

            if value is not None:
                values.add(str(value))

    return values


class GraphService:

    def __init__(self, repository: Any, settings: ExtractionSettings):
        self.repository = repository
        self.entity_labels = settings.entity_labels
        self.relation_labels = settings.relation_labels

    async def search(
        self,
        query: str,
    ) -> dict[str, Any]:

        raw = await asyncio.to_thread(
            self.repository.search,
            query,
        )

        normalized = normalize_graph_result(raw)

        return filter_graph_to_copter_schema(normalized, self.entity_labels, self.relation_labels)

    async def get_state(self) -> dict[str, Any]:
        raw = await asyncio.to_thread(self.repository.get_state)
        normalized = normalize_graph_result(raw)
        return filter_graph_to_copter_schema(normalized, self.entity_labels, self.relation_labels)

    def serialize(
        self,
        graph_result: dict[str, Any],
    ) -> str:
        return json.dumps(
            to_plain_data(graph_result),
            ensure_ascii=False,
            default=str,
        )

    def get_atoms(
        self,
        graph_result: dict[str, Any],
    ) -> set[str]:
        return graph_atom_values(graph_result)
