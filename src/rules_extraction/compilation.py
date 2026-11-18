from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import Goal, Rule


def render_goal(goal: Goal) -> str:
    """Render a canonical goal into its Prolog representation string."""
    if not goal.args:
        return goal.predicate
    return f"{goal.predicate}({', '.join(goal.args)})"


def render_prolog(rules: list[Rule]) -> str:
    """Render validated rules into Prolog clauses, deduplicating exact matches."""
    clauses: list[str] = []

    for rule in rules:
        head = render_goal(rule.head)
        for conjunction in rule.alternatives:
            body = ", ".join(render_goal(goal) for goal in conjunction)
            clauses.append(f"{head} :- {body}.")

    return "\n".join(dict.fromkeys(clauses))


def atomic_write_text(path: Path, content: str) -> None:
    """Write text content atomically via a temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, data: Any) -> None:
    """Serialize data to JSON and write atomically."""
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(path, content)
