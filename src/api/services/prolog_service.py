from __future__ import annotations

import asyncio
import multiprocessing
import re
import tempfile
import traceback

from pathlib import Path
from typing import Any

from langchain_prolog import (
    PrologConfig,
    PrologRunnable,
)

from src.data_models import PipelineContext


MAX_RETURNED_SOLUTIONS = 10
PROLOG_TIME_LIMIT_SECONDS = 20.0


def safe_atom(value: str) -> str:

    escaped = []

    for char in value:

        codepoint = ord(char)

        if char == "\\":
            escaped.append("\\\\")

        elif char == "'":
            escaped.append("''")

        elif char == "\n":
            escaped.append("\\n")

        elif char == "\r":
            escaped.append("\\r")

        elif char == "\t":
            escaped.append("\\t")

        elif codepoint < 32 or codepoint == 127:
            escaped.append(f"\\x{codepoint:x}\\")

        else:
            escaped.append(char)

    return "'" + "".join(escaped) + "'"


def ensure_sources_have_no_module_declaration(
    source: str,
    path: Path,
) -> None:

    if re.search(
        r"(?m)^\s*:-\s*module\s*\(",
        source,
    ):
        raise ValueError(f"Module declarations are not supported: {path}")


def dynamic_declaration(
    artifacts: list[Any],
) -> str:

    indicators: set[str] = set()

    for artifact in artifacts:

        source = artifact.prolog_path.read_text(encoding="utf-8")

        ensure_sources_have_no_module_declaration(
            source,
            artifact.prolog_path,
        )

        # Declare rule heads dynamically.
        source = re.sub(
            r"%[^\n]*|/\*.*?\*/",
            "",
            source,
            flags=re.DOTALL,
        )

        for clause in re.findall(
            r"(?s)(.*?\.)",
            source,
        ):

            head = clause.split(":-", 1)[0].strip()

            match = re.match(
                r"([a-z][A-Za-z0-9_]*)\s*(?:\((.*)\))?$",
                head,
                flags=re.DOTALL,
            )

            if not match:
                continue

            predicate = match.group(1)
            arguments = match.group(2)

            if arguments is None:
                arity = 0
            else:
                # Count top-level commas rather than all commas.
                arity = 1
                depth = 0
                quoted = False

                for char in arguments:
                    if char == "'":
                        quoted = not quoted
                    elif not quoted:
                        if char in "([{":
                            depth += 1
                        elif char in ")]}":
                            depth -= 1
                        elif char == "," and depth == 0:
                            arity += 1

                if not arguments.strip():
                    arity = 0

            indicators.add(f"{predicate}/{arity}")

    if not indicators:
        return ""

    return ":- dynamic " + ",\n           ".join(sorted(indicators)) + "."


def write_combined_program(
    context: PipelineContext,
    directory: Path,
) -> Path:

    program_path = directory / "query_program.pl"

    lines = []

    declaration = dynamic_declaration(context.rule_files)

    if declaration:
        lines.append(declaration)

    for artifact in context.rule_files:

        source = artifact.prolog_path.read_text(encoding="utf-8")

        ensure_sources_have_no_module_declaration(
            source,
            artifact.prolog_path,
        )

        lines.append(f":- include({safe_atom(str(artifact.prolog_path))}).")

    if context.graph_facts.strip():
        lines.append(context.graph_facts)

    program_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return program_path


def _prolog_worker(
    rules_file: str,
    goal: str,
    connection: Any,
) -> None:

    try:

        config = PrologConfig(
            rules_file=rules_file,
        )

        runnable = PrologRunnable(
            prolog_config=config,
        )

        result = runnable.invoke(goal)

        if isinstance(result, list):

            truncated = len(result) > MAX_RETURNED_SOLUTIONS

            result = result[:MAX_RETURNED_SOLUTIONS]

            payload = {
                "status": "success",
                "result": result,
                "truncated": truncated,
            }

        elif result is False:

            payload = {
                "status": "failure",
                "result": False,
                "truncated": False,
            }

        else:

            payload = {
                "status": "success",
                "result": result,
                "truncated": False,
            }

        connection.send(payload)

    except BaseException as exc:

        connection.send(
            {
                "status": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )

    finally:
        connection.close()


def run_prolog_isolated(
    rules_file: Path,
    goal: str,
    timeout_seconds: float,
) -> dict[str, Any]:

    context = multiprocessing.get_context("spawn")

    receive_connection, send_connection = context.Pipe(duplex=False)

    process = context.Process(
        target=_prolog_worker,
        args=(
            str(rules_file),
            goal,
            send_connection,
        ),
        daemon=True,
    )

    process.start()
    send_connection.close()

    try:

        if not receive_connection.poll(timeout_seconds):

            process.terminate()
            process.join(timeout=2)

            if process.is_alive():
                process.kill()
                process.join()

            return {
                "status": "timeout",
                "timeout_seconds": timeout_seconds,
                "error": "Prolog execution timed out.",
            }

        try:
            result = receive_connection.recv()

        except EOFError:
            result = {
                "status": "exception",
                "error_type": "WorkerExit",
                "error": (f"Worker exited with code " f"{process.exitcode}."),
            }

        process.join(timeout=2)

        if process.is_alive():
            process.terminate()
            process.join()

        return result

    finally:

        receive_connection.close()

        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


class PrologService:

    def __init__(
        self,
        timeout_seconds: float = PROLOG_TIME_LIMIT_SECONDS,
    ):
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        context: PipelineContext,
        goal: str,
    ) -> dict[str, Any]:

        with tempfile.TemporaryDirectory(prefix="copter-prolog-") as temp_dir:

            rules_file = write_combined_program(
                context,
                Path(temp_dir),
            )

            return await asyncio.to_thread(
                run_prolog_isolated,
                rules_file,
                goal,
                self.timeout_seconds,
            )
