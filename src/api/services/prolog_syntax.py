from __future__ import annotations

import re
from typing import TypeAlias

Term: TypeAlias = tuple[str, str]
Goal: TypeAlias = tuple[str, list[Term]]


BARE_ATOM = re.compile(r"^[a-z][A-Za-z0-9_]*$")
VARIABLE = re.compile(r"^(?:[A-Z][A-Za-z0-9_]*|_[A-Za-z0-9_]*)$")
NUMBER = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")
QUOTED_ATOM = re.compile(r"^'(?:''|\\.|[^'\\])*'$")


def _strip_comments(text: str) -> str:
    """Remove Prolog comments, preserving quoted atoms and line breaks."""
    out: list[str] = []
    i = 0
    quoted = False

    while i < len(text):
        char = text[i]

        if quoted:
            out.append(char)

            if char == "\\" and i + 1 < len(text):
                i += 1
                out.append(text[i])
            elif char == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 1
                    out.append(text[i])
                else:
                    quoted = False

            i += 1
            continue

        if char == "'":
            quoted = True
            out.append(char)
            i += 1
            continue

        if char == "%":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("Unterminated Prolog block comment.")
            out.extend("\n" for c in text[i : end + 2] if c == "\n")
            i = end + 2
            continue

        out.append(char)
        i += 1

    if quoted:
        raise ValueError("Unterminated quoted Prolog atom.")

    return "".join(out)


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on a separator outside quoted atoms and parentheses."""
    parts: list[str] = []
    start = 0
    depth = 0
    quoted = False
    i = 0

    while i < len(text):
        char = text[i]

        if quoted:
            if char == "\\":
                i += 2
                continue
            if char == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                quoted = False
            i += 1
            continue

        if char == "'":
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses in Prolog text.")
        elif char == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1

        i += 1

    if quoted or depth != 0:
        raise ValueError("Unbalanced quoted atom or parentheses in Prolog text.")

    parts.append(text[start:].strip())
    return parts


def _quoted_atom_end(text: str, start: int) -> int:
    """Return the index after a quoted atom starting at start."""
    i = start + 1

    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == "'":
            if i + 1 < len(text) and text[i + 1] == "'":
                i += 2
            else:
                return i + 1
        else:
            i += 1

    raise ValueError("Unterminated quoted Prolog atom.")


def split_clauses(text: str) -> list[str]:
    """Split clauses on periods outside quoted atoms and parentheses."""
    clean = _strip_comments(text)
    clauses: list[str] = []
    start = 0
    depth = 0
    quoted = False
    i = 0

    while i < len(clean):
        char = clean[i]

        if char == "'":
            i = _quoted_atom_end(clean, i)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses in Prolog text.")
        elif char == "." and depth == 0:
            clause = clean[start:i].strip()
            if clause:
                clauses.append(clause)
            start = i + 1

        i += 1

    if quoted or depth != 0:
        raise ValueError("Unbalanced quoted atom or parentheses in Prolog text.")

    if clean[start:].strip():
        raise ValueError("Prolog clause is missing a final period.")

    return clauses


def parse_term(token: str, *, allow_variables: bool) -> Term:
    token = token.strip()

    if BARE_ATOM.fullmatch(token):
        return ("atom", token)

    if QUOTED_ATOM.fullmatch(token):
        # Decode the quoting form used by safe_atom(): doubled apostrophes
        # represent a literal apostrophe. Retain other escapes as text.
        value = token[1:-1].replace("''", "'")
        return ("atom", value)

    if NUMBER.fullmatch(token):
        return ("number", token)

    if allow_variables and VARIABLE.fullmatch(token):
        return ("variable", token)

    raise ValueError(f"Unsupported Prolog term: {token!r}")


def parse_goal(text: str, *, allow_variables: bool) -> Goal:
    text = text.strip()
    match = re.fullmatch(
        r"([a-z][A-Za-z0-9_]*)(?:\s*\((.*)\))?",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Unsupported Prolog goal: {text!r}")

    predicate, raw_args = match.groups()
    if raw_args is None:
        return predicate, []

    if not raw_args.strip():
        raise ValueError("Empty argument lists are not supported.")

    args = [
        parse_term(part, allow_variables=allow_variables)
        for part in _split_top_level(raw_args, ",")
    ]
    return predicate, args


def parse_fact_text(text: str) -> list[Goal]:
    """Parse ground facts only; reject rules and directives."""
    facts: list[Goal] = []

    for clause in split_clauses(text):
        if ":-" in clause or clause.startswith("?-"):
            raise ValueError("Only facts are allowed.")

        predicate, args = parse_goal(clause, allow_variables=False)
        facts.append((predicate, args))

    return facts


def extract_rule_body_signatures(text: str) -> set[tuple[str, int]]:
    """Extract body predicate/arity pairs from simple compiled Horn clauses."""
    signatures: set[tuple[str, int]] = set()

    for clause in split_clauses(text):
        if ":-" not in clause:
            continue

        head, body = clause.split(":-", 1)
        # Validate the head too; this rejects directives and unsupported syntax.
        parse_goal(head.strip(), allow_variables=True)

        for goal_text in _split_top_level(body, ","):
            predicate, args = parse_goal(
                goal_text,
                allow_variables=True,
            )
            signatures.add((predicate, len(args)))

    return signatures


def extract_rule_head_signatures(text: str) -> set[tuple[str, int]]:
    signatures: set[tuple[str, int]] = set()

    for clause in split_clauses(text):
        if ":-" not in clause:
            continue

        head_text, _body_text = clause.split(":-", 1)
        predicate, args = parse_goal(
            head_text.strip(),
            allow_variables=True,
        )
        signatures.add((predicate, len(args)))

    return signatures


def extract_prolog_atoms(text: str) -> set[str]:
    """Extract all unique predicate names (atoms) from a Prolog text."""
    atoms: set[str] = set()
    try:
        for clause in split_clauses(text):
            if ":-" in clause:
                # It's a rule: head :- body.
                head_text, body_text = clause.split(":-", 1)
                # Head
                p_head, _ = parse_goal(head_text.strip(), allow_variables=True)
                atoms.add(p_head)
                # Body (could be multiple goals)
                for goal_text in _split_top_level(body_text, ","):
                    p_body, _ = parse_goal(goal_text.strip(), allow_variables=True)
                    atoms.add(p_body)
            else:
                # It's a fact
                p, _ = parse_goal(clause, allow_variables=False)
                atoms.add(p)
    except Exception:
        pass
    return atoms


def safe_atom(value: str) -> str:
    escaped = []
    for char in value:
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
        elif ord(char) < 32 or ord(char) == 127:
            escaped.append(f"\\x{ord(char):x}\\")
        else:
            escaped.append(char)
    return "'" + "".join(escaped) + "'"
