
import ast
import os

ROOT = "."

def inspect_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return

    print(f"\nMODULE: {filepath}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            print(f"\n  CLASS: {node.name}")

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    print(f"    METHOD: {item.name}()")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level functions
            if not any(
                isinstance(parent, ast.ClassDef)
                for parent in []
            ):
                pass

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"  FUNCTION: {node.name}()")

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            print(f"  IMPORT: {ast.unparse(node)}")


for root, dirs, files in os.walk(ROOT):
    dirs[:] = [
        d for d in dirs
        if d not in {".git", "__pycache__", ".venv", "venv"}
    ]

    for filename in sorted(files):
        if filename.endswith(".py"):
            inspect_file(os.path.join(root, filename))
