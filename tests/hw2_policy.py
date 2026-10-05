"""The same NumPy-only import policy is used locally and on BRUTE."""

import ast
import builtins
import importlib
import importlib.abc
import sys


FORBIDDEN = frozenset({"torch", "jax", "jaxlib", "autograd", "tensorflow", "keras", "cupy"})


class ForbiddenImport(ImportError):
    pass


def _string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string(node.left), _string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _qualified(node.value) + "." + node.attr
    return ""


def check_source(path):
    """Inspect imports throughout the AST, including functions and dead branches."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (node.module or "") + "." + alias.name
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Call):
            name = _qualified(node.func)
            root, _, tail = name.partition(".")
            resolved = aliases.get(root, root) + ("." + tail if tail else "")
            if resolved in {"__import__", "builtins.__import__", "importlib.import_module"}:
                argument = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg == "name"), None)
                value = _string(argument)
                if value:
                    names = [value]
        for name in names:
            if name.split(".")[0] in FORBIDDEN:
                raise ForbiddenImport(
                    f"{path.name}:{node.lineno}: importing {name!r} is forbidden in Part 1. "
                    "Use NumPy and your own derivatives; this rule applies to the entire file, "
                    "including imports inside functions. Part 1 receives 0 points.")


class ImportGuard(importlib.abc.MetaPathFinder):
    """Block dynamic imports too, and remember attempts even if student code catches them."""

    def __init__(self):
        self.attempts = []
        self.original_import = builtins.__import__
        self.original_import_module = importlib.import_module

    def check(self, name):
        if name.split(".")[0] in FORBIDDEN:
            self.attempts.append(name)
            raise ForbiddenImport(f"Importing {name!r} is forbidden in hw2_numpy.py. Use NumPy only.")

    def find_spec(self, fullname, path=None, target=None):
        self.check(fullname)
        return None

    def install(self):
        def guarded_import(name, *args, **kwargs):
            self.check(name)
            return self.original_import(name, *args, **kwargs)

        def guarded_import_module(name, *args, **kwargs):
            self.check(name)
            return self.original_import_module(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        importlib.import_module = guarded_import_module
        sys.meta_path.insert(0, self)
        return self
