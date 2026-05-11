import builtins
import sys
from io import StringIO

# Best-effort sandbox: removes the obvious foot-guns and limits imports to a safe
# stdlib whitelist. NOT a real security boundary — exec() can always be escaped.
_BLOCKED_BUILTINS = {"exec", "eval", "compile", "open", "input", "breakpoint", "__import__"}

_ALLOWED_MODULES = {
    "math", "cmath", "statistics", "random", "secrets", "fractions", "decimal", "numbers",
    "itertools", "functools", "operator", "collections", "heapq", "bisect", "array", "copy",
    "datetime", "calendar", "time", "json", "re", "string", "textwrap", "unicodedata",
    "pprint", "typing", "enum", "dataclasses",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict:
    safe = dict(vars(builtins))
    for name in _BLOCKED_BUILTINS:
        safe.pop(name, None)
    safe["__import__"] = _safe_import
    return safe


_SAFE_BUILTINS = _safe_builtins()


def run_python(code: str) -> str:
    buf = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        exec(code, {"__builtins__": _SAFE_BUILTINS})  # noqa: S102
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
    out = buf.getvalue()
    return out if out else "[Code ran successfully with no output]"
