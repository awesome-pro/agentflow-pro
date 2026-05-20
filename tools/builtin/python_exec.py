import ast
import builtins
import sys
from io import StringIO

# Best-effort sandbox: removes the obvious foot-guns and limits imports to a
# safe whitelist. NOT a real security boundary — exec() can always be escaped.
_BLOCKED_BUILTINS = {"exec", "eval", "compile", "open", "input", "breakpoint", "__import__"}

_ALLOWED_MODULES = {
    # stdlib
    "math", "cmath", "statistics", "random", "secrets", "fractions", "decimal", "numbers",
    "itertools", "functools", "operator", "collections", "heapq", "bisect", "array", "copy",
    "datetime", "calendar", "time", "json", "re", "string", "textwrap", "unicodedata",
    "pprint", "typing", "enum", "dataclasses",
    # third-party math — needed for AIME-style symbolic/numeric problems
    "sympy", "mpmath", "numpy",
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


def _parse_lenient(code: str) -> ast.Module:
    """Parse code, tolerating the stray leading whitespace LLMs often emit.

    Small models frequently produce flat top-level scripts with a stray space
    or two on some lines. If the code fails to parse but compiles cleanly once
    every line's leading whitespace is removed, the indentation was noise and
    the stripped version is used. Genuinely nested code won't compile when
    stripped (the block bodies vanish), so it falls through to the real error.
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        stripped = "\n".join(line.lstrip() for line in code.splitlines())
        return ast.parse(stripped)  # re-raises if it is a real syntax error


def run_python(code: str) -> str:
    """Execute code in a restricted sandbox.

    Behaves like a REPL: if the final statement is a bare expression, its value
    is printed automatically — so the model doesn't have to remember an explicit
    print() (LLMs are trained on notebook-style code that auto-displays results).
    """
    buf = StringIO()
    old_stdout = sys.stdout
    try:
        tree = _parse_lenient(code)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
            ast.fix_missing_locations(last_expr)
        namespace = {"__builtins__": _SAFE_BUILTINS}
        sys.stdout = buf
        exec(compile(tree, "<sandbox>", "exec"), namespace)  # noqa: S102
        if last_expr is not None:
            value = eval(compile(last_expr, "<sandbox>", "eval"), namespace)  # noqa: S307
            if value is not None:
                print(repr(value))
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
    out = buf.getvalue()
    return out if out else "[Code ran successfully with no output]"
