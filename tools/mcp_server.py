from fastmcp import FastMCP
from tools.builtin.search import web_search
from tools.builtin.python_exec import run_python

mcp = FastMCP("agentflow-tools")


@mcp.tool()
def search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return web_search(query)


@mcp.tool()
def python_exec(code: str) -> str:
    """Execute Python code and return stdout."""
    return run_python(code)


if __name__ == "__main__":
    mcp.run()
