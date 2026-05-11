import os

from tavily import TavilyClient


def web_search(query: str, max_results: int = 5) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Search unavailable: set TAVILY_API_KEY (free tier at https://tavily.com)."
    client = TavilyClient(api_key=api_key)
    resp = client.search(query=query, max_results=max_results, include_answer=True)
    results = resp.get("results", [])
    answer = resp.get("answer")
    if not results and not answer:
        return "No results found."
    parts: list[str] = []
    if answer:
        parts.append(f"Summary: {answer}")
    for r in results:
        parts.append(f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}")
    return "\n\n".join(parts)
