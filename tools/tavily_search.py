"""
tavily_search.py - Tavily web search tool for live internet queries.
Requires TAVILY_API_KEY in .env.
Returns top 5 results with title and content snippet.
"""

import os
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def tavily_search(query: str) -> str:
    """Search the internet for current, live information using Tavily.
    Use this for news, recent events, or any up-to-date information.

    Args:
        query: The search query string.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query, max_results=5)
        results = []
        for r in response.get("results", []):
            results.append(f"- {r['title']}: {r['content'][:150]}")
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Error searching: {e}"
