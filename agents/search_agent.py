"""
search_agent.py - Unified web search specialist agent.
Routes to Tavily or DuckDuckGo based on WEB_SEARCH_PROVIDER in config.py.
Three-step process: optimize query → search → summarize results.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config import WEB_SEARCH_PROVIDER, MAX_TOKENS, OLLAMA_BASE_URL

# --- Load the correct search tool based on config ---
if WEB_SEARCH_PROVIDER == "tavily":
    from tools.tavily_search import tavily_search as search_tool
    PROVIDER_NAME = "Tavily"
else:
    from tools.web_search import web_search as search_tool
    PROVIDER_NAME = "DuckDuckGo"

# --- Configuration ---
TEMPERATURE = 0.3

# --- LLM for query optimization and summarization ---
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)


def run(query: str) -> str:
    """Run the search agent with explicit query optimization."""
    from datetime import date
    today = date.today()

    # --- Step 1: Optimize the query for search ---
    optimize_prompt = (
        f"Today is {today.isoformat()}. The current year is {today.year}. "
        f"Rewrite the following user question into an optimal web search query. "
        f"Rules: "
        f"- Include the correct year (current={today.year}, last year={today.year - 1}) "
        f"- Use specific keywords, not full sentences "
        f"- Remove filler words "
        f"- Output ONLY the search query, nothing else.\n\n"
        f"User question: {query}"
    )

    optimized = llm.invoke([
        SystemMessage(content="You rewrite user questions into optimal search engine queries. Output only the query."),
        HumanMessage(content=optimize_prompt)
    ])
    search_query = optimized.content.strip().strip('"').strip("'")
    print(f"    [Search Agent ({PROVIDER_NAME}) → Optimized Query] {search_query}")

    # --- Step 2: Execute the search ---
    search_result = search_tool.invoke(search_query)
    print(f"    [Search Agent ({PROVIDER_NAME}) ← Result] {str(search_result)[:200]}...")

    # --- Step 3: Summarize the results ---
    summarize_prompt = (
        f"Today is {today.isoformat()} (year {today.year}). "
        f"Based on the following search results, answer the user's question concisely. "
        f"If results are from the wrong year, note the discrepancy. "
        f"Keep response under 500 characters. Plain text, no markdown.\n\n"
        f"User question: {query}\n\n"
        f"Search results:\n{str(search_result)[:1500]}"
    )

    summary = llm.invoke([
        SystemMessage(content="You summarize search results into concise answers."),
        HumanMessage(content=summarize_prompt)
    ])

    result = summary.content.strip()[:500]
    return result
