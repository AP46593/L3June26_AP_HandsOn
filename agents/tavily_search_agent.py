"""
tavily_search_agent.py - Web search specialist agent using Tavily API.
Searches internet for live/current information.
Injects today's date for temporal accuracy.
Summarizes results within 500 characters.
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools.tavily_search import tavily_search
from config import TAVILY_SEARCH_MODEL, MAX_TOKENS, OLLAMA_BASE_URL

# --- Configuration ---
TEMPERATURE = 0.3
SYSTEM_MESSAGE = (
    "You are a web search specialist assistant. "
    "Use the tavily_search tool to search the internet for current, live information. "
    "CRITICAL RULES: "
    "1. The current year is 2026. ALWAYS include '2026' in your search queries when the user asks about 'current', 'latest', or 'this season'. "
    "2. NEVER search for 2024 or 2025 unless the user explicitly asks about those years. "
    "3. If search results return data from a different year than requested, discard them and state that current data was not found. "
    "4. Make only 1-2 search calls maximum. Do not retry with similar queries. "
    "After getting results, summarize concisely under 500 characters. "
    "Respond in plain text, no markdown formatting."
)

# --- Agent Setup ---
llm = init_chat_model(
    TAVILY_SEARCH_MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)

agent = create_agent(
    model=llm,
    tools=[tavily_search],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the Tavily search agent and return the final response."""
    from datetime import date
    today = date.today()
    dated_query = f"[Today is {today.isoformat()}, current year is {today.year}] {query}"
    result = agent.invoke({"messages": [{"role": "user", "content": dated_query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Tavily Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Tavily Agent ← Tool Result] {msg.content[:200]}...")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content[:500]
    return "Tavily search agent could not generate a response."
