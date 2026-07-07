"""
web_search_agent.py - Web search specialist agent using DuckDuckGo.
Searches internet for live/current information.
Summarizes results within 500 characters. Alternative to Tavily agent.
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools.web_search import web_search
from config import WEB_SEARCH_MODEL, MAX_TOKENS, OLLAMA_BASE_URL

# --- Configuration ---
TEMPERATURE = 0.3
SYSTEM_MESSAGE = (
    "You are a web search specialist assistant. "
    "Use the duckduckgo_results_json tool to search the internet for current, live information. "
    "After getting search results, summarize the key findings concisely. "
    "Keep your final response under 500 characters. "
    "Include relevant source links when possible. "
    "Respond in plain text, no markdown formatting."
)

# --- Agent Setup ---
llm = init_chat_model(
    WEB_SEARCH_MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)

agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the web search agent and return the final response."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Web Search Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Web Search Agent ← Tool Result] {msg.content[:200]}...")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content[:500]
    return "Web search agent could not generate a response."
