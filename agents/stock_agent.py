"""
stock_agent.py - Stock price specialist agent.
Chains two tools: find_ticker (company→ticker) then get_ticker_info (ticker→price).
Currently hardcoded with Zensar Technologies and Apple Inc.
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools.stock import find_ticker, get_ticker_info
from config import STOCK_MODEL, MAX_TOKENS, OLLAMA_BASE_URL

# --- Configuration ---
TEMPERATURE = 0.0
SYSTEM_MESSAGE = (
    "You are a stock price specialist assistant. "
    "When a user asks about a stock or company price: "
    "1. First use find_ticker to get the ticker symbol for the company. "
    "2. Then use get_ticker_info with that ticker to get the stock price. "
    "If a company or ticker is not found, tell the user it's not currently available. "
    "Always respond in plain text, no markdown formatting, no bold."
)

# --- Agent Setup ---
llm = init_chat_model(
    STOCK_MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)

agent = create_agent(
    model=llm,
    tools=[find_ticker, get_ticker_info],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the stock agent and return the final response as a string."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Stock Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Stock Agent ← Tool Result] {msg.content}")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content
    return "Stock agent could not generate a response."
