"""
weather_agent.py - Weather specialist agent.
Fetches current weather via Open-Meteo API.
Callable by orchestrator via run(query) function.
Uses init_chat_model with model from config.py.
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools import get_weather
from config import WEATHER_MODEL, MAX_TOKENS, OLLAMA_BASE_URL

# --- Configuration ---
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a weather specialist assistant. "
    "Use the get_weather tool to fetch current weather for any city the user asks about. "
    "If the city is not found, tell the user directly and ask for a valid city name."
)

# --- Agent Setup ---
llm = init_chat_model(
    WEATHER_MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=SYSTEM_MESSAGE
)


def run(query: str) -> str:
    """Run the weather agent and return the final response as a string."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Log internal tool calls
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    [Weather Agent → Tool Call] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"    [Weather Agent ← Tool Result] {msg.content}")
    # Return final AI message
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content
    return "Weather agent could not generate a response."
