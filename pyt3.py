"""
pyt3.py - Multi-agent orchestrator using create_agent.
Delegates to specialist agents: Weather, Calculator, Stock, Web Search (DuckDuckGo).
Includes Opik tracing, conversation memory, and truststore SSL fix.
"""

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from config import ORCHESTRATOR_MODEL, MAX_TOKENS, OLLAMA_BASE_URL, ENABLE_OPIK_TRACING
from tools.agent_tools import ask_weather_agent, ask_calc_agent, ask_stock_agent
from agents.search_agent import run as search_run

# --- Conditional Opik Tracing ---
opik_tracer = None
if ENABLE_OPIK_TRACING:
    from opik.integrations.langchain import OpikTracer
    opik_tracer = OpikTracer()

# --- Local Configuration ---
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a helpful orchestrator assistant. "
    "You can answer general questions directly. "
    "For weather-related questions, delegate to the weather agent using the ask_weather_agent tool. "
    "For math calculations, delegate to the calculator agent using the ask_calc_agent tool. "
    "For stock prices or company share information, first try the stock agent using the ask_stock_agent tool. "
    "If the stock agent cannot find the information, follow up by delegating to the web search agent using ask_web_search_agent tool to find the stock price online. "
    "For questions requiring current/live information, latest news, or recent events, delegate to the web search agent using the ask_web_search_agent tool. "
    "Keep your final responses concise, under 500 characters. No markdown formatting."
)

# Model name without provider prefix for ChatOllama direct usage
MODEL = ORCHESTRATOR_MODEL.removeprefix("ollama:")


# --- Web Search Tool (wraps search_agent) ---
from langchain_core.tools import tool

@tool
def ask_web_search_agent(query: str) -> str:
    """Delegate questions requiring live internet search to the web search specialist agent.
    Use this when the user asks about current events, latest news, recent updates,
    or any information that requires up-to-date internet search.
    Pass the user's full question as the query.
    """
    print(f"\n  [Web Search Agent] Received: {query}")
    result = search_run(query)
    print(f"  [Web Search Agent] Returning: {result}")
    return result


# --- Orchestrator Agent Setup ---
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)

orchestrator = create_agent(
    model=llm,
    tools=[ask_weather_agent, ask_calc_agent, ask_stock_agent, ask_web_search_agent],
    system_prompt=SYSTEM_MESSAGE
)


# --- Conversation Memory ---
conversation_history = []


def chat(user_input: str):
    print(f"\n{'='*60}")
    print(f"[User] {user_input}")
    print(f"{'='*60}")

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    invoke_config = {}
    if opik_tracer:
        invoke_config["callbacks"] = [opik_tracer]

    result = orchestrator.invoke(
        {"messages": conversation_history},
        config=invoke_config
    )

    # Find the last AI message with content (that's the final answer)
    final_answer_idx = None
    for i in range(len(result["messages"]) - 1, -1, -1):
        msg = result["messages"][i]
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            final_answer_idx = i
            break

    for i, msg in enumerate(result["messages"]):
        if msg.type == "human":
            continue
        elif msg.type == "ai" and msg.tool_calls:
            print(f"\n[Orchestrator → Delegating]")
            for tc in msg.tool_calls:
                print(f"  Tool: {tc['name']}")
                print(f"  Args: {tc['args']}")
        elif msg.type == "tool":
            print(f"\n[Agent Response ← {msg.name}]")
            print(f"  {msg.content[:300]}")
        elif msg.type == "ai" and msg.content:
            if i == final_answer_idx:
                print(f"\n[Orchestrator → Final Answer]")
                print(f"  {msg.content}")
                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": msg.content})

    print(f"\n{'='*60}")


if __name__ == "__main__":
    print("=== Orchestrator Agent ===")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print("I can chat or delegate to specialist agents.")
    print("Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not prompt:
            continue
        chat(prompt)
        print()
