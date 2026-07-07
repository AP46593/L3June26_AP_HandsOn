"""
pyt5.py - Multi-agent orchestrator built with LangGraph StateGraph.
Each specialist agent is an explicit graph node with conditional routing.
Features: explicit graph routing, MemorySaver checkpointing, Opik tracing,
session logging to User-Chat/ folder. Uses init_chat_model for provider-agnostic config.
"""

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from config import (
    ORCHESTRATOR_MODEL, MAX_TOKENS, OLLAMA_BASE_URL,
    ENABLE_OPIK_TRACING, ENABLE_SESSION_LOGGING, LOG_DIR
)

from typing import Annotated, Literal, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

if ENABLE_OPIK_TRACING:
    from opik.integrations.langchain import OpikTracer


# --- Configuration ---
MODEL = ORCHESTRATOR_MODEL
TEMPERATURE = 0.7
SYSTEM_MESSAGE = (
    "You are a helpful orchestrator assistant. Today's date is 2026-07-07. The current year is 2026. "
    "You have the following specialist capabilities: "
    "1. Weather Agent - get current weather for any city "
    "2. Calculator Agent - basic math (add, subtract, multiply, divide, square, cube, sqrt, cbrt) "
    "3. Stock Price Agent - look up stock prices (currently supports Zensar Technologies and Apple) "
    "4. Web Search Agent - search the internet for current events, news, and live information "
    "You can also answer general knowledge questions directly. "
    "When asked what you can do, mention all of the above capabilities. "
    "ONLY use routing when the user explicitly asks about one of these topics: "
    "- Weather in a specific city → respond with ROUTE:WEATHER on its own line "
    "- Math calculation → respond with ROUTE:CALC on its own line "
    "- Stock price of a company → respond with ROUTE:STOCK on its own line "
    "- Current events, news, live internet search, OR any question about recent/last year's events → respond with ROUTE:SEARCH on its own line "
    "IMPORTANT: Any question about events from 2024, 2025, or 2026 MUST be routed to ROUTE:SEARCH. "
    "Do NOT answer time-sensitive questions from your own knowledge — always route to search. "
    "When routing, include a clear rewritten query after the ROUTE keyword that resolves any pronouns. "
    "Example: if user says 'How is he doing this season' and context shows 'he' = Lando Norris, "
    "respond with: ROUTE:SEARCH Lando Norris 2026 F1 season performance "
    "For ALL other questions (greetings, general knowledge, 'what can you do', etc.), "
    "answer directly WITHOUT any ROUTE keyword. "
    "Keep responses under 500 characters. No markdown."
)


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # Tracks which agent to call


# --- LLM (no tools bound - routing is explicit) ---
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=TEMPERATURE,
    num_predict=MAX_TOKENS,
    base_url=OLLAMA_BASE_URL
)


# =============================================================================
# GRAPH NODES
# =============================================================================

def orchestrator_node(state: AgentState) -> AgentState:
    """Main orchestrator - decides to answer directly or route to a specialist."""
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages
    response = llm.invoke(messages)
    return {"messages": [response], "route": ""}


def weather_agent_node(state: AgentState) -> AgentState:
    """Weather specialist - fetches weather via Open-Meteo."""
    from agents.weather_agent import run as weather_run
    # Get the user query from the last human message
    query = _get_last_user_query(state)
    print(f"\n  [Weather Agent] Processing: {query}")
    result = weather_run(query)
    print(f"  [Weather Agent] Result: {result}")
    return {"messages": [AIMessage(content=result)], "route": "done"}


def calc_agent_node(state: AgentState) -> AgentState:
    """Calculator specialist - basic math operations."""
    from agents.calc_agent import run as calc_run
    query = _get_last_user_query(state)
    print(f"\n  [Calc Agent] Processing: {query}")
    result = calc_run(query)
    print(f"  [Calc Agent] Result: {result}")
    return {"messages": [AIMessage(content=result)], "route": "done"}


def stock_agent_node(state: AgentState) -> AgentState:
    """Stock specialist - finds tickers and prices."""
    from agents.stock_agent import run as stock_run
    query = _get_last_user_query(state)
    print(f"\n  [Stock Agent] Processing: {query}")
    result = stock_run(query)
    print(f"  [Stock Agent] Result: {result}")
    return {"messages": [AIMessage(content=result)], "route": "done"}


def web_search_node(state: AgentState) -> AgentState:
    """Web search specialist - searches internet via configured provider."""
    from agents.search_agent import run as search_run, PROVIDER_NAME
    query = _get_last_user_query(state)
    print(f"\n  [Web Search Agent ({PROVIDER_NAME})] Processing: {query}")
    result = search_run(query)
    print(f"  [Web Search Agent ({PROVIDER_NAME})] Result: {result}")
    return {"messages": [AIMessage(content=result)], "route": "done"}


def summarizer_node(state: AgentState) -> AgentState:
    """Takes agent results and formulates a final user-friendly response.
    Skips LLM call for short, already-clear answers.
    """
    messages = state["messages"]
    agent_response = messages[-1].content if messages else ""

    # If the response is already short and clear, skip the LLM summarization
    if len(agent_response) < 200:
        return {"messages": [AIMessage(content=agent_response)], "route": ""}

    # For longer responses, summarize
    prompt = (
        "Summarize the following information into a concise, helpful response "
        "for the user. Keep it under 500 characters. No markdown formatting.\n\n"
        f"Information: {agent_response}"
    )
    response = llm.invoke([SystemMessage(content="You are a helpful assistant."),
                           HumanMessage(content=prompt)])
    return {"messages": [response], "route": ""}


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def _get_last_user_query(state: AgentState) -> str:
    """Extract the query to pass to specialist agents.
    First checks if the orchestrator included a rewritten query after ROUTE:keyword.
    Falls back to the last human message.
    """
    # Check if orchestrator provided a rewritten query after ROUTE:
    for msg in reversed(state["messages"]):
        if msg.type == "ai" and msg.content:
            content = msg.content.strip()
            for prefix in ["ROUTE:WEATHER", "ROUTE:CALC", "ROUTE:STOCK", "ROUTE:SEARCH"]:
                if prefix in content:
                    # Extract everything after the ROUTE:keyword
                    after_route = content.split(prefix, 1)[1].strip()
                    if after_route:
                        return after_route
            break

    # Fallback to last human message
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return msg.content
    return ""


def route_from_orchestrator(state: AgentState) -> str:
    """Parse orchestrator response for routing keywords."""
    last_msg = state["messages"][-1]
    content = last_msg.content.strip() if last_msg.content else ""

    if "ROUTE:WEATHER" in content:
        return "weather_agent"
    elif "ROUTE:CALC" in content:
        return "calc_agent"
    elif "ROUTE:STOCK" in content:
        return "stock_agent"
    elif "ROUTE:SEARCH" in content:
        return "web_search"
    else:
        return END  # Direct answer, no routing needed


def route_after_agent(state: AgentState) -> str:
    """After a specialist responds, go to summarizer."""
    return "summarizer"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("weather_agent", weather_agent_node)
graph.add_node("calc_agent", calc_agent_node)
graph.add_node("stock_agent", stock_agent_node)
graph.add_node("web_search", web_search_node)
graph.add_node("summarizer", summarizer_node)

# Entry point
graph.add_edge(START, "orchestrator")

# Orchestrator routes to agents or ends
graph.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {
        "weather_agent": "weather_agent",
        "calc_agent": "calc_agent",
        "stock_agent": "stock_agent",
        "web_search": "web_search",
        END: END
    }
)

# Each agent goes to summarizer after completing
graph.add_edge("weather_agent", "summarizer")
graph.add_edge("calc_agent", "summarizer")
graph.add_edge("stock_agent", "summarizer")
graph.add_edge("web_search", "summarizer")

# Summarizer ends the flow
graph.add_edge("summarizer", END)

# Compile with MemorySaver
memory = MemorySaver()
orchestrator = graph.compile(checkpointer=memory)


# =============================================================================
# OPIK TRACER
# =============================================================================

if ENABLE_OPIK_TRACING:
    opik_tracer = OpikTracer()
else:
    opik_tracer = None

THREAD_ID = "default"


# =============================================================================
# SESSION LOGGING
# =============================================================================

import os
import uuid
import time
from datetime import datetime

if ENABLE_SESSION_LOGGING:
    os.makedirs(LOG_DIR, exist_ok=True)
    SESSION_ID = str(uuid.uuid4())[:8]
    MSG_FILE = os.path.join(LOG_DIR, f"pyt5-Message-{SESSION_ID}.txt")
    LOG_FILE = os.path.join(LOG_DIR, f"pyt5-agentLog-{SESSION_ID}.txt")

    with open(MSG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Session ID: {SESSION_ID}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"{'='*50}\n\n")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Session ID: {SESSION_ID}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"{'='*50}\n\n")


# =============================================================================
# CHAT FUNCTION
# =============================================================================

def chat(user_input: str):
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[User] {user_input}")
    print(f"{'='*60}")

    invoke_config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 10}
    if opik_tracer:
        invoke_config["callbacks"] = [opik_tracer]

    result = orchestrator.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=invoke_config
    )

    elapsed = time.time() - start_time

    # Collect agent log entries
    agent_log = []
    agent_log.append(f"Timestamp: {datetime.now().isoformat()}")
    agent_log.append(f"Latency: {elapsed:.2f}s")
    agent_log.append(f"Model: {MODEL}")
    agent_log.append(f"{'='*50}")

    # Display messages
    final_answer = ""
    for msg in result["messages"]:
        if msg.type == "human" or msg.type == "system":
            continue
        elif msg.type == "ai" and msg.content:
            agent_log.append(f"\n[AI] {msg.content[:300]}")

    # Last AI message is the final answer
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            final_answer = msg.content
            break

    print(f"\n[Final Answer]")
    print(f"  {final_answer}")
    print(f"  (Latency: {elapsed:.2f}s)")
    print(f"\n{'='*60}")

    # Append to session logs (if enabled)
    if ENABLE_SESSION_LOGGING:
        with open(MSG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] User: {user_input}\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Assistant: {final_answer}\n\n")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'─'*50}\n")
            f.write(f"Time: {datetime.now().isoformat()} | Latency: {elapsed:.2f}s\n")
            f.write(f"User: {user_input}\n")
            f.write("\n".join(agent_log[4:]))
            f.write("\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--graph" in sys.argv:
        print(orchestrator.get_graph().draw_mermaid())
        sys.exit(0)

    print("=== LangGraph Multi-Agent Orchestrator ===")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print("Nodes: orchestrator → weather_agent | calc_agent | stock_agent | web_search → summarizer")
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

    if opik_tracer:
        opik_tracer.flush()
