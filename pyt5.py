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
    "You are a helpful orchestrator assistant. "
    "You can answer general questions directly. "
    "When you need a specialist, respond with EXACTLY one of these routing keywords on a line by itself: "
    "ROUTE:WEATHER - for weather questions about a city "
    "ROUTE:CALC - for math calculations "
    "ROUTE:STOCK - for stock prices or company share info "
    "ROUTE:SEARCH - for current events, news, or live internet info "
    "If you can answer directly without a specialist, just respond normally. "
    "IMPORTANT: Only answer what the user asked. Keep responses under 500 characters. No markdown."
)


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # Tracks which agent to call


# --- LLM (no tools bound - routing is explicit) ---
llm = init_chat_model(
    MODEL,
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


def tavily_search_node(state: AgentState) -> AgentState:
    """Web search specialist - searches internet via Tavily."""
    from agents.tavily_search_agent import run as tavily_run
    query = _get_last_user_query(state)
    print(f"\n  [Tavily Search Agent] Processing: {query}")
    result = tavily_run(query)
    print(f"  [Tavily Search Agent] Result: {result}")
    return {"messages": [AIMessage(content=result)], "route": "done"}


def summarizer_node(state: AgentState) -> AgentState:
    """Takes agent results and formulates a final user-friendly response."""
    messages = state["messages"]
    prompt = (
        "Summarize the following information into a concise, helpful response "
        "for the user. Keep it under 500 characters. No markdown formatting.\n\n"
        f"Information: {messages[-1].content}"
    )
    response = llm.invoke([SystemMessage(content="You are a helpful assistant."),
                           HumanMessage(content=prompt)])
    return {"messages": [response], "route": ""}


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def _get_last_user_query(state: AgentState) -> str:
    """Extract the last user message from state."""
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
        return "tavily_search"
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
graph.add_node("tavily_search", tavily_search_node)
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
        "tavily_search": "tavily_search",
        END: END
    }
)

# Each agent goes to summarizer after completing
graph.add_edge("weather_agent", "summarizer")
graph.add_edge("calc_agent", "summarizer")
graph.add_edge("stock_agent", "summarizer")
graph.add_edge("tavily_search", "summarizer")

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
    print("Nodes: orchestrator → weather_agent | calc_agent | stock_agent | tavily_search → summarizer")
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
