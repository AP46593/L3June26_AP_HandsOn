"""
agent_tools.py - Tool wrappers that delegate to specialist agents.
Each wrapper logs the delegation and result.
Import these into any orchestrator to enable agent delegation.
"""

from langchain_core.tools import tool


@tool
def ask_weather_agent(query: str) -> str:
    """Delegate weather-related questions to the weather specialist agent.
    Use this when the user asks about current weather, temperature, or conditions in a city.
    Pass the user's full question as the query.
    """
    from agents.weather_agent import run as weather_run
    print(f"\n  [Weather Agent] Received: {query}")
    result = weather_run(query)
    print(f"  [Weather Agent] Returning: {result}")
    return result


@tool
def ask_calc_agent(query: str) -> str:
    """Delegate calculation questions to the calculator specialist agent.
    Use this when the user asks for math operations like addition, subtraction,
    multiplication, division, square, cube, square root, or cube root.
    Pass the user's full question as the query.
    """
    from agents.calc_agent import run as calc_run
    print(f"\n  [Calc Agent] Received: {query}")
    result = calc_run(query)
    print(f"  [Calc Agent] Returning: {result}")
    return result


@tool
def ask_stock_agent(query: str) -> str:
    """Delegate stock price questions to the stock specialist agent.
    Use this when the user asks about stock prices, share prices, or company valuations.
    Pass the user's full question as the query.
    """
    from agents.stock_agent import run as stock_run
    print(f"\n  [Stock Agent] Received: {query}")
    result = stock_run(query)
    print(f"  [Stock Agent] Returning: {result}")
    return result


@tool
def ask_web_search_agent(query: str) -> str:
    """Delegate questions requiring live internet search to the web search specialist agent.
    Use this when the user asks about current events, latest news, recent updates,
    or any information that requires up-to-date internet search.
    Pass the user's full question as the query.
    """
    from agents.web_search_agent import run as web_search_run
    print(f"\n  [Web Search Agent] Received: {query}")
    result = web_search_run(query)
    print(f"  [Web Search Agent] Returning: {result}")
    return result
