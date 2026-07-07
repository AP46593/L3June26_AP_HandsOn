"""
stock.py - Stock tools with hardcoded company data.
find_ticker: resolves company name to ticker symbol.
get_ticker_info: returns stock price for a ticker.
Currently supports: Zensar Technologies (ZENS), Apple Inc (AAPL).
"""

from langchain_core.tools import tool

# Hardcoded company data
COMPANIES = {
    "zensar technologies": {"ticker": "ZENS", "price": 145, "currency": "INR"},
    "apple": {"ticker": "AAPL", "price": 300, "currency": "USD"},
    "apple inc": {"ticker": "AAPL", "price": 300, "currency": "USD"},
}


@tool
def find_ticker(company: str) -> str:
    """Find the stock ticker symbol for a company name.

    Args:
        company: The company name to look up (e.g., 'Apple', 'Zensar Technologies').
    """
    key = company.lower().strip()
    for name, data in COMPANIES.items():
        if key in name or name in key:
            return f"Ticker for {company}: {data['ticker']}"
    return f"Could not find ticker for '{company}'. Currently only supporting: Zensar Technologies, Apple Inc."


@tool
def get_ticker_info(ticker: str) -> str:
    """Get stock price information for a given ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'ZENS').
    """
    ticker_upper = ticker.upper().strip()
    for name, data in COMPANIES.items():
        if data["ticker"] == ticker_upper:
            return f"{name.title()} ({ticker_upper}): Stock price {data['price']} {data['currency']}"
    return f"No information found for ticker '{ticker}'. Currently only supporting: ZENS, AAPL."
