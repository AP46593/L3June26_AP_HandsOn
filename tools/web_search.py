"""
web_search.py - DuckDuckGo web search tool for live internet queries.
No API key required.
Returns top 5 search results via langchain_community.
"""

import warnings
warnings.filterwarnings("ignore")

from langchain_community.tools import DuckDuckGoSearchResults

# Expose the search tool directly
web_search = DuckDuckGoSearchResults(num_results=5)
