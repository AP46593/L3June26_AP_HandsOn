"""Quick test for Tavily search API."""
import os
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""
import urllib3
urllib3.disable_warnings()

from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
import requests

# Monkey-patch to disable SSL verification (corporate proxy)
old_post = requests.Session.post
def patched_post(self, *args, **kwargs):
    kwargs["verify"] = False
    return old_post(self, *args, **kwargs)
requests.Session.post = patched_post

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
result = client.search("FIFA World Cup 2026 latest results", max_results=3)

for r in result.get("results", []):
    title = r["title"]
    content = r["content"][:150]
    print(f"- {title}: {content}")
    print()
