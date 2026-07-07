"""Test truststore - uses Windows certificate store for SSL."""
import truststore
truststore.inject_into_ssl()

import requests
import httpx

# Test 1: requests (used by Tavily, Open-Meteo)
print("Testing requests (Open-Meteo)...")
r = requests.get("https://geocoding-api.open-meteo.com/v1/search?name=Pune&count=1")
print(f"  Status: {r.status_code} ✓")

# Test 2: httpx (used by Opik, LangSmith)
print("Testing httpx (api.smith.langchain.com)...")
with httpx.Client() as client:
    r = client.get("https://api.smith.langchain.com/info")
    print(f"  Status: {r.status_code} ✓")

print("\nAll SSL connections successful via Windows trust store!")
