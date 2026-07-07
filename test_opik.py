"""Quick test for Opik cloud connectivity."""
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

import os
import opik

# Configure Opik
opik.configure(
    api_key=os.getenv("OPIK_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),
    use_local=False
)

# Quick trace test
@opik.track
def test_function(input_text: str) -> str:
    return f"Processed: {input_text}"

result = test_function("Hello from Opik test!")
print(f"Result: {result}")
print("Check your Opik dashboard for the trace.")
