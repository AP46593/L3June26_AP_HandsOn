"""
pyt1.py - Simple interactive chat client using ChatOllama.
Sends user prompts to local Ollama endpoint with configurable
temperature and response length.
"""

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from config import ORCHESTRATOR_MODEL, MAX_TOKENS, OLLAMA_BASE_URL

# --- Local Configuration ---
TEMPERATURE = 0.7  # Adjust creativity: 0.0 (deterministic) to 1.0+ (creative)

# Model name without provider prefix for ChatOllama direct usage
MODEL = ORCHESTRATOR_MODEL.removeprefix("ollama:")


def chat(prompt: str, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
    llm = ChatOllama(
        model=MODEL,
        temperature=temperature,
        num_predict=max_tokens,
        base_url=OLLAMA_BASE_URL
    )

    response = llm.invoke(prompt)

    print(f"\nModel: {MODEL}")
    print(f"Temperature: {temperature}")
    print(f"Response:\n{response.content}")


if __name__ == "__main__":
    print("=== Ollama Chat ===")
    print(f"Model: {MODEL} | Max tokens: {MAX_TOKENS} | Temperature: {TEMPERATURE}")
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
