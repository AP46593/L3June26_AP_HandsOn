"""
pyt1.py - Simple interactive chat client using ChatOllama.
Sends user prompts to local Ollama endpoint with configurable
temperature and response length.
"""

from langchain_ollama import ChatOllama

# --- Configuration ---
MODEL = "gpt-oss:120b-cloud"
MAX_TOKENS = 500          # Limit response length
TEMPERATURE = 0.7         # Adjust creativity: 0.0 (deterministic) to 1.0+ (creative)
OLLAMA_BASE_URL = "http://localhost:11434"


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
