import json
from textwrap import shorten

import requests

# Ollama defaults
OLLAMA_HOST = "http://localhost:11434"
MODEL = "mistral"  # change to "llama3.1:8b-instruct" or another pulled model

PROMPT = """You are a helpful assistant.
Answer the following question clearly and concisely.

Question: What is the capital of France?
"""

print(f"🔍 Testing Ollama model '{MODEL}' at {OLLAMA_HOST} ...")

try:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": PROMPT,
            # optional settings
            "options": {"temperature": 0.2, "num_predict": 128},
            "stream": False,  # easier to parse
        },
        timeout=180,
    )
except Exception as e:
    print("❌ Could not reach Ollama API:", e)
    raise SystemExit(1)

if resp.status_code != 200:
    print(f"❌ Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")
    raise SystemExit(1)

try:
    data = resp.json()
except json.JSONDecodeError:
    print("❌ Response was not valid JSON.")
    print(resp.text[:500])
    raise SystemExit(1)

# Basic inspection
answer = data.get("response", "").strip()
tokens = data.get("eval_count", "N/A")
duration = data.get("total_duration", "N/A")

print("\n✅ Ollama responded successfully!")
print(f"Model: {MODEL}")
print(f"Response time: {duration} ns  |  Tokens evaluated: {tokens}")
print(f"Answer preview: {shorten(answer, width=200)}\n")

if "paris" in answer.lower():
    print("🎉 Looks good — the model seems to respond properly.")
else:
    print("⚠️ Model responded, but the output looks unexpected. Check the text above.")
