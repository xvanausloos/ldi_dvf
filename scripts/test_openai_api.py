"""Simple script to test OPENAI_API_KEY and OpenAI API connectivity."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment or .env")
    sys.exit(1)

print("OPENAI_API_KEY found (length:", len(api_key), ")")
print("Calling OpenAI API...")

try:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
    )
    reply = response.choices[0].message.content.strip()
    print("Response:", reply)
    if reply.upper() == "OK":
        print("SUCCESS: OpenAI API is working.")
    else:
        print("WARNING: Unexpected reply, but API call succeeded.")
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)
