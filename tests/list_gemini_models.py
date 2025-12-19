import os
import google.generativeai as genai

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print("GEMINI_API_KEY not set")
    raise SystemExit(1)

genai.configure(api_key=key)

try:
    models = genai.list_models()
    print("Available models:")
    for m in models:
        # print some useful fields
        print(m)
except Exception as e:
    print("Error listing models:", e)
    raise
