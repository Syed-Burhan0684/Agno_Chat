# app/agent.py
import os, concurrent.futures
from agno.agent import Agent
from agno.models.openai import OpenAIChat

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "64"))

_MODEL = OpenAIChat(id=MODEL_NAME, api_key=API_KEY, temperature=0.0, max_tokens=MAX_TOKENS)
AGENT = Agent(model=_MODEL, description="Banking assistant (demo)", markdown=False)

FALLBACK_TEXT = "I could not generate a full answer instantly; quick note: please check your recent transactions."

def agent_run_with_timeout(prompt: str, timeout_sec: float = 2.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(AGENT.run, prompt)
        try:
            res = fut.result(timeout=timeout_sec)
            return getattr(res, 'content', None) or getattr(res, 'text', None) or str(res)
        except concurrent.futures.TimeoutError:
            return FALLBACK_TEXT
        except Exception as e:
            return f"(LLM error: {e})"
