# app/main.py
import os, time, re, json
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from app.agent import agent_run_with_timeout, AGENT, FALLBACK_TEXT
from app.db import VectorDB

import concurrent.futures

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", "2.0"))  # 2s default
TOP_K = int(os.getenv("TOP_K", "3"))
EF_SEARCH = int(os.getenv("EF_SEARCH", "64"))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'transactions.json')

vdb = VectorDB(path_json=DATA_PATH)
app = FastAPI(title='Agno Chat Demo')
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "static"))
app.mount('/static', StaticFiles(directory=os.path.join(BASE_DIR, "app", 'static')), name='static')

CHAT_HISTORY = {}
REFINED_RESULTS = {}
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


def extract_price(text: str):
    """
    Extract a numeric price only if there's clear money context.
    Examples: "$200", "200 USD", "pay 300", "buy for 150"
    Will ignore plain numbers like '3' in 'last 3 transactions'.
    """
    money_words = ["usd", "dollars", "bucks", "eur", "rs", "inr"]
    purchase_verbs = ["buy", "afford", "pay", "purchase", "spend"]

    # currency with symbol or word
    m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if not m:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(%s)" % "|".join(money_words), text, flags=re.I)

    if m:
        try:
            return float(m.group(1))
        except:
            return None

    # numeric amount with purchase verb nearby
    for v in purchase_verbs:
        m = re.search(rf"{v}\s+(?:for\s+)?([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except:
                return None
    return None


def summarize_history(user_id: str, max_turns: int = 6) -> str:
    hist = CHAT_HISTORY.get(user_id, [])[-max_turns:]
    if not hist:
        return "No prior messages."
    return "\n".join(f"{h['role']}: {h['content']}" for h in hist)


def build_prompt(user_id: str, user_msg: str, profile: dict, evidence: list):
    profile_summary = f"Balance: ${profile.get('current_balance',0):.0f}; income: ${profile.get('monthly_income',0):.0f}."
    ev_lines = " ".join([e['text'] if len(e['text']) < 120 else e['text'][:115] + "..." for e in evidence[:3]])
    prompt = (
        f"You are a concise banking assistant. Answer in 1 short sentence.\n"
        f"Profile: {profile_summary}\n"
        f"Evidence: {ev_lines}\n"
        f"User question: {user_msg}\n"
        f"Give a direct recommendation (yes/no/short advice)."
    )
    return prompt

def fast_rule_answer(user_msg: str, profile: dict, safety_buffer: float = 200.0):
    """
    Quick yes/no affordability rule.
    Runs only if a valid price with money context is detected.
    """
    amt = extract_price(user_msg)
    if amt is not None and profile:
        bal = profile.get('current_balance', 0.0)
        available = bal - safety_buffer
        if available >= amt:
            return f"Yes — you can afford ${amt:.0f} now (balance ${bal:.0f})."
        else:
            return f"No — you likely don't have enough for ${amt:.0f} now (balance ${bal:.0f})."
    return None

def background_refine(user_id: str, prompt: str, evidence):
    try:
        run_res = AGENT.run(prompt)
        refined_text = getattr(run_res, 'content', None) or getattr(run_res, 'text', None) or str(run_res)
    except Exception as e:
        refined_text = f"(Refinement failed: {e})"
    REFINED_RESULTS[user_id] = {'text': refined_text, 'evidence': evidence, 'ts': time.time()}
    CHAT_HISTORY.setdefault(user_id, []).append({'role': 'assistant', 'content': refined_text})


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/health')
def health():
    return {'ok': True}


class ChatIn(BaseModel):
    user_id: str
    message: str


@app.post('/chat')
def chat(inp: ChatIn):
    t0 = time.time()
    profile = vdb.get_profile(inp.user_id)
    if not profile:
        raise HTTPException(status_code=404, detail='unknown user_id')

    CHAT_HISTORY.setdefault(inp.user_id, []).append({'role': 'user', 'content': inp.message})

    # 1) Fast rule (affordability) — only triggers on monetary intent
    fast_ans = fast_rule_answer(inp.message, profile)
    if fast_ans:
        evidence = vdb.retrieve(inp.user_id, inp.message, top_k=TOP_K, ef_search=EF_SEARCH)
        prompt = build_prompt(inp.user_id, inp.message, profile, evidence)
        executor.submit(background_refine, inp.user_id, prompt, evidence)
        latency = (time.time() - t0) * 1000.0
        return {'reply': fast_ans, 'latency_ms': latency, 'evidence': [], 'refined_pending': True}

    # 2) RAG + LLM path
    evidence = vdb.retrieve(inp.user_id, inp.message, top_k=TOP_K, ef_search=EF_SEARCH)
    prompt = build_prompt(inp.user_id, inp.message, profile, evidence)
    result_text = agent_run_with_timeout(prompt, timeout_sec=LLM_TIMEOUT)

    if result_text == FALLBACK_TEXT:
        executor.submit(background_refine, inp.user_id, prompt, evidence)
        latency = (time.time() - t0) * 1000.0
        return {'reply': result_text, 'latency_ms': latency, 'evidence': evidence, 'refined_pending': True}

    CHAT_HISTORY.setdefault(inp.user_id, []).append({'role': 'assistant', 'content': result_text})
    latency = (time.time() - t0) * 1000.0
    return {'reply': result_text, 'latency_ms': latency, 'evidence': evidence, 'refined_pending': False}


@app.get('/refined/{user_id}')
def get_refined(user_id: str):
    payload = REFINED_RESULTS.pop(user_id, None)
    if not payload:
        return {'ready': False}
    return {'ready': True, 'text': payload['text'], 'evidence': payload.get('evidence', [])}


@app.post('/store')
def store(user_id: str = Form(...), text: str = Form(...)):
    vdb.add_transaction(user_id, text)
    return {'ok': True, 'user_id': user_id, 'text': text}


@app.post('/reset/{user_id}')
def reset(user_id: str):
    CHAT_HISTORY[user_id] = []
    return {'ok': True, 'user_id': user_id, 'msg': 'history cleared'}
