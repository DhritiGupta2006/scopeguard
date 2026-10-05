"""ScopeGuard agent core.

Design rule: the LLM judges and writes; code verifies and counts.
  * Clauses are parsed deterministically from the scope text.
  * The agent must cite a verbatim quote; code re-verifies it against the clause.
  * Revision counting, the rate card and the running unbilled total are plain code.
  * If the model is unavailable or cannot produce a verifiable answer, the message is
    marked "needs_review" for the human instead of being guessed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

def _load_dotenv() -> None:
    """Read KEY=VALUE lines from .env next to this file without overriding real environment variables."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

MAX_STEPS = 6
MODEL =os.environ.get("SCOPEGUARD_MODEL", "claude-sonnet-5-5")
CACHE_PATH = Path(os.environ.get("SCOPEGUARD_CACHE", Path(__file__).parent / "data" / "cache.json"))
_cache_lock = threading.Lock()

VERDICTS = ("in_scope", "out_of_scope", "ambiguous", "no_request")

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
STOPWORDS = set(
    "a an the and or of to in on for with we you i it is are be can could would please just also "
    "that this some any our your my me us do does make add want need get have has".split()
)

SYSTEM = """You are ScopeGuard, an assistant for a freelancer who has a signed scope document.
For ONE new client message, decide whether the request is inside the agreed scope.

Process (use the tools, do not guess):
1. Call search_scope_clauses to read the clauses relevant to the request.
2. If the request is a change to work already delivered or in progress, call get_revision_count.
3. If it is a real request, estimate the effort in hours with your own judgement, then call estimate_price.
4. Call submit_verdict exactly once with your decision.

Rules:
- verdict "in_scope": a clause includes this work. "out_of_scope": a clause excludes it or no clause covers it.
- verdict "ambiguous": the request is vague or could reasonably fall either way. Provide a short, friendly
  clarifying_question to send to the client. Never accuse the client.
- verdict "no_request": thanks, pleasantries or pressure with no concrete work requested.
- For in_scope and out_of_scope you MUST cite clause_ids and a quote copied word for word from one cited clause.
- Set is_revision=true when the request tweaks work that already exists (colors, fonts, layout, wording of delivered
  pages), false when it asks for a new deliverable or feature.
- Always give estimated_hours for any concrete request. Do not do currency arithmetic yourself.
- Do not decide the revision limit yourself: report is_revision and the system will count rounds."""

TOOL_SCHEMAS = [
    {
        "name": "search_scope_clauses",
        "description": "Search the signed scope document. Returns the most relevant clauses with their ids and full text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords describing the client's request"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_revision_count",
        "description": "Returns revision rounds already used and the limit stated in the scope.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "estimate_price",
        "description": "Converts an hours estimate into a price using the freelancer's rate card.",
        "input_schema": {
            "type": "object",
            "properties": {"hours": {"type": "number"}},
            "required": ["hours"],
        },
    },
    {
        "name": "verify_citation",
        "description": "Checks that a quote appears word for word in a clause. Use before submit_verdict.",
        "input_schema": {
            "type": "object",
            "properties": {"clause_id": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["clause_id", "quote"],
        },
    },
    {
        "name": "submit_verdict",
        "description": "Submit the final decision for this client message. Call exactly once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "request_summary": {"type": "string", "description": "One short line: what the client is asking for"},
                "clause_ids": {"type": "array", "items": {"type": "string"}},
                "quote": {"type": "string", "description": "Verbatim quote from a cited clause"},
                "reasoning": {"type": "string", "description": "One or two plain sentences"},
                "is_revision": {"type": "boolean"},
                "estimated_hours": {"type": "number"},
                "clarifying_question": {"type": "string"},
            },
            "required": ["verdict", "request_summary", "reasoning"],
        },
    },
]


# ----------------------------------------------------------------------------- scope parsing
def parse_scope(text: str) -> list[dict]:
    """Numbered lines ("1. ...", "2) ...", "3.1 ...") become clauses. Other lines before the
    first clause are treated as a header; lines after are continuations of the last clause."""
    clauses: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(.+)$", line)
        if m:
            body = m.group(2).strip()
            title = body.split(":", 1)[0].strip() if ":" in body[:48] else " ".join(body.split()[:5])
            clauses.append({"id": m.group(1), "title": title, "text": body})
        elif clauses:
            clauses[-1]["text"] += " " + line
    return clauses


def revision_limit(clauses: list[dict]) -> tuple[int | None, str | None]:
    pat = re.compile(r"(?:up to|maximum of|max(?:imum)?|limited to|includes?)\s+(\d+|one|two|three|four|five|six)\s+(?:rounds?|revisions?)", re.I)
    for c in clauses:
        m = pat.search(c["text"])
        if m:
            v = m.group(1).lower()
            return (int(v) if v.isdigit() else NUMBER_WORDS[v]), c["id"]
    return None, None


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _tokens(s: str) -> set[str]:
    return {_stem(w) for w in _norm(s).split() if w not in STOPWORDS and len(w) > 2}


# ----------------------------------------------------------------------------- tools + state
def new_state() -> dict:
    return {"revisions_used": 0, "unbilled": 0, "items": []}


class ScopeTools:
    def __init__(self, clauses: list[dict], rate: float, state: dict):
        self.clauses = clauses
        self.by_id = {c["id"]: c for c in clauses}
        self.rate = float(rate)
        self.state = state
        self.limit, self.limit_clause = revision_limit(clauses)

    def search_scope_clauses(self, query: str) -> dict:
        q = _tokens(query)
        scored = [(len(q & _tokens(c["title"] + " " + c["text"])), i, c) for i, c in enumerate(self.clauses)]
        hits = [c for s, i, c in sorted(scored, key=lambda t: (-t[0], t[1])) if s > 0][:4]
        if not hits:  # nothing matched: show everything rather than hide a clause
            hits = self.clauses[:8]
        return {"clauses": [{"id": c["id"], "text": c["text"]} for c in hits]}

    def get_revision_count(self) -> dict:
        return {"rounds_used": self.state["revisions_used"], "limit": self.limit, "clause_id": self.limit_clause}

    def estimate_price(self, hours: float) -> dict:
        return {"hours": hours, "rate_per_hour": self.rate, "price": round(float(hours) * self.rate)}

    def verify_citation(self, clause_id: str, quote: str) -> dict:
        c = self.by_id.get(str(clause_id))
        if not c:
            return {"verified": False, "reason": f"no clause with id {clause_id}"}
        qn = _norm(quote)
        if len(qn.split()) < 3:
            return {"verified": False, "reason": "quote too short; copy at least three words"}
        ok = qn in _norm(c["text"])
        return {"verified": ok, "reason": "ok" if ok else "quote not found word for word in that clause"}

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name not in {"search_scope_clauses", "get_revision_count", "estimate_price", "verify_citation"} or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def check_submission(self, sub: dict) -> tuple[bool, str]:
        """Reliability gate. Returns (ok, message_for_the_agent)."""
        v = sub.get("verdict")
        if v not in VERDICTS:
            return False, f"verdict must be one of {VERDICTS}"
        if v in ("in_scope", "out_of_scope"):
            ids = [str(i) for i in (sub.get("clause_ids") or [])]
            if not ids or not sub.get("quote"):
                return False, "in_scope and out_of_scope need clause_ids and a verbatim quote"
            if not any(self.verify_citation(i, sub["quote"])["verified"] for i in ids):
                return False, "quote is not word for word inside any cited clause; read the clause again and copy exactly"
            if not sub.get("estimated_hours") or float(sub["estimated_hours"]) <= 0:
                return False, "give estimated_hours greater than 0 for a concrete request"
        if v == "ambiguous" and not sub.get("clarifying_question"):
            return False, "ambiguous verdicts need a clarifying_question"
        return True, "accepted"


def _short(obj, n=150) -> str:
    s = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------- the agent loop
def run_agent(client, scope_text: str, rate: float, message: str, state: dict, model: str = MODEL) -> tuple[dict | None, list[dict], str]:
    """Returns (accepted_submission | None, trace, failure_reason)."""
    clauses = parse_scope(scope_text)
    if not clauses:
        return None, [], "no numbered clauses found in the scope"
    tools = ScopeTools(clauses, rate, state)
    toc = "\n".join(f"  {c['id']}. {c['title']}" for c in clauses)
    messages = [{"role": "user", "content": f"Scope clauses (titles only, use search_scope_clauses to read them):\n{toc}\n\n"
                 f"Rate: {rate} per hour. Revision rounds used so far: {state['revisions_used']}.\n\nNew client message:\n\"\"\"{message}\"\"\""}]
    trace: list[dict] = []
    for _ in range(MAX_STEPS):
        resp = client.messages.create(model=model, max_tokens=1024, system=SYSTEM, messages=messages,
                                      tools=TOOL_SCHEMAS, tool_choice={"type": "any"})
        messages.append({"role": "assistant", "content": resp.content})
        results, accepted = [], None
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name == "submit_verdict":
                ok, note = tools.check_submission(dict(block.input))
                trace.append({"tool": "submit_verdict", "input": _short({k: v for k, v in block.input.items() if k in ("verdict", "clause_ids")}), "output": note})
                if ok:
                    accepted = dict(block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": note, "is_error": not ok})
            else:
                out = tools.call(block.name, dict(block.input))
                trace.append({"tool": block.name, "input": _short(dict(block.input)), "output": _short(out)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
        if accepted:
            return accepted, trace, ""
        if not results:
            return None, trace, "model returned no tool call"
        messages.append({"role": "user", "content": results})
    return None, trace, "agent did not reach a verifiable verdict in the step limit"


# ----------------------------------------------------------------------------- deterministic ledger layer
def apply_ledger(sub: dict, scope_text: str, rate: float, state: dict) -> tuple[dict, dict]:
    """Takes the accepted submission, applies revision counting and money. Returns (result_fields, new_state)."""
    clauses = parse_scope(scope_text)
    limit, limit_clause = revision_limit(clauses)
    st = copy.deepcopy(state)
    verdict = sub["verdict"]
    hours = float(sub.get("estimated_hours") or 0)
    res = {
        "verdict": verdict,
        "request_summary": sub.get("request_summary", ""),
        "clause_ids": [str(i) for i in (sub.get("clause_ids") or [])],
        "quote": sub.get("quote", ""),
        "reasoning": sub.get("reasoning", ""),
        "is_revision": bool(sub.get("is_revision")),
        "estimated_hours": hours,
        "price": 0,
        "clarifying_question": sub.get("clarifying_question", ""),
        "override_note": "",
    }
    if verdict == "in_scope" and res["is_revision"]:
        st["revisions_used"] += 1
        if limit is not None and st["revisions_used"] > limit:
            lc = next(c for c in clauses if c["id"] == limit_clause)
            res.update(verdict="out_of_scope", clause_ids=[lc["id"]], quote=lc["text"],
                       override_note=f"Revision round {st['revisions_used']} exceeds the {limit} included in the scope.",
                       reasoning=f"This would be revision round {st['revisions_used']}, but the scope includes only {limit}.")
            if hours <= 0:
                hours = 1.0
                res["estimated_hours"] = hours
    res["revision_count"] = st["revisions_used"]
    res["revision_limit"] = limit
    if res["verdict"] == "out_of_scope":
        res["price"] = round(hours * float(rate))
        st["unbilled"] += res["price"]
        st["items"].append({"summary": res["request_summary"], "hours": hours, "price": res["price"], "clause_ids": res["clause_ids"]})
    return res, st


def needs_review_result(reason: str, state: dict, scope_text: str) -> tuple[dict, dict]:
    limit, _ = revision_limit(parse_scope(scope_text))
    return ({"verdict": "needs_review", "request_summary": "", "clause_ids": [], "quote": "", "reasoning": reason,
             "is_revision": False, "estimated_hours": 0, "price": 0, "clarifying_question": "", "override_note": "",
             "revision_count": state["revisions_used"], "revision_limit": limit}, copy.deepcopy(state))


# ----------------------------------------------------------------------------- cache (replay of real model outputs only)
def _key(scope_text: str, rate: float, message: str) -> str:
    return hashlib.sha1(f"{_norm(scope_text)}|{rate}|{_norm(message)}".encode()).hexdigest()


def _cache_load() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_store(key: str, sub: dict, trace: list[dict]) -> None:
    with _cache_lock:
        data = _cache_load()
        data[key] = {"sub": sub, "trace": trace}
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


class OpenAICompatClient:
    """Adapter exposing client.messages.create(...) in the Anthropic shape on top of any
    OpenAI-compatible /chat/completions endpoint (Gemini, Groq, OpenAI, OpenRouter...). Stdlib only."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 40.0):
        self.api_key, self.base_url, self.model, self.timeout = api_key, base_url.rstrip("/"), model, timeout
        self.messages = SimpleNamespace(create=self._create)

    @staticmethod
    def _convert(system: str, messages) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            c = m["content"]
            if m["role"] == "user":
                if isinstance(c, str):
                    out.append({"role": "user", "content": c})
                else:  # list of tool_result dicts
                    for r in c:
                        out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": str(r["content"])})
            else:
                text = "".join(b.text for b in c if b.type == "text")
                calls = [{"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                         for b in c if b.type == "tool_use"]
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    def _create(self, model=None, max_tokens=1024, system="", messages=(), tools=None, tool_choice=None, **_):
        body = {"model": self.model, "messages": self._convert(system, messages), "max_tokens": max(max_tokens, 2048), "temperature": 0}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
            if tool_choice and tool_choice.get("type") == "any":
                body["tool_choice"] = "required"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key,
                                              "User-Agent": "ScopeGuard/1.0"})
        backoff = (1, 2)
        for attempt in range(len(backoff) + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")
                # Rate limits and overload clear within seconds, and the model sometimes emits a malformed
                # tool call that succeeds on the next try. Retry those; fail fast on everything else.
                retryable = e.code in (429, 503) or (e.code == 400 and ("tool_use_failed" in detail or "failed_generation" in detail))
                if retryable and attempt < len(backoff):
                    time.sleep(backoff[attempt])
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail[:200]}") from None
        msg = data["choices"][0]["message"]
        blocks = []
        if msg.get("content"):
            blocks.append(SimpleNamespace(type="text", text=msg["content"]))
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc["function"]
            args = fn.get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                args = {}  # malformed arguments: the gate will reject and the agent retries
            blocks.append(SimpleNamespace(type="tool_use", id=tc.get("id") or f"call_{i}", name=fn["name"], input=args))
        return SimpleNamespace(content=blocks)


def get_client():
    if os.environ.get("LLM_API_KEY"):
        return OpenAICompatClient(
            os.environ["LLM_API_KEY"],
            os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            os.environ.get("LLM_MODEL", "gemini-2.5-flash"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic(timeout=30.0, max_retries=1)
    return None


def analyze_message(client, scope_text: str, rate: float, message: str, state: dict, model: str = MODEL, use_cache: bool = True) -> tuple[dict, dict]:
    """Full pipeline for one client message. Never raises; returns (result, new_state)."""
    key = _key(scope_text, rate, message)
    sub, trace, why, source = None, [], "", "live"
    if client is not None:
        try:
            sub, trace, why = run_agent(client, scope_text, rate, message, state, model)
        except Exception as e:  # network, auth, rate limit...
            sub, why = None, f"model unavailable ({type(e).__name__}: {str(e)[:140]})"
    else:
        why = "no model configured"
    if sub is not None:
        if use_cache:
            _cache_store(key, sub, trace)
    elif use_cache:
        hit = _cache_load().get(key)
        if hit:
            sub, trace, source = hit["sub"], hit["trace"], "cached"
    if sub is None:
        res, st = needs_review_result(f"Could not verify this one automatically ({why}). Please judge it yourself.", state, scope_text)
        res.update(trace=trace, source="none")
        return res, st
    res, st = apply_ledger(sub, scope_text, rate, state)
    res.update(trace=trace, source=source)
    return res, st


# ----------------------------------------------------------------------------- change order
COVER_FALLBACK = ("Hi! Thanks for the latest requests. These items fall outside the scope we agreed on, so I've "
                  "listed them below with an estimate for each. If you'd like me to go ahead, just confirm and I'll "
                  "start once approved. Happy to discuss any of them first.")


def build_change_order(client, items: list[dict], rate: float, model: str = MODEL) -> dict:
    total = sum(int(i["price"]) for i in items)
    cover, source = COVER_FALLBACK, "template"
    if client is not None and items:
        try:
            listing = "\n".join(f"- {i['summary']}" for i in items)
            r = client.messages.create(model=model, max_tokens=300, system=(
                "Write a warm, brief (3 sentences) cover note from a freelancer to a client explaining that the listed requests "
                "are outside the agreed scope and a short quote follows. Be friendly, never accusatory. Do NOT mention any "
                "amounts, hours or totals. Output only the note."),
                messages=[{"role": "user", "content": listing}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
            if text and not re.search(r"[₹$]|\d{3,}", text):
                cover, source = text, "live"
        except Exception:
            pass
    lines = [f"{n}. {i['summary']}: {i['hours']:g} h × ₹{rate:,.0f} = ₹{int(i['price']):,}" for n, i in enumerate(items, 1)]
    body = f"{cover}\n\n" + "\n".join(lines) + f"\n\nTotal: ₹{total:,}\nNothing starts until you confirm in writing."
    return {"message": body, "total": total, "items": items, "source": source}
