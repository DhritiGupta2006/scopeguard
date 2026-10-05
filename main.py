from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent
import demo

app = FastAPI(title="ScopeGuard")
STATIC = Path(__file__).parent / "static"
_client = agent.get_client()


class MessageIn(BaseModel):
    scope: str
    rate: float = Field(gt=0)
    message: str = Field(min_length=1, max_length=2000)
    state: dict = Field(default_factory=agent.new_state)


class ParseIn(BaseModel):
    scope: str
    thread: str


class OrderIn(BaseModel):
    items: list[dict]
    rate: float = Field(gt=0)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"llm_configured": _client is not None, "model": agent.MODEL}


@app.get("/api/demo")
def get_demo():
    return {"scope": demo.SCOPE, "rate": demo.RATE, "thread": demo.thread_text()}


def parse_thread(text: str) -> list[dict]:
    """'Client: ...' lines are analysed; any other 'Name: ...' line is the freelancer. No prefixes -> all client."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    prefixed = [l for l in lines if ":" in l[:20]]
    out = []
    for i, l in enumerate(lines):
        if len(prefixed) >= max(1, len(lines) // 2) and ":" in l[:20]:
            who, body = l.split(":", 1)
            out.append({"index": i, "sender": who.strip(), "text": body.strip(), "is_client": who.strip().lower() in ("client", "them", "customer")})
        else:
            out.append({"index": i, "sender": "Client", "text": l, "is_client": True})
    return out


@app.post("/api/parse")
def parse(body: ParseIn):
    return {"clauses": agent.parse_scope(body.scope), "messages": parse_thread(body.thread)}


@app.post("/api/message")
def analyze(body: MessageIn):
    state = {**agent.new_state(), **body.state}
    result, new_state = agent.analyze_message(_client, body.scope, body.rate, body.message, state)
    return {"result": result, "state": new_state}


@app.post("/api/change_order")
def change_order(body: OrderIn):
    return agent.build_change_order(_client, body.items, body.rate)
