"""Tests use a scripted stand-in for the model. They verify the pipeline around the model
(tools, citation gate, ledger, fallbacks), NOT the model's judgement. Judgement is measured
separately by eval_demo.py against the real API."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402
import demo  # noqa: E402

RATE = 1500


def tu(name, inp, id_="t"):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class Scripted:
    """Fake Anthropic client: each create() call returns the next scripted list of tool_use blocks."""
    def __init__(self, steps):
        self.steps, self.calls = list(steps), 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        step = self.steps.pop(0) if self.steps else self.steps_last
        if isinstance(step, Exception):
            raise step
        self.steps_last = step
        return SimpleNamespace(content=step)


QUOTE = "E-commerce, online payments or ordering"
OUT = {"verdict": "out_of_scope", "request_summary": "Online ordering with payments", "clause_ids": ["4"],
       "quote": QUOTE, "reasoning": "Excluded by clause 4.", "is_revision": False, "estimated_hours": 20}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "CACHE_PATH", tmp_path / "cache.json")


def test_parse_scope_and_limit():
    cl = agent.parse_scope(demo.SCOPE)
    assert [c["id"] for c in cl] == ["1", "2", "3", "4", "5", "6", "7"]
    assert agent.revision_limit(cl) == (3, "2")


def test_verify_citation():
    t = agent.ScopeTools(agent.parse_scope(demo.SCOPE), RATE, agent.new_state())
    assert t.verify_citation("4", QUOTE)["verified"]
    assert t.verify_citation("4", "e-commerce online payments or ordering")["verified"]  # case/punctuation tolerant
    assert not t.verify_citation("4", "mobile apps are excluded too")["verified"]
    assert not t.verify_citation("9", QUOTE)["verified"]
    assert not t.verify_citation("4", "online")["verified"]  # too short


def test_search_finds_exclusion():
    t = agent.ScopeTools(agent.parse_scope(demo.SCOPE), RATE, agent.new_state())
    ids = [c["id"] for c in t.search_scope_clauses("add online ordering with payments")["clauses"]]
    assert ids[0] == "4"


def test_happy_path_ledger_is_code_not_llm():
    fake = Scripted([
        [tu("search_scope_clauses", {"query": "online ordering payments"})],
        [tu("verify_citation", {"clause_id": "4", "quote": QUOTE})],
        [tu("estimate_price", {"hours": 20})],
        [tu("submit_verdict", OUT)],
    ])
    res, st = agent.analyze_message(fake, demo.SCOPE, RATE, "add online ordering", agent.new_state())
    assert res["verdict"] == "out_of_scope" and res["source"] == "live"
    assert res["price"] == 30000 and st["unbilled"] == 30000
    assert [s["tool"] for s in res["trace"]] == ["search_scope_clauses", "verify_citation", "estimate_price", "submit_verdict"]


def test_fabricated_quote_is_rejected_then_corrected():
    bad = dict(OUT, quote="Mobile apps are never included in any project")
    fake = Scripted([[tu("submit_verdict", bad)], [tu("submit_verdict", OUT)]])
    res, st = agent.analyze_message(fake, demo.SCOPE, RATE, "x", agent.new_state())
    assert fake.calls == 2 and res["verdict"] == "out_of_scope"
    assert "not word for word" in res["trace"][0]["output"]


def test_never_verifiable_becomes_needs_review_and_state_untouched():
    bad = dict(OUT, quote="totally invented clause text here")
    fake = Scripted([[tu("submit_verdict", bad)]])
    start = agent.new_state()
    res, st = agent.analyze_message(fake, demo.SCOPE, RATE, "x", start)
    assert res["verdict"] == "needs_review" and st == start
    assert fake.calls == agent.MAX_STEPS


def test_model_exception_becomes_needs_review():
    res, st = agent.analyze_message(Scripted([RuntimeError("boom")]), demo.SCOPE, RATE, "x", agent.new_state())
    assert res["verdict"] == "needs_review" and "RuntimeError" in res["reasoning"]


def test_no_client_becomes_needs_review():
    res, _ = agent.analyze_message(None, demo.SCOPE, RATE, "x", agent.new_state())
    assert res["verdict"] == "needs_review"


def test_ambiguous_requires_question():
    amb = {"verdict": "ambiguous", "request_summary": "more premium", "reasoning": "vague"}
    ok, _ = agent.ScopeTools(agent.parse_scope(demo.SCOPE), RATE, agent.new_state()).check_submission(amb)
    assert not ok
    ok, _ = agent.ScopeTools(agent.parse_scope(demo.SCOPE), RATE, agent.new_state()).check_submission(dict(amb, clarifying_question="What feels off?"))
    assert ok


def test_revision_limit_override():
    rev = {"verdict": "in_scope", "request_summary": "Move map lower", "clause_ids": ["2"],
           "quote": "Up to 3 rounds of design revisions", "reasoning": "design tweak", "is_revision": True, "estimated_hours": 0.5}
    state = {"revisions_used": 3, "unbilled": 0, "items": []}
    res, st = agent.apply_ledger(rev, demo.SCOPE, RATE, state)
    assert res["verdict"] == "out_of_scope" and res["revision_count"] == 4 and res["override_note"]
    assert res["price"] == 750 and st["unbilled"] == 750 and res["clause_ids"] == ["2"]
    # within the limit: counted, not billed
    res2, st2 = agent.apply_ledger(rev, demo.SCOPE, RATE, {"revisions_used": 1, "unbilled": 0, "items": []})
    assert res2["verdict"] == "in_scope" and res2["revision_count"] == 2 and st2["unbilled"] == 0


def test_ledger_does_not_mutate_input_state():
    state = agent.new_state()
    agent.apply_ledger(OUT, demo.SCOPE, RATE, state)
    assert state == agent.new_state()


def test_cache_replays_only_real_outputs():
    fake = Scripted([[tu("submit_verdict", OUT)]])
    live, _ = agent.analyze_message(fake, demo.SCOPE, RATE, "add online ordering", agent.new_state())
    replay, st = agent.analyze_message(None, demo.SCOPE, RATE, "add online ordering", agent.new_state())
    assert live["source"] == "live" and replay["source"] == "cached"
    assert replay["verdict"] == "out_of_scope" and replay["price"] == live["price"] == st["unbilled"]
    miss, _ = agent.analyze_message(None, demo.SCOPE, RATE, "a message never seen before", agent.new_state())
    assert miss["verdict"] == "needs_review"


def test_change_order_total_from_code_and_cover_note_guard():
    items = [{"summary": "Online ordering", "hours": 20, "price": 30000, "clause_ids": ["4"]},
             {"summary": "Blog", "hours": 6, "price": 9000, "clause_ids": ["4"]}]
    # model tries to smuggle in its own numbers -> rejected, template used
    leaky = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text", text="That will be ₹99999 total.")])))
    co = agent.build_change_order(leaky, items, RATE)
    assert co["total"] == 39000 and "Total: ₹39,000" in co["message"] and co["source"] == "template" and "99999" not in co["message"]
    good = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text", text="Hi! A few of these are new work, so here's a quick quote.")])))
    assert agent.build_change_order(good, items, RATE)["source"] == "live"
    assert agent.build_change_order(None, items, RATE)["source"] == "template"


def test_api_endpoints():
    from fastapi.testclient import TestClient
    import main
    main._client = None
    c = TestClient(main.app)
    assert c.get("/").status_code == 200
    d = c.get("/api/demo").json()
    p = c.post("/api/parse", json={"scope": d["scope"], "thread": d["thread"]}).json()
    assert len(p["clauses"]) == 7 and len(p["messages"]) == 14
    assert sum(m["is_client"] for m in p["messages"]) == 11
    r = c.post("/api/message", json={"scope": d["scope"], "rate": 1500, "message": "hello"}).json()
    assert r["result"]["verdict"] == "needs_review" and r["state"]["unbilled"] == 0
    assert c.post("/api/message", json={"scope": d["scope"], "rate": 0, "message": "x"}).status_code == 422
    assert c.post("/api/change_order", json={"items": [], "rate": 1500}).json()["total"] == 0


def test_openai_compat_adapter_roundtrip(monkeypatch):
    """Adapter turns OpenAI-style tool calls into the shape the agent loop expects, and sends tool results back."""
    import io, json as _json
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append(_json.loads(req.data))
        if len(sent) == 1:
            msg = {"content": None, "tool_calls": [{"id": "", "type": "function", "function": {"name": "search_scope_clauses", "arguments": _json.dumps({"query": "online ordering"})}}]}
        else:
            msg = {"content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "submit_verdict", "arguments": _json.dumps(OUT)}}]}
        return io.BytesIO(_json.dumps({"choices": [{"message": msg}]}).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    client = agent.OpenAICompatClient("k", "https://example.test/v1", "some-model")
    res, st = agent.analyze_message(client, demo.SCOPE, RATE, "add online ordering", agent.new_state())
    assert res["verdict"] == "out_of_scope" and st["unbilled"] == 30000
    assert sent[0]["tool_choice"] == "required" and sent[0]["model"] == "some-model" and sent[0]["messages"][0]["role"] == "system"
    roles = [m["role"] for m in sent[1]["messages"]]
    assert roles[-2:] == ["assistant", "tool"] and sent[1]["messages"][-1]["tool_call_id"] == "call_0"


def test_adapter_http_error_becomes_needs_review(monkeypatch):
    import io, urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(agent.urllib.request, "urlopen", boom)
    res, _ = agent.analyze_message(agent.OpenAICompatClient("k", "https://x.test", "m"), demo.SCOPE, RATE, "x", agent.new_state())
    assert res["verdict"] == "needs_review" and "401" in res["reasoning"]


def test_adapter_retries_failed_tool_generation(monkeypatch):
    """A 400 tool_use_failed is retried after a short backoff; the next good response is used."""
    import io, json as _json, urllib.error
    calls, sleeps = [], []

    def flaky(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b'{"error":{"code":"tool_use_failed"}}'))
        msg = {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "submit_verdict", "arguments": _json.dumps(OUT)}}]}
        return io.BytesIO(_json.dumps({"choices": [{"message": msg}]}).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)
    client = agent.OpenAICompatClient("k", "https://x.test", "m")
    res, _ = agent.analyze_message(client, demo.SCOPE, RATE, "add online ordering", agent.new_state(), use_cache=False)
    assert res["verdict"] == "out_of_scope" and len(calls) == 2 and sleeps == [1]
