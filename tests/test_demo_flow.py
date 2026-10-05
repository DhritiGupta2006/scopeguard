"""End-to-end flow of the 14-message demo through the HTTP API with a scripted stand-in model that
returns the hand-labeled answers. Proves state (revision count, unbilled total) threads correctly
through the API. It says nothing about how the real model performs."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent, demo, main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

CLAUSE_FOR = {"4": "E-commerce, online payments or ordering", "2": "Up to 3 rounds of design revisions", "3": "One contact form that emails the owner"}


class StandIn:
    def __init__(self):
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        text = kw["messages"][0]["content"].split('"""')[1]
        idx = next(i for i, (_, t) in enumerate(demo.THREAD) if t == text)
        verdict, is_rev = demo.EXPECTED[idx]
        v = "out_of_scope" if idx == 10 else verdict
        v = "in_scope" if idx == 10 else v  # the model says in-scope revision; code must override
        clause = "3" if idx == 13 else ("2" if is_rev else "4")
        sub = {"verdict": v, "request_summary": text[:40], "reasoning": "test", "is_revision": is_rev,
               "clause_ids": [clause], "quote": CLAUSE_FOR[clause], "estimated_hours": 4}
        if v == "ambiguous":
            sub["clarifying_question"] = "What feels off?"
        if v == "no_request":
            sub = {"verdict": "no_request", "request_summary": "", "reasoning": "thanks"}
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="x", name="submit_verdict", input=sub)])


def test_full_demo_thread_through_api(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "CACHE_PATH", tmp_path / "c.json")
    main._client = StandIn()
    c = TestClient(main.app)
    d = c.get("/api/demo").json()
    msgs = [m for m in c.post("/api/parse", json={"scope": d["scope"], "thread": d["thread"]}).json()["messages"] if m["is_client"]]
    state, seen = agent.new_state(), {}
    for m in msgs:
        out = c.post("/api/message", json={"scope": d["scope"], "rate": 1500, "message": m["text"], "state": state}).json()
        state, seen[m["index"]] = out["state"], out["result"]
    for idx, (verdict, _) in demo.EXPECTED.items():
        assert seen[idx]["verdict"] == verdict, (idx, seen[idx]["verdict"])
    assert seen[10]["override_note"] and seen[10]["revision_count"] == 4  # 4th revision billed by code
    assert state["revisions_used"] == 4
    # out-of-scope: online ordering, blog, About copy, 4th revision, SEO -> 5 items x 4h x 1500
    assert len(state["items"]) == 5 and state["unbilled"] == 5 * 4 * 1500
    co = c.post("/api/change_order", json={"items": state["items"], "rate": 1500}).json()
    assert co["total"] == 30000 and f"Total: ₹30,000" in co["message"]
