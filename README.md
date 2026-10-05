# ScopeGuard

An agent that checks each client request against a freelancer's signed scope, cites the exact clause,
tracks unbilled extra work, and drafts a change order the freelancer approves.

**Track:** Agentic AI

## How it works
For every client message the agent runs a bounded tool loop (max 6 steps):
`search_scope_clauses` → `get_revision_count` → `estimate_price` → `verify_citation` → `submit_verdict`.

Reliability layer (plain code, not the model):
- Scope clauses are parsed deterministically from the numbered text.
- Every in-scope or out-of-scope verdict must quote the clause word for word. Code re-checks the quote;
  a fabricated quote is rejected and the agent must retry.
- Revision rounds, the rate card and the running unbilled total are computed by code. If the client goes over the
  included revisions, code flips the verdict and cites the revision clause.
- If the model is down or cannot produce a verifiable answer, the message is marked **Review manually**. It is never guessed.
- Change-order totals are computed by code. The model only writes a cover note and is rejected if it states any amounts.
- Human in control: nothing is sent anywhere. The freelancer edits and copies the draft.

## Run
```
pip install -r requirements.txt
# Any OpenAI-compatible provider (default: Google Gemini via AI Studio):
export LLM_API_KEY=...
export LLM_MODEL=gemini-2.5-flash                                   # check the current model name in your provider's console
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai   # Groq: https://api.groq.com/openai/v1
# or use Anthropic instead: export ANTHROPIC_API_KEY=...  (model override: SCOPEGUARD_MODEL)
uvicorn main:app --port 8000
```
Open http://localhost:8000 and press "Load bakery demo", then "Check the chat".

## Tests and evaluation
```
python -m pytest tests          # pipeline tests with a scripted stand-in model (no API calls)
python eval_demo.py             # real model on the 14-message demo vs our own hand-written labels; records data/cache.json
```
The tests check the code around the model, not the model's judgement. `eval_demo.py` is a small self-labeled
check, not a benchmark. Cached replays are marked "replayed" in the UI.

## Not built (on purpose)
Accounts, WhatsApp/email integration, PDF contract parsing (paste text), payments, database, auto-sending.
