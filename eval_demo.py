"""Run the bakery demo thread through the REAL model, compare with our hand-written labels,
and record the outputs into data/cache.json (used as a disclosed fallback replay).

    export ANTHROPIC_API_KEY=...
    python eval_demo.py

Labels are our own judgement on a 14-message thread we wrote ourselves. Report the number as
exactly that: a small self-labeled check, not a benchmark."""
import sys

import agent
import demo


def main() -> int:
    client = agent.get_client()
    if client is None:
        print("Set LLM_API_KEY (or ANTHROPIC_API_KEY) first.")
        return 1
    state, rows, ok_v, ok_rev = agent.new_state(), [], 0, 0
    for idx, (who, text) in enumerate(demo.THREAD):
        if who != "Client":
            continue
        res, state = agent.analyze_message(client, demo.SCOPE, demo.RATE, text, state)
        exp_v, exp_r = demo.EXPECTED[idx]
        v_ok = res["verdict"] == exp_v
        r_ok = res["is_revision"] == exp_r or exp_v in ("ambiguous", "no_request")
        ok_v += v_ok
        ok_rev += r_ok
        rows.append((idx, exp_v, res["verdict"], "OK " if v_ok else "MISS", res["source"], text[:48]))
    print(f"{'#':>2} {'expected':<13}{'got':<14}{'':<5}{'src':<7}message")
    for r in rows:
        print(f"{r[0]:>2} {r[1]:<13}{r[2]:<14}{r[3]:<5}{r[4]:<7}{r[5]}")
    n = len(rows)
    print(f"\nVerdict agreement with our labels: {ok_v}/{n}   Revision flag agreement: {ok_rev}/{n}")
    print(f"Final unbilled total: ₹{state['unbilled']:,}   Revision rounds used: {state['revisions_used']}")
    if any(r[4] != "live" for r in rows):
        print("WARNING: some rows were not live model calls (cache or failure). Re-run with a working key.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
