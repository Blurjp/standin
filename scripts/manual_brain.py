"""Step through agents with a human (or any offline model) acting as the brain.

Re-runs the chosen agents from the start using recorded answers; stops at the first step with no
answer and prints exactly what the Claude brain would be sent (persona, what was noticed, the frame).
Add the answer to the answers file and run again. When every journey finishes, prints outcomes next to
the heuristic brain's outcome for the same agents, and the estimated API cost per journey.

    python scripts/manual_brain.py examples/vilia/experiment.yaml answers.json --agents newmom:0 partner:0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standin.agent import run_agent  # noqa: E402
from standin.brain import PRICES, HeuristicBrain, NeedAppraisal, ScriptedBrain  # noqa: E402
from standin.models import Experiment  # noqa: E402
from standin.render import StoreRenderer  # noqa: E402
from standin.runner import make_context  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("answers")
    ap.add_argument("--agents", nargs="+", required=True, help="segment_id:index, e.g. newmom:0")
    ap.add_argument("--frames", default="runs/_manual/frames")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    a = ap.parse_args()

    exp = Experiment.load(a.experiment)
    answers_path = Path(a.answers)
    answers = json.loads(answers_path.read_text()) if answers_path.exists() else {}
    seg_index = {s.id: i for i, s in enumerate(exp.segments)}
    brain = ScriptedBrain(answers, model=a.model)
    heuristic = HeuristicBrain()
    rows = []

    async with StoreRenderer(exp, Path(a.frames)) as rend:
        for spec in a.agents:
            seg_id, idx = spec.split(":")
            for v in exp.variants:
                ctx = make_context(exp, seg_index[seg_id], int(idx), v)
                try:
                    s = await run_agent(ctx.session_id, "manual", v.id, ctx, rend, brain, exp.params, exp.journey.max_steps)
                except NeedAppraisal as need:
                    print(f"NEED {need.key}")
                    print(f"FRAME {need.frame}")
                    print("NOTICED " + json.dumps([e.id for e in need.attended]))
                    print("BOXES " + json.dumps([[round(e.x), round(e.y), round(e.w), round(e.h)] for e in need.attended]))
                    print("---- prompt ----")
                    print(need.request["messages"][0]["content"][1]["text"])
                    return 3
                hctx = make_context(exp, seg_index[seg_id], int(idx), v)
                h = await run_agent(hctx.session_id, "manual-h", v.id, hctx, rend, heuristic, exp.params, exp.journey.max_steps)
                rows.append((s, h))

    p_in, p_out = PRICES.get(a.model, (2.0, 10.0))
    print(f"{'session':18} {'claude-as-brain':24} {'heuristic':24}")
    for s, h in rows:
        fmt = lambda x: f"{x.outcome} @ step {len(x.steps)}" + (f" ({x.abandon_concern})" if x.outcome != "install" else "")
        print(f"{s.id:18} {fmt(s):24} {fmt(h):24}")
    reqs = brain.requests
    t_in = sum(r["input_tokens"] for r in reqs) / len(reqs)
    t_out = sum(r["output_tokens"] for r in reqs) / len(reqs)
    steps = sum(len(s.steps) for s, _ in rows) / len(rows)
    print(f"\n{len(reqs)} appraisals, {steps:.2f} steps per journey, ~{t_in:.0f} input / ~{t_out:.0f} output tokens per step")
    for model, (pi, po) in PRICES.items():
        per_step = (t_in * pi + t_out * po) / 1e6
        print(f"  {model:18} ${per_step:.5f}/step  ${per_step * steps:.4f}/journey")
    json.dump({"rows": [{"id": s.id, "claude": s.model_dump(exclude={"steps"}), "heuristic": h.model_dump(exclude={"steps"}),
                         "steps": [st.model_dump() for st in s.steps]} for s, h in rows],
               "tokens_per_step": [t_in, t_out], "steps_per_journey": steps},
              open(answers_path.with_suffix(".result.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
