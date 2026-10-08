"""Command line: python -m standin {run,calibrate,backtest,runs} ..."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .analysis import analyze
from .calibrate import backtest, calibrate, calibrated_params, fidelity
from .models import Experiment
from .report import build_report
from .runner import run_experiment
from .store import Store


def _progress(done: int, total: int) -> None:
    if done == total or done % max(total // 20, 1) == 0:
        sys.stderr.write(f"\r  {done}/{total} journeys")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")


async def cmd_run(a) -> None:
    exp = Experiment.load(a.experiment)
    if a.brain:
        exp.brain = a.brain
    store = Store(a.db)
    params, cal_id = (exp.params, None) if a.uncalibrated else calibrated_params(store, exp)
    print(f"Running {exp.name}: {len(exp.segments)} segments x {len(exp.variants)} variants x "
          f"{a.n or exp.n_per_cell} agents, brain={exp.brain}" + (f", calibration #{cal_id}" if cal_id else ", uncalibrated"))
    res = await run_experiment(exp, Path(a.out), params=params, n_per_cell=a.n, progress=_progress)
    an = analyze(exp, res.sessions)
    fid = fidelity(store, exp.app)
    report = build_report(res, an, fid, res.run_dir / "report.html")
    store.save_run(res, an, report, cal_id)
    for v, r in an.overall.items():
        print(f"  {v}: {r:.1%} simulated installs")
    for c in an.comparisons:
        print(f"  {c.challenger} - {c.baseline}: {c.diff * 100:+.1f} pp (95% {c.ci_low * 100:+.1f} to {c.ci_high * 100:+.1f}); {c.verdict}")
    for f in an.findings[:5]:
        print(f"  [{f.variant_id}] {f.share:.0%} leave: {f.title}")
    for c in an.checks:
        print(f"  ! {c}")
    print(f"  fidelity: {fid['text']}")
    print(f"Report: {report}")


async def cmd_calibrate(a) -> None:
    exp = Experiment.load(a.experiment)
    store = Store(a.db)
    out = await calibrate(exp, a.variant, a.page_views, a.installs, store, Path(a.out), force=a.force)
    print(json.dumps({k: out[k] for k in ("id", "status", "message")}, indent=2))
    if out["status"] != "insufficient_data":
        print("fitted:", {k: out["fitted"][k] for k in ("base_hazard", "install_bar")})


async def cmd_backtest(a) -> None:
    store = Store(a.db)
    for r in await backtest(Path(a.spec), store, Path(a.out), n_per_cell=a.n):
        mark = "HIT " if r["hit"] else "MISS"
        print(f"  {mark} {r['change']}: predicted {r['predicted']}, actual {r['actual']} "
              f"(sim {r['sim_diff'] * 100:+.1f} pp, real {r['actual_diff'] * 100:+.1f} pp)")
    print(store.backtest_summary())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="standin", description="Simulate users on an App Store page before you ship it.")
    p.add_argument("--db", default="standin.db")
    p.add_argument("--out", default="runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment and write a report")
    r.add_argument("experiment")
    r.add_argument("--n", type=int, help="agents per segment x variant (overrides the file)")
    r.add_argument("--brain", choices=["heuristic", "claude"])
    r.add_argument("--uncalibrated", action="store_true", help="ignore stored calibration")

    c = sub.add_parser("calibrate", help="fit behavior params to your real product-page conversion")
    c.add_argument("experiment")
    c.add_argument("--variant", required=True, help="id of the variant that is live now")
    c.add_argument("--page-views", type=int, required=True, help="App Store Connect: product page views")
    c.add_argument("--installs", type=int, required=True, help="App Store Connect: first-time downloads from those views")
    c.add_argument("--force", action="store_true", help="fit even with too little data")

    b = sub.add_parser("backtest", help="score the simulator on past changes with known real results")
    b.add_argument("spec")
    b.add_argument("--n", type=int)

    sub.add_parser("runs", help="list stored runs")

    a = p.parse_args(argv)
    if a.cmd == "runs":
        for row in Store(a.db).runs():
            print(f"{row['id']}  {row['experiment']}  {row['brain']}  n={row['n_per_cell']}  {row['report_path']}")
        return
    asyncio.run({"run": cmd_run, "calibrate": cmd_calibrate, "backtest": cmd_backtest}[a.cmd](a))


if __name__ == "__main__":
    main()
