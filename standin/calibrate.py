"""Calibration (fit behavior params to a real funnel) and backtesting (score past predictions).

Calibration fits two global parameters - base_hazard and install_bar - so that the simulated
current page converts like the real one. With one observed rate the two are not separately
identifiable, so the fit is regularized toward the defaults. Reports quote the challenger's
difference from the calibrated baseline, which is far less sensitive to calibration error than
any absolute rate.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import yaml

from .analysis import analyze
from .models import Experiment, SimParams
from .render import StoreRenderer
from .runner import run_experiment
from .store import Store

MIN_INSTALLS, MIN_VIEWS = 10, 300


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


async def calibrate(exp: Experiment, variant_id: str, page_views: int, installs: int, store: Store,
                    out_dir: Path, source: str = "app_store_connect", n_per_cell: int = 40,
                    force: bool = False) -> dict:
    observed = installs / page_views if page_views else 0.0
    lo, hi = wilson(installs, page_views)
    obs = {"page_views": page_views, "installs": installs, "rate": observed, "ci95": [lo, hi]}
    if not force and (installs < MIN_INSTALLS or page_views < MIN_VIEWS):
        cid = store.save_calibration(exp.app, variant_id, source, obs, exp.params.model_dump(), -1,
                                     "insufficient_data")
        return {"id": cid, "status": "insufficient_data", "observed": obs, "fitted": exp.params.model_dump(),
                "message": (f"{installs} install{'s' if installs != 1 else ''} from {page_views} product page views: the real rate is somewhere "
                            f"between {lo:.1%} and {hi:.1%}. Calibrating to that would be fitting noise. "
                            f"Need at least {MIN_INSTALLS} installs and {MIN_VIEWS} page views.")}

    var_obs = max(observed * (1 - observed) / page_views, 1e-5)
    grid = [(bh, bar) for bh in (0.08, 0.12, 0.18, 0.26, 0.36) for bar in (0.34, 0.41, 0.48, 0.55, 0.62)]
    best = None
    work = out_dir / "_calibration"
    async with StoreRenderer(exp, work / "frames") as rend:
        for bh, bar in grid:
            params = exp.params.model_copy(update={"base_hazard": bh, "install_bar": bar})
            res = await run_experiment(exp, work, params=params, n_per_cell=n_per_cell,
                                       variant_ids=[variant_id], renderer=rend, run_id=f"cal-{bh}-{bar}")
            sim = analyze(exp, res.sessions, n_boot=10).overall[variant_id]
            var_sim = max(sim * (1 - sim) / (n_per_cell * len(exp.segments)), 1e-5)
            loss = (sim - observed) ** 2 / (var_obs + var_sim) + 0.5 * ((bh - 0.18) / 0.1) ** 2 + 0.5 * ((bar - 0.48) / 0.1) ** 2
            if best is None or loss < best[0]:
                best = (loss, bh, bar, sim)
    _, bh, bar, sim = best
    fitted = exp.params.model_copy(update={"base_hazard": bh, "install_bar": bar}).model_dump()
    fit_error = abs(sim - observed)
    status = "ok" if lo <= sim <= hi else "poor_fit"
    cid = store.save_calibration(exp.app, variant_id, source, obs, fitted, fit_error, status)
    return {"id": cid, "status": status, "observed": obs, "fitted": fitted, "simulated_rate": sim,
            "message": f"Simulated {sim:.1%} vs observed {observed:.1%} (95% CI {lo:.1%}-{hi:.1%})."}


def calibrated_params(store: Store, exp: Experiment) -> tuple[SimParams, Optional[int]]:
    cal = store.latest_calibration(exp.app)
    if cal and cal["status"] in ("ok", "poor_fit"):
        return SimParams(**cal["fitted"]), cal["id"]
    return exp.params, None


def two_prop_winner(a: dict, b: dict, a_id: str, b_id: str) -> tuple[str, float, float]:
    pa, pb = a["installs"] / a["page_views"], b["installs"] / b["page_views"]
    p = (a["installs"] + b["installs"]) / (a["page_views"] + b["page_views"])
    se = math.sqrt(max(p * (1 - p) * (1 / a["page_views"] + 1 / b["page_views"]), 1e-12))
    z = (pb - pa) / se
    winner = b_id if z > 1.96 else a_id if z < -1.96 else "none"
    return winner, pb - pa, z


async def backtest(spec_path: Path, store: Store, out_dir: Path, n_per_cell: Optional[int] = None) -> list[dict]:
    spec = yaml.safe_load(Path(spec_path).read_text())
    results = []
    for ch in spec["changes"]:
        exp = Experiment.load(Path(spec_path).parent / ch["experiment"])
        a_id, b_id = exp.variants[0].id, exp.variants[1].id
        res = await run_experiment(exp, out_dir, n_per_cell=n_per_cell)
        an = analyze(exp, res.sessions)
        c = an.comparisons[0]
        predicted = b_id if c.ci_low > 0 else a_id if c.ci_high < 0 else "none"
        actual, actual_diff, z = two_prop_winner(ch["actual"][a_id], ch["actual"][b_id], a_id, b_id)
        hit = predicted == actual
        direction_hit = (c.diff > 0) == (actual_diff > 0) if actual != "none" else None
        data = {"sim_diff": c.diff, "sim_ci": [c.ci_low, c.ci_high], "actual_diff": actual_diff, "z": z,
                "run_id": res.run_id}
        store.save_backtest(exp.app, ch["name"], predicted, actual, hit, direction_hit, data)
        results.append({"change": ch["name"], "predicted": predicted, "actual": actual, "hit": hit,
                        "direction_hit": direction_hit, **data})
    return results


def fidelity(store: Store, app: str) -> dict:
    """What the report is allowed to claim. Shown as a badge on every report."""
    bt = store.backtest_summary(app)
    cal = store.latest_calibration(app)
    if bt["n"] >= 3:
        return {"level": "backtested", "text": f"Backtested: {bt['hits']}/{bt['n']} past winners called correctly"
                + (f", direction right {bt['direction_hits']}/{bt['direction_n']}" if bt["direction_n"] else "")}
    if cal and cal["status"] == "ok":
        o = cal["observed"]
        return {"level": "calibrated",
                "text": f"Calibrated to {o['installs']} installs / {o['page_views']} page views; not yet backtested"}
    if cal and cal["status"] == "insufficient_data":
        o = cal["observed"]
        return {"level": "uncalibrated",
                "text": f"Uncalibrated - only {o['installs']} install{'s' if o['installs'] != 1 else ''} / {o['page_views']} views of real data. Directional only."}
    return {"level": "uncalibrated", "text": "Uncalibrated - category priors only. Directional only."}
