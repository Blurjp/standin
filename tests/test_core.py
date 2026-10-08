import asyncio
from pathlib import Path

import numpy as np
import pytest

from standin.agent import abandon_hazard, install_probability
from standin.analysis import analyze
from standin.brain import AgentContext, Beliefs, HeuristicBrain
from standin.calibrate import calibrate, fidelity, two_prop_winner, wilson
from standin.models import Element, Experiment, SimParams
from standin.perception import attend, salience
from standin.runner import run_experiment
from standin.store import Store

EXP = Path(__file__).parents[1] / "examples" / "vilia" / "experiment.yaml"
TRAITS = {"intent": .6, "patience": .5, "price_sensitivity": .5, "trust_threshold": .5,
          "tech_savvy": .5, "attention_budget": .5, "privacy_sensitivity": .8}


@pytest.fixture(scope="module")
def exp():
    return Experiment.load(EXP)


def el(id, kind="text", text="", y=100, h=40, w=300, font=15):
    return Element(id=id, kind=kind, text=text, x=10, y=y, w=w, h=h, font_px=font)


def test_trait_sampling_bounded_and_situation_shifts(exp):
    seg = next(s for s in exp.segments if s.id == "newmom")
    rng = np.random.default_rng(0)
    draws = [seg.sample(rng) for _ in range(500)]
    assert all(0.02 <= v <= 0.98 for d in draws for v in d.values())
    # low_brightness + interrupted pull attention below the configured mean
    assert np.mean([d["attention_budget"] for d in draws]) < seg.traits["attention_budget"].mean


def test_hazard_rises_with_steps_and_falls_with_value():
    p = SimParams()
    lo, hi = Beliefs(value=.8, trust=.6, comprehension=.8), Beliefs(value=.1, trust=.6, comprehension=.8)
    assert abandon_hazard(5, hi, TRAITS, p) > abandon_hazard(0, hi, TRAITS, p)
    assert abandon_hazard(0, hi, TRAITS, p) > abandon_hazard(0, lo, TRAITS, p)


def test_install_needs_trust_and_value():
    p = SimParams()
    good = Beliefs(value=.8, trust=.8, comprehension=.8)
    distrust = Beliefs(value=.8, trust=.1, comprehension=.8)
    useless = Beliefs(value=.1, trust=.8, comprehension=.8)
    assert install_probability(good, TRAITS, p) > install_probability(distrust, TRAITS, p) > install_probability(useless, TRAITS, p)


def test_low_attention_sees_less_and_skips_small_print():
    els = [el("shot-0", "screenshot", "a", y=200, h=490, w=228), el("name", "title", "b", y=60),
           el("iap", "small", "In-App Purchases", y=120, font=10), el("description", "text", "c " * 30, y=700)]
    rng = np.random.default_rng(1)
    low = [len(attend(els, {**TRAITS, "attention_budget": .05}, ["low_brightness"], rng, set())) for _ in range(300)]
    high = [len(attend(els, {**TRAITS, "attention_budget": .95}, [], rng, set())) for _ in range(300)]
    assert np.mean(low) < np.mean(high)
    small_seen = sum("iap" in [e.id for e in attend(els, {**TRAITS, "attention_budget": .1}, ["low_brightness"], rng, set())]
                     for _ in range(300))
    assert small_seen < 60
    assert salience(els[0]) > salience(els[2])


def test_heuristic_brain_reacts_to_privacy(exp):
    seg = exp.segments[0]
    page = exp.variants[0].page
    brain = HeuristicBrain()

    def run(e):
        ctx = AgentContext(seg, TRAITS, page, "x", np.random.default_rng(0))
        a, _ = asyncio.run(brain.appraise(ctx, [e], Beliefs(), "f.jpg"))
        return a
    bad = run(el("privacy-linked", "privacy", "Data Linked to You: Health & Fitness, Contact Info"))
    good = run(el("privacy-not-collected", "privacy", "Data Not Collected"))
    assert good.trust > 0.5 > bad.trust
    assert "privacy" in bad.concerns


def test_stats_helpers():
    lo, hi = wilson(1, 53)
    assert lo < 1 / 53 < hi and hi > 0.08     # one install in 53 views tells you very little
    w, d, z = two_prop_winner({"page_views": 4000, "installs": 600}, {"page_views": 4000, "installs": 720}, "A", "B")
    assert w == "B" and d > 0
    w, *_ = two_prop_winner({"page_views": 100, "installs": 10}, {"page_views": 100, "installs": 12}, "A", "B")
    assert w == "none"


def test_end_to_end_small_run(exp, tmp_path):
    res = asyncio.run(run_experiment(exp, tmp_path, n_per_cell=6))
    assert len(res.sessions) == 6 * len(exp.segments) * len(exp.variants)
    assert all(s.steps and s.outcome in ("install", "abandon", "stuck") for s in res.sessions)
    # paired design: agent i has identical traits in both variants
    a = {s.id[:-2]: s.traits for s in res.sessions if s.variant_id == "A"}
    b = {s.id[:-2]: s.traits for s in res.sessions if s.variant_id == "B"}
    assert a == b
    an = analyze(exp, res.sessions, n_boot=200)
    c = an.comparisons[0]
    assert c.ci_low <= c.diff <= c.ci_high
    assert abs(sum(an.overall.values())) <= 2


def test_deterministic_given_seed(exp, tmp_path):
    r1 = asyncio.run(run_experiment(exp, tmp_path / "1", n_per_cell=4))
    r2 = asyncio.run(run_experiment(exp, tmp_path / "2", n_per_cell=4))
    key = lambda r: sorted((s.id, s.outcome, len(s.steps)) for s in r.sessions)
    assert key(r1) == key(r2)


def test_calibration_refuses_tiny_data_and_fidelity_says_so(exp, tmp_path):
    store = Store(tmp_path / "t.db")
    out = asyncio.run(calibrate(exp, "A", page_views=53, installs=1, store=store, out_dir=tmp_path))
    assert out["status"] == "insufficient_data"
    assert fidelity(store, exp.app)["level"] == "uncalibrated"


def test_calibration_moves_toward_observed_rate(exp, tmp_path):
    store = Store(tmp_path / "t.db")
    out = asyncio.run(calibrate(exp, "A", page_views=2000, installs=200, store=store, out_dir=tmp_path, n_per_cell=20))
    assert out["status"] in ("ok", "poor_fit")
    default_rate = analyze(exp, asyncio.run(run_experiment(exp, tmp_path, n_per_cell=20, variant_ids=["A"])).sessions,
                           n_boot=10).overall["A"]
    assert abs(out["simulated_rate"] - 0.10) < abs(default_rate - 0.10)
