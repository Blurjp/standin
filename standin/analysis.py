"""Turns sessions into a verdict, a funnel, attention stats, findings and sanity checks."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .brain import CONCERNS
from .models import AgentSession, Experiment

RECOMMENDATIONS = {
    "unclear_value": "Make screenshot 1 state the single outcome this segment wants, in their words. Cut feature-list captions.",
    "irrelevant": "Name the situation explicitly in the subtitle or first caption so the right people self-select in.",
    "generic": "Replace generic wellness language with a specific moment or problem only your app handles.",
    "low_trust": "Add a credible signal early: who built or reviewed it, and only claims you can back up.",
    "no_social_proof": "Prompt early users for ratings (SKStoreReviewController after a success moment); until then lean on concrete specifics.",
    "privacy": "Say how health data is handled in a screenshot, not only in the privacy card; reduce linked data categories if possible.",
    "price": "Make what is free explicit before the paywall question comes up; avoid subscription words above the fold.",
    "too_much_text": "Shorten captions to under 8 words; move detail to the description.",
}

ELEMENT_LABELS = {
    "name": "App name", "subtitle": "Subtitle", "get": "GET button", "sticky-get": "GET (sticky)",
    "iap": "In-App Purchases note", "rating": "Rating chip", "age": "Age chip", "category": "Category chip",
    "developer-chip": "Developer chip", "promo": "Promotional text", "description": "Description",
    "more": "'more' link", "developer": "Developer link", "ratings-summary": "Ratings summary",
    "whats-new": "What's New", "privacy-tracking": "Privacy: tracking", "privacy-linked": "Privacy: linked data",
    "privacy-not-linked": "Privacy: not linked", "privacy-not-collected": "Privacy: not collected",
}


def label(el: str) -> str:
    if el.startswith("shot-"):
        return f"Screenshot {int(el.split('-')[1]) + 1}"
    if el.startswith("review-"):
        return f"Review {int(el.split('-')[1]) + 1}"
    return ELEMENT_LABELS.get(el, el)


@dataclass
class Comparison:
    baseline: str
    challenger: str
    diff: float
    ci_low: float
    ci_high: float
    p_better: float
    verdict: str


@dataclass
class Finding:
    variant_id: str
    concern: str
    title: str
    share: float                      # weighted share of this variant's sessions that quit on it
    segments: dict[str, float]        # per-segment share
    element: str                      # most common thing on screen when they quit
    evidence: list[str]               # session ids
    recommendation: str
    vs_baseline: float | None = None  # change in share vs baseline variant (pp), for challengers


@dataclass
class Analysis:
    rates: dict[str, dict[str, float]]          # variant -> segment -> install rate
    overall: dict[str, float]                   # variant -> weighted install rate
    comparisons: list[Comparison]
    survival: dict[str, list[float]]            # variant -> cumulative share who left without installing by step k
    exposure: dict[str, dict[str, dict[str, float]]]   # variant -> segment -> element -> share attended
    actions: dict[str, dict[str, float]]        # variant -> action -> share of sessions that did it
    findings: list[Finding]
    checks: list[str] = field(default_factory=list)


def _weights(exp: Experiment) -> dict[str, float]:
    tot = sum(s.weight for s in exp.segments)
    return {s.id: s.weight / tot for s in exp.segments}


def analyze(exp: Experiment, sessions: list[AgentSession], n_boot: int = 2000, seed: int = 0) -> Analysis:
    w = _weights(exp)
    vids = [v.id for v in exp.variants if any(s.variant_id == v.id for s in sessions)]
    segs = [s.id for s in exp.segments]

    by_cell: dict[tuple[str, str], list[AgentSession]] = defaultdict(list)
    for s in sessions:
        by_cell[(s.variant_id, s.segment_id)].append(s)
    for k in by_cell:
        by_cell[k].sort(key=lambda s: s.id)

    rates = {v: {g: float(np.mean([s.outcome == "install" for s in by_cell[(v, g)]])) if by_cell[(v, g)] else 0.0
                 for g in segs} for v in vids}
    overall = {v: sum(w[g] * rates[v][g] for g in segs) for v in vids}

    # paired bootstrap over agents within each segment
    rng = np.random.default_rng(seed)
    comparisons = []
    base = vids[0]
    mats = {v: {g: np.array([s.outcome == "install" for s in by_cell[(v, g)]], dtype=float) for g in segs} for v in vids}
    for ch in vids[1:]:
        diffs = np.zeros(n_boot)
        for g in segs:
            a, b = mats[base][g], mats[ch][g]
            n = min(len(a), len(b))
            if n == 0:
                continue
            idx = rng.integers(0, n, size=(n_boot, n))
            diffs += w[g] * (b[:n][idx].mean(axis=1) - a[:n][idx].mean(axis=1))
        d = overall[ch] - overall[base]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_better = float((diffs > 0).mean() + 0.5 * (diffs == 0).mean())
        if lo > 0:
            verdict = f"{ch} beats {base}"
        elif hi < 0:
            verdict = f"{base} beats {ch}"
        else:
            verdict = "No clear winner at this sample size"
        comparisons.append(Comparison(base, ch, d, float(lo), float(hi), p_better, verdict))

    # cumulative give-up curve (left without installing by step k), weighted
    last = max((len(s.steps) for s in sessions), default=1)
    survival = {}
    for v in vids:
        curve = []
        for k in range(last):
            val = 0.0
            for g in segs:
                cell = by_cell[(v, g)]
                if cell:
                    val += w[g] * np.mean([s.outcome != "install" and (s.abandon_step or 0) <= k for s in cell])
            curve.append(round(val, 4))
        survival[v] = curve

    exposure = {}
    actions = {}
    for v in vids:
        exposure[v] = {}
        act = Counter()
        n_v = 0
        for g in segs:
            cell = by_cell[(v, g)]
            seen = Counter()
            for s in cell:
                seen.update({e for st in s.steps for e in st.attended})
                act.update({st.action for st in s.steps})
            n_v += len(cell)
            exposure[v][g] = {e: c / len(cell) for e, c in seen.items()} if cell else {}
        actions[v] = {a: c / max(n_v, 1) for a, c in act.items()}

    findings = []
    for v in vids:
        quits = [s for s in sessions if s.variant_id == v and s.outcome != "install"]
        groups: dict[str, list[AgentSession]] = defaultdict(list)
        for s in quits:
            groups[s.abandon_concern or "unclear_value"].append(s)
        for concern, group in groups.items():
            seg_share = {g: len([s for s in group if s.segment_id == g]) / max(len(by_cell[(v, g)]), 1) for g in segs}
            share = sum(w[g] * seg_share[g] for g in segs)
            if share < 0.03:
                continue
            last_seen = Counter(e for s in group for e in (s.steps[-1].attended[:2] if s.steps else []))
            element = last_seen.most_common(1)[0][0] if last_seen else "-"
            evidence = [s.id for s in sorted(group, key=lambda s: (-(w[s.segment_id]), len(s.steps)))[:4]]
            findings.append(Finding(v, concern, CONCERNS.get(concern, concern), share,
                                    {g: round(x, 3) for g, x in seg_share.items() if x > 0},
                                    element, evidence, RECOMMENDATIONS.get(concern, "")))
    base_share = {f.concern: f.share for f in findings if f.variant_id == base}
    for f in findings:
        if f.variant_id != base:
            f.vs_baseline = f.share - base_share.get(f.concern, 0.0)
    findings.sort(key=lambda f: (f.variant_id != base, -f.share))

    checks = []
    for v in vids:
        rs = [rates[v][g] for g in segs]
        steps = [np.mean([len(s.steps) for s in by_cell[(v, g)]]) for g in segs if by_cell[(v, g)]]
        if len(segs) > 1 and max(rs) - min(rs) < 0.05 and (max(steps) - min(steps)) < 0.6:
            checks.append(f"Segments behave almost identically on {v}: install rates within 5 pp and similar "
                          "browsing depth. Either the page doesn't differentiate them or the segment traits are too close.")
    for c in comparisons:
        width = c.ci_high - c.ci_low
        if width > 0.15:
            checks.append(f"{c.challenger} vs {c.baseline}: the 95% interval is {width * 100:.0f} pp wide. "
                          "Raise n_per_cell to narrow it.")
    all_rates = [overall[v] for v in vids]
    if max(all_rates) < 0.03:
        checks.append("Almost nobody installs on any variant; results are dominated by noise. Check calibration.")
    if min(all_rates) > 0.95:
        checks.append("Almost everybody installs on every variant; the simulation is too generous. Check calibration.")

    return Analysis(rates, overall, comparisons, survival, exposure, actions, findings, checks)
