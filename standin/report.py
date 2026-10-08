"""Builds a single self-contained HTML report (frames embedded) for one run."""
from __future__ import annotations

import base64
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .analysis import Analysis, label
from .brain import CONCERNS
from .runner import RunResult

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]))

PALETTE = ["#2563eb", "#d9480f", "#2b8a3e", "#7048e8"]


def _survival_svg(an: Analysis, variant_names: dict[str, str]) -> str:
    W, H, L, R, T, B = 640, 240, 44, 16, 16, 34
    curves = an.survival
    n = max(len(c) for c in curves.values())
    x = lambda k: L + (W - L - R) * k / max(n - 1, 1)
    y = lambda v: T + (H - T - B) * (1 - v)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Share still browsing by step">']
    for v in (0, .25, .5, .75, 1):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" class="tick">{int(v * 100)}%</text>')
    for k in range(n):
        parts.append(f'<text x="{x(k):.1f}" y="{H - 12}" text-anchor="middle" class="tick">{k + 1}</text>')
    for i, (vid, c) in enumerate(curves.items()):
        pts = " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in enumerate(c))
        col = PALETTE[i % len(PALETTE)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.25"/>')
        parts.append(f'<circle cx="{x(0):.1f}" cy="{y(c[0]):.1f}" r="3" fill="{col}"/>')
    parts.append(f'<text x="{(W + L) / 2}" y="{H}" text-anchor="middle" class="tick">screen viewed (step)</text></svg>')
    return "".join(parts)


def build_report(result: RunResult, an: Analysis, fidelity: dict, out: Path) -> Path:
    exp = result.exp
    vnames = {v.id: v.label for v in exp.variants}
    snames = {s.id: s.name for s in exp.segments}
    colors = {v.id: PALETTE[i % len(PALETTE)] for i, v in enumerate(exp.variants)}

    frames: dict[str, str] = {}
    sessions = []
    for s in sorted(result.sessions, key=lambda s: s.id):
        steps = []
        for st in s.steps:
            key = Path(st.frame).name
            if key not in frames:
                frames[key] = "data:image/jpeg;base64," + base64.b64encode(Path(st.frame).read_bytes()).decode()
            d = st.model_dump(exclude={"frame"})
            d["frame"] = key
            d["attended_labels"] = [label(e) for e in st.attended]
            steps.append(d)
        sessions.append({"id": s.id, "segment": s.segment_id, "variant": s.variant_id, "outcome": s.outcome,
                         "concern": s.abandon_concern, "traits": s.traits, "steps": steps})

    # attention table: top elements by max exposure across cells
    elems = {}
    for v, segs in an.exposure.items():
        for g, d in segs.items():
            for e, share in d.items():
                elems[e] = max(elems.get(e, 0), share)
    top_elems = sorted(elems, key=lambda e: -elems[e])[:14]
    attention_rows = [{"el": label(e), "cells": [round(an.exposure[v][g].get(e, 0) * 100) for v in an.exposure for g in an.exposure[v]]}
                      for e in top_elems]
    attention_cols = [(vnames[v], snames[g]) for v in an.exposure for g in an.exposure[v]]

    html = _env.get_template("report.html.j2").render(
        exp=exp, result=result, an=an, fidelity=fidelity, vnames=vnames, snames=snames, colors=colors,
        survival_svg=_survival_svg(an, vnames), attention_rows=attention_rows, attention_cols=attention_cols,
        findings=[asdict(f) for f in an.findings], label=label, concerns=CONCERNS,
        created=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data_json=json.dumps({"sessions": sessions, "frames": frames, "vnames": vnames, "snames": snames,
                              "concerns": CONCERNS, "colors": colors}, ensure_ascii=False),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
