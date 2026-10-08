"""Attention model: an agent only takes in part of what is on screen.

Low-attention agents notice the few most salient things near the top; small grey text is
usually missed. This is the main lever that stops every agent from 'reading every word'.
"""
from __future__ import annotations

import math

import numpy as np

from .models import Element
from .render import VH, VW

KIND_WEIGHT = {
    "screenshot": 1.0, "button": 0.85, "title": 0.85, "subtitle": 0.65, "rating": 0.6,
    "review": 0.5, "privacy": 0.5, "promo": 0.45, "text": 0.4, "link": 0.25, "small": 0.25, "nav": 0.0,
}
SMALL_FONT_PX = 13
READABLE_FRACTION = 0.8   # a screenshot cut off at the screen edge can't be read


def visible_fraction(el: Element) -> float:
    vis_w = max(0.0, min(el.x + el.w, VW) - max(el.x, 0.0))
    vis_h = max(0.0, min(el.y + el.h, VH) - max(el.y, 0.0))
    return (vis_w * vis_h) / max(el.w * el.h, 1.0)


def salience(el: Element) -> float:
    kind = KIND_WEIGHT.get(el.kind, 0.3)
    vis_w = max(0.0, min(el.x + el.w, VW) - max(el.x, 0.0))
    vis_h = max(0.0, min(el.y + el.h, VH) - max(el.y, 0.0))
    yc = max(el.y, 0.0) + vis_h / 2
    position = 0.55 + 0.45 * (1 - yc / VH)              # top of screen is read first
    size = min(1.0, 0.35 + math.sqrt(max(vis_w * vis_h, 1)) / 260)
    if el.kind == "screenshot":
        size *= vis_w / max(el.w, 1)                     # a half-visible screenshot draws less of the eye
    font = 0.55 if el.font_px < SMALL_FONT_PX else 1.0
    return round(kind * position * size * font, 4)


def attend(visible: list[Element], traits: dict[str, float], situation: list[str],
           rng: np.random.Generator, already_seen: set[str]) -> list[Element]:
    """Pick the elements this agent actually notices on this screen."""
    att = traits["attention_budget"]
    budget = 2 + round(att * 7)
    scored = sorted((e.model_copy(update={"salience": salience(e)}) for e in visible),
                    key=lambda e: e.salience, reverse=True)
    picked: list[Element] = []
    for rank, e in enumerate(scored):
        if e.salience <= 0:
            continue
        if e.kind == "screenshot" and visible_fraction(e) < READABLE_FRACTION:
            e = e.model_copy(update={"text": "(screenshot partly off screen, caption cut off)"})
        p = 0.25 + 0.55 * att + 0.45 * e.salience
        if rank == 0 and e.id not in already_seen:
            p = 1.0                      # everyone takes in the single most prominent new thing
        if e.font_px < SMALL_FONT_PX:
            p -= 0.35 if att < 0.55 else 0.1
            if "low_brightness" in situation:
                p -= 0.2
        if e.id in already_seen:
            p -= 0.25                    # re-reading the same thing is less likely
        if rng.random() < min(max(p, 0.02), 0.97):
            picked.append(e)
        if len(picked) >= budget:
            break
    return picked
