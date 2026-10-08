"""One simulated person browsing one store-page variant.

Per step: perceive (attention-limited) -> appraise (brain updates beliefs) -> decide.
Decisions are NOT left to the LLM:
  p_install = sigmoid(k * (utility - bar))
  p_abandon = base_hazard * patience_decay(step) * (1 - value) * friction_coef
Otherwise the agent explores (swipe screenshots, scroll, expand description), choosing what it is
curious about given its unresolved concerns.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .brain import AgentContext, Beliefs
from .models import AgentSession, Element, SimParams, Step
from .perception import attend
from .render import ScreenState, StoreRenderer


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def install_probability(b: Beliefs, traits: dict[str, float], p: SimParams) -> float:
    trust_need = 0.2 + 0.55 * traits["trust_threshold"]
    trust_gate = sigmoid((b.trust - trust_need) * 9)
    utility = b.value * (0.6 + 0.4 * b.comprehension) * (0.5 + 0.5 * trust_gate) - 0.4 * b.friction
    bar = p.install_bar - 0.18 * (traits["intent"] - 0.5)
    return sigmoid(p.install_sharpness * (utility - bar))


def abandon_hazard(step: int, b: Beliefs, traits: dict[str, float], p: SimParams) -> float:
    patience_decay = 1 + step * (1 - traits["patience"]) * 0.55
    trust_need = 0.2 + 0.55 * traits["trust_threshold"]
    friction_coef = 1 + 1.3 * b.friction + 0.9 * (1 - b.comprehension) + 2.0 * max(0.0, trust_need - b.trust)
    return float(min(max(p.base_hazard * patience_decay * (1 - b.value) * friction_coef, 0.0), 0.95))


def explore_options(state: ScreenState, visible: list[Element], b: Beliefs, ctx: AgentContext,
                    max_scroll: int, n_shots: int, curiosity: float, concerns: list[str]) -> dict[str, float]:
    t = ctx.traits
    opts: dict[str, float] = {}
    if state.scroll == 0 and state.shot < max(n_shots - 2, 0):
        opts["swipe_screenshots"] = 0.5 + 0.9 * (1 - b.comprehension) + 0.5 * ("unclear_value" in concerns)
    if state.scroll < max_scroll:
        want_proof = b.trust < 0.2 + 0.55 * t["trust_threshold"]
        want_privacy = t["privacy_sensitivity"] > 0.6 and not b.seen_privacy
        opts["scroll_down"] = 0.35 + 0.3 * t["patience"] + 0.7 * want_proof + 0.6 * want_privacy
    if not state.expanded and any(e.id == "more" for e in visible) and t["attention_budget"] > 0.3:
        opts["expand_description"] = (0.3 + 0.7 * (1 - b.comprehension)) * (0.4 if "too_much_text" in concerns else 1.0)
    return {k: v * (0.3 + curiosity) for k, v in opts.items()}


def apply(state: ScreenState, action: str) -> ScreenState:
    if action == "swipe_screenshots":
        return ScreenState(state.scroll, state.shot + 1, state.expanded)
    if action == "scroll_down":
        return ScreenState(state.scroll + 1, state.shot, state.expanded)
    if action == "expand_description":
        return ScreenState(state.scroll, state.shot, True)
    return state


async def run_agent(session_id: str, run_id: str, variant_id: str, ctx: AgentContext,
                    renderer: StoreRenderer, brain, params: SimParams, max_steps: int) -> AgentSession:
    rng: np.random.Generator = ctx.rng
    t = ctx.traits
    b = Beliefs()
    state = ScreenState()
    steps: list[Step] = []
    concern_log: Counter[str] = Counter()
    cost = 0.0
    outcome, abandon_step, abandon_concern = "stuck", None, None

    for idx in range(max_steps):
        frame, visible = await renderer.observe(variant_id, state)
        attended = attend(visible, t, ctx.segment.situation, rng, b.seen)
        a, c = await brain.appraise(ctx, attended, b, frame)
        cost += c
        b.value, b.trust, b.comprehension, b.friction = a.value, a.trust, a.comprehension, a.friction
        b.thoughts.append(a.thought)
        concern_log.update(a.concerns)

        p_inst = install_probability(b, t, params)
        p_ab = abandon_hazard(idx, b, t, params)
        if rng.random() < p_inst:
            action = "tap_get"
        elif rng.random() < p_ab:
            action = "leave"
        else:
            opts = explore_options(state, visible, b, ctx, renderer.max_scroll[variant_id],
                                   renderer.n_shots[variant_id], a.curiosity, a.concerns)
            if opts:
                keys = list(opts)
                w = np.array([opts[k] for k in keys])
                action = keys[int(rng.choice(len(keys), p=w / w.sum()))]
            else:
                action = "leave"   # nothing left worth looking at

        words = sum(len(e.text.split()) for e in attended)
        ms_per_word = 260 * (1.3 - 0.6 * t["tech_savvy"])
        uncertainty = 1 - abs(p_inst - 0.5) * 2
        hesitation = int(500 + words * ms_per_word * (0.6 + 0.8 * uncertainty))

        steps.append(Step(
            idx=idx, state_key=state.key(variant_id), frame=frame,
            attended=[e.id for e in attended],
            attended_boxes=[(round(e.x), round(e.y), round(e.w), round(e.h)) for e in attended],
            action=action, hesitation_ms=hesitation, thought=a.thought,
            value=round(b.value, 3), trust=round(b.trust, 3), comprehension=round(b.comprehension, 3),
            friction=round(b.friction, 3), p_install=round(p_inst, 3), p_abandon=round(p_ab, 3),
            concerns=a.concerns, emotion=a.emotion,
        ))

        if action == "tap_get":
            outcome = "install"
            break
        if action == "leave":
            outcome, abandon_step = "abandon", idx
            recent = a.concerns or [c for c, _ in concern_log.most_common(1)]
            abandon_concern = recent[0] if recent else ("unclear_value" if b.value < 0.4 else "low_trust")
            break
        state = apply(state, action)

    if outcome == "stuck":
        abandon_step = len(steps) - 1
        abandon_concern = concern_log.most_common(1)[0][0] if concern_log else "unclear_value"

    return AgentSession(id=session_id, run_id=run_id, segment_id=ctx.segment.id, variant_id=variant_id,
                        traits={k: round(v, 3) for k, v in t.items()}, outcome=outcome,
                        abandon_step=abandon_step, abandon_concern=abandon_concern, steps=steps,
                        cost_usd=round(cost, 5))
