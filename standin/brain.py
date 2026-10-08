"""The 'brain' turns what an agent noticed into updated beliefs (value, trust, comprehension, friction).

Two backends share one interface:
  * HeuristicBrain - deterministic, free, runs offline; used for tests, calibration sweeps and as a
    baseline. It only knows the words on screen and the segment's stated needs.
  * ClaudeBrain - sends the actual frame + the attended elements to Claude and asks for the same
    structured judgment. Falls back to the heuristic on any API error.

The brain never decides to quit. Abandonment comes from the hazard function in agent.py, so an
LLM's tendency to be endlessly patient cannot leak into the funnel numbers.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .models import Appraisal, Element, Segment, StorePage

CONCERNS = {
    "unclear_value": "Can't tell what it does for me",
    "irrelevant": "Not for my situation",
    "generic": "Feels generic / like every other app",
    "low_trust": "Not sure I can trust it",
    "no_social_proof": "No ratings or reviews yet",
    "privacy": "Worried about my health data",
    "price": "Looks like it will cost money",
    "too_much_text": "Too much to read right now",
}

CREDIBILITY = ("nurse", "doctor", "ob-gyn", "obgyn", "clinician", "midwife", "evidence-based",
               "reviewed by", "certified", "research-backed", "lactation consultant")
PRIVACY_GOOD = ("stays on your phone", "on your device", "on-device", "no tracking", "no ads",
                "never sold", "never share", "data not collected", "private by default", "end-to-end")
COST = ("subscription", "free trial", "premium", "per month", "/month", "/year", "unlock", "pro plan")
ACCOUNT = ("create an account", "sign up", "log in to", "account required")
JARGON = ("ai-powered", "holistic", "ecosystem", "platform", "leverage", "synergy", "journey",
          "empower", "seamless", "all-in-one")


@dataclass
class Beliefs:
    value: float = 0.0
    trust: float = 0.5
    comprehension: float = 0.4
    friction: float = 0.0
    seen: set[str] = field(default_factory=set)
    seen_privacy: bool = False
    relevant: bool = False          # saw that this is the kind of app they searched for
    thoughts: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    segment: Segment
    traits: dict[str, float]
    page: StorePage
    goal: str
    rng: np.random.Generator
    search_terms: list[str] = field(default_factory=list)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


def _hits(text: str, phrases) -> list[str]:
    return [p for p in phrases if p.lower() in text]


class HeuristicBrain:
    name = "heuristic"

    async def appraise(self, ctx: AgentContext, attended: list[Element], b: Beliefs, frame: str) -> tuple[Appraisal, float]:
        t, seg = ctx.traits, ctx.segment
        new = [e for e in attended if e.id not in b.seen]
        concerns: list[str] = []
        notes: list[tuple[float, str, str]] = []   # (weight, thought, emotion)

        # ---- value: did I see what I came for? ----
        need_score, best_need = 0.0, None
        turnoff_score = 0.0
        for e in new:
            txt = e.text.lower()
            hits = _hits(txt, seg.needs)
            if hits:
                need_score += (0.45 + e.salience) * len(hits)
                if best_need is None:
                    best_need = hits[0]
            turnoff_score += 0.5 * len(_hits(txt, seg.turnoffs))
        if not b.relevant and any(_hits(e.text.lower(), ctx.search_terms) for e in new):
            b.relevant = True
        screen_value = 1 - math.exp(-0.7 * need_score)
        screen_value *= 1 - min(turnoff_score, 0.7)
        if new:
            if screen_value > b.value:
                value = b.value + (screen_value - b.value) * 0.5
            else:
                value = b.value - (b.value - screen_value) * 0.12 - 0.12 * min(turnoff_score, 1)
        else:
            value = b.value
        floor = t["intent"] * (0.12 + (0.3 if b.relevant else 0.0))   # "right category" is worth something
        value = _clip(max(value, floor))
        if best_need:
            # positive notes only win the 'voice' when the overall picture is positive
            notes.append((need_score * value, f"'{best_need}' - that is exactly my situation.", "hopeful"))
        elif new and value < 0.35 and any(e.kind in ("screenshot", "subtitle", "title", "promo") for e in new):
            concerns.append("unclear_value")
            notes.append((0.6, "Nice pictures, but I still can't tell how this helps me.", "unsure"))
        if turnoff_score > 0:
            concerns.append("generic")
            notes.append((turnoff_score + 0.3, "This sounds like every other wellness app.", "bored"))

        # ---- comprehension: could I take it in? ----
        words = sum(len(e.text.split()) for e in new)
        budget = 18 + 120 * t["attention_budget"] * (0.5 + 0.5 * t["tech_savvy"])
        overload = max(0.0, words / budget - 1)
        jargon = sum(len(_hits(e.text.lower(), JARGON)) for e in new)
        # short, plain statements (headline captions, subtitle) tell people what the app is
        readable = sum(1 for e in new if e.kind in ("screenshot", "subtitle", "title", "promo")
                       and len(e.text.split()) <= 14 and not _hits(e.text.lower(), JARGON))
        comprehension = (b.comprehension + 0.08 * min(readable, 3) + 0.2 * min(need_score, 1.0)
                         - 0.2 * min(overload, 1.5) - min(0.04 * jargon, 0.12))
        comprehension = _clip(comprehension, 0.05, 0.98)
        if overload > 0.35:
            concerns.append("too_much_text")
            notes.append((overload, "Too much text, I can't read all this right now.", "overwhelmed"))

        # ---- trust ----
        trust = b.trust
        for e in new:
            txt = e.text.lower()
            if e.kind == "rating":
                if "not enough" in txt or txt.startswith("no "):
                    trust -= 0.14 * t["trust_threshold"]
                    concerns.append("no_social_proof")
                    notes.append((0.5 * t["trust_threshold"] + 0.2, "No ratings at all? Nobody has tried this yet.", "skeptical"))
                else:
                    m = re.search(r"(\d\.\d)", txt)
                    stars = float(m.group(1)) if m else 4.0
                    trust += 0.25 * (stars - 3.8)
            if e.kind == "review":
                trust += 0.07 if txt[:1] in "45" else -0.1
            cred = _hits(txt, CREDIBILITY)
            if cred:
                trust += min(0.08 * len(cred), 0.2)
                notes.append((0.5, f"Mentions {cred[0]} - that makes me trust it more.", "reassured"))
            ps = t["privacy_sensitivity"]
            if e.id == "privacy-tracking":
                trust -= 0.45 * ps
                concerns.append("privacy")
                notes.append((ps + 0.4, "They track me with my health data? No.", "uneasy"))
            elif e.id == "privacy-linked":
                sensitive = "health" in txt or "contact" in txt
                trust -= (0.28 if sensitive else 0.12) * ps
                concerns.append("privacy")
                notes.append((ps, "My health data is linked to my identity. Hmm.", "uneasy"))
            elif e.id == "privacy-not-collected" or _hits(txt, PRIVACY_GOOD):
                trust += 0.22 * ps
                notes.append((ps, "Good, my data stays private.", "reassured"))
        if any(e.kind == "privacy" or _hits(e.text.lower(), PRIVACY_GOOD) for e in new):
            b.seen_privacy = True
        trust = _clip(trust, 0.02, 0.98)
        if trust < t["trust_threshold"] - 0.15 and "low_trust" not in concerns and new:
            concerns.append("low_trust")

        # ---- friction ----
        friction = b.friction
        ps = t["price_sensitivity"]
        if ctx.page.price.lower() != "get" and not b.seen:
            friction += 0.25 + 0.45 * ps
            concerns.append("price")
        for e in new:
            txt = e.text.lower()
            if e.id == "iap":
                friction += 0.18 * ps
                concerns.append("price")
            n_cost = len(_hits(txt, COST))
            if n_cost:
                friction += 0.12 * ps * n_cost
                concerns.append("price")
                notes.append((ps * 0.8, "So it's a subscription. Probably paywalled.", "wary"))
            if _hits(txt, ACCOUNT):
                friction += 0.12
        friction = _clip(friction)

        curiosity = _clip(0.2 + 0.4 * t["intent"] + 0.3 * t["patience"] - 0.35 * abs(value - 0.55))

        if notes:
            notes.sort(key=lambda n: n[0], reverse=True)
            thought, emotion = notes[0][1], notes[0][2]
        elif not new:
            thought, emotion = "Nothing new here.", "bored"
        else:
            thought, emotion = "Okay, still looking.", "neutral"

        b.seen.update(e.id for e in attended)
        return Appraisal(value=value, trust=trust, comprehension=comprehension, friction=friction,
                         curiosity=curiosity, concerns=list(dict.fromkeys(concerns)),
                         thought=thought, emotion=emotion), 0.0


# ---------------------------------------------------------------------------

_TOOL = {
    "name": "update_beliefs",
    "description": "Report how this screen changed what you, the person described, believe about the app.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "0-1: how much this app would solve your problem, given everything seen so far"},
            "trust": {"type": "number", "description": "0-1: how much you believe it is credible and safe"},
            "comprehension": {"type": "number", "description": "0-1: how well you understand what the app is"},
            "friction": {"type": "number", "description": "0-1: perceived cost - price, subscription, account, effort"},
            "curiosity": {"type": "number", "description": "0-1: how much you want to look further before deciding"},
            "concerns": {"type": "array", "items": {"type": "string", "enum": list(CONCERNS)}},
            "thought": {"type": "string", "description": "one short first-person sentence, in your own voice"},
            "emotion": {"type": "string", "description": "one word"},
        },
        "required": ["value", "trust", "comprehension", "friction", "curiosity", "concerns", "thought", "emotion"],
    },
}


def _describe_traits(t: dict[str, float]) -> str:
    def lvl(x: float) -> str:
        return "very low" if x < 0.2 else "low" if x < 0.4 else "medium" if x < 0.6 else "high" if x < 0.8 else "very high"
    return ", ".join(f"{k.replace('_', ' ')}: {lvl(v)}" for k, v in t.items())


class ClaudeBrain:
    """Claude-backed appraisal. Model and prices are configurable via env vars."""
    name = "claude"

    def __init__(self) -> None:
        import anthropic
        self.client = anthropic.AsyncAnthropic()   # reads ANTHROPIC_API_KEY
        self.model = os.environ.get("STANDIN_MODEL", "claude-sonnet-5-5")
        self.price_in = float(os.environ.get("STANDIN_PRICE_IN_PER_MTOK", "3"))
        self.price_out = float(os.environ.get("STANDIN_PRICE_OUT_PER_MTOK", "15"))
        self.fallback = HeuristicBrain()

    async def appraise(self, ctx: AgentContext, attended: list[Element], b: Beliefs, frame: str) -> tuple[Appraisal, float]:
        seg = ctx.segment
        noticed = "\n".join(f"- [{e.kind}] {e.text[:300]}" for e in attended) or "- (nothing caught your eye)"
        prior = (f"value {b.value:.2f}, trust {b.trust:.2f}, comprehension {b.comprehension:.2f}, "
                 f"friction {b.friction:.2f}. Recent thoughts: {' | '.join(b.thoughts[-3:]) or 'none'}")
        system = (
            "You simulate one specific real person browsing an App Store product page on their phone. "
            "Stay in character. You only know what this person noticed; do not use anything else on the screenshot. "
            "Be as impatient, distracted or skeptical as this person actually is. Never be generous by default."
        )
        user_text = (
            f"Who you are: {seg.name}. {seg.context}\nSituation: {', '.join(seg.situation) or 'normal'}.\n"
            f"Your tendencies: {_describe_traits(ctx.traits)}.\nWhat you want: {ctx.goal}.\n"
            f"Your beliefs so far: {prior}\n\nOn this screen you noticed only:\n{noticed}\n\n"
            "Update your beliefs with the update_beliefs tool."
        )
        img = base64.b64encode(open(frame, "rb").read()).decode()
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=400, system=system, tools=[_TOOL],
                tool_choice={"type": "tool", "name": "update_beliefs"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
                    {"type": "text", "text": user_text}]}],
            )
            data = next(c.input for c in resp.content if c.type == "tool_use")
            cost = (resp.usage.input_tokens * self.price_in + resp.usage.output_tokens * self.price_out) / 1e6
            a = Appraisal(**{k: (_clip(v) if isinstance(v, (int, float)) else v) for k, v in data.items()})
            a.concerns = [c for c in a.concerns if c in CONCERNS]
            if any(e.kind == "privacy" for e in attended):
                b.seen_privacy = True
            b.seen.update(e.id for e in attended)
            return a, cost
        except Exception as err:  # network, quota, schema: keep the run alive, mark the step
            a, _ = await self.fallback.appraise(ctx, attended, b, frame)
            a.thought = f"[heuristic fallback: {type(err).__name__}] {a.thought}"
            return a, 0.0


def make_brain(kind: str):
    if kind == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("brain=claude needs ANTHROPIC_API_KEY; use brain=heuristic to run offline.")
        return ClaudeBrain()
    return HeuristicBrain()


def dumps(a: Appraisal) -> str:
    return json.dumps(a.model_dump(), ensure_ascii=False)
