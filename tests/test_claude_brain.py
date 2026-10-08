"""ClaudeBrain against a fake Messages client: request shape, parsing, cost, fallback, fail-fast."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from standin.brain import (IMAGE_TOKENS, AgentContext, Beliefs, ClaudeBrain, NeedAppraisal, ScriptedBrain,
                           build_request, estimate_input_tokens)
from standin.models import Element, Experiment

EXP = Path(__file__).parents[1] / "examples" / "vilia" / "experiment.yaml"
FRAME = Path(__file__).parents[1] / "examples" / "vilia" / "assets" / "icon.png"
TRAITS = {"intent": .6, "patience": .5, "price_sensitivity": .5, "trust_threshold": .5,
          "tech_savvy": .5, "attention_budget": .5, "privacy_sensitivity": .8}
GOOD = {"value": 0.7, "trust": 1.4, "comprehension": 0.6, "friction": -0.2, "curiosity": 0.3,
        "concerns": ["privacy", "made_up_code"], "thought": "Okay, this is for me.", "emotion": "hopeful",
        "extra_field": "ignored"}


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.sent = reply, error, []
        self.messages = self

    async def create(self, **req):
        self.sent.append(req)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply, usage=SimpleNamespace(input_tokens=1700, output_tokens=180))


def tool_use(data):
    return [SimpleNamespace(type="tool_use", input=data)]


def ctx():
    exp = Experiment.load(EXP)
    return AgentContext(exp.segments[0], TRAITS, exp.variants[0].page, "find help", np.random.default_rng(0),
                        session_id="newmom-000-A")


def els():
    return [Element(id="privacy-linked", kind="privacy", text="Data Linked to You: Health & Fitness",
                    x=0, y=0, w=300, h=60, font_px=13)]


def test_request_is_well_formed_and_sized():
    req = build_request(ctx(), els(), Beliefs(), str(FRAME), "claude-sonnet-5-5")
    assert req["tool_choice"] == {"type": "tool", "name": "update_beliefs"}
    content = req["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[1]["type"] == "text"
    assert "Data Linked to You" in content[1]["text"] and "New mom" in content[1]["text"]
    assert IMAGE_TOKENS < estimate_input_tokens(req) < 3000


def test_parses_clips_filters_and_costs():
    client = FakeClient(reply=tool_use(GOOD))
    brain = ClaudeBrain(client=client)
    b = Beliefs()
    a, cost = asyncio.run(brain.appraise(ctx(), els(), b, str(FRAME)))
    assert a.trust == 1.0 and a.friction == 0.0           # clipped into [0, 1]
    assert a.concerns == ["privacy"]                       # unknown concern codes dropped
    assert b.seen_privacy and "privacy-linked" in b.seen
    assert cost == pytest.approx((1700 * 2 + 180 * 10) / 1e6)   # Sonnet 5.5 prices
    assert brain.fallbacks == 0 and brain.tokens_in == 1700


def test_transient_error_falls_back_and_is_counted():
    brain = ClaudeBrain(client=FakeClient(error=TimeoutError("slow")))
    a, cost = asyncio.run(brain.appraise(ctx(), els(), Beliefs(), str(FRAME)))
    assert a.thought.startswith("[heuristic fallback: TimeoutError]") and cost == 0.0
    assert brain.fallbacks == 1


def test_malformed_tool_output_falls_back():
    brain = ClaudeBrain(client=FakeClient(reply=[SimpleNamespace(type="text", text="I think...")]))
    a, _ = asyncio.run(brain.appraise(ctx(), els(), Beliefs(), str(FRAME)))
    assert "heuristic fallback" in a.thought and brain.fallbacks == 1


def test_auth_error_aborts_instead_of_silent_heuristic_run():
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    brain = ClaudeBrain(client=FakeClient(error=AuthenticationError("bad key")))
    with pytest.raises(AuthenticationError):
        asyncio.run(brain.appraise(ctx(), els(), Beliefs(), str(FRAME)))


def test_scripted_brain_replays_and_stops_at_gaps():
    brain = ScriptedBrain({"newmom-000-A:0": GOOD})
    b = Beliefs()
    a, _ = asyncio.run(brain.appraise(ctx(), els(), b, str(FRAME)))
    b.thoughts.append(a.thought)
    with pytest.raises(NeedAppraisal) as e:
        asyncio.run(brain.appraise(ctx(), els(), b, str(FRAME)))
    assert e.value.key == "newmom-000-A:1"
    assert brain.requests[0]["input_tokens"] > IMAGE_TOKENS
