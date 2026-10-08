"""Core data model. Mirrors the tables in the design doc (Project/Variant/Segment/Journey/Run/...)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

# The seven behavioral traits every agent carries (all in [0, 1]).
TRAITS = (
    "intent",              # how much they came wanting a solution
    "patience",            # tolerance for spending time before deciding
    "price_sensitivity",   # reaction to price / subscription / IAP signals
    "trust_threshold",     # how much evidence they need before installing
    "tech_savvy",          # reading speed, comfort with UI
    "attention_budget",    # how much of a screen they actually take in
    "privacy_sensitivity", # reaction to data collection signals
)


class TraitDist(BaseModel):
    mean: float
    sd: float = 0.12


class Segment(BaseModel):
    id: str
    name: str
    context: str = ""                       # situational description, fed to the brain
    weight: float = 1.0                     # share of real traffic
    traits: dict[str, TraitDist]
    needs: list[str] = Field(default_factory=list)     # phrases this segment is looking for
    turnoffs: list[str] = Field(default_factory=list)  # phrases that lower perceived value
    situation: list[Literal["low_brightness", "one_handed", "interrupted", "distracted"]] = Field(default_factory=list)

    @field_validator("traits")
    @classmethod
    def _all_traits(cls, v: dict[str, TraitDist]) -> dict[str, TraitDist]:
        missing = [t for t in TRAITS if t not in v]
        if missing:
            raise ValueError(f"segment missing traits: {missing}")
        return v

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        """Draw one agent's trait vector; situations shift it."""
        t = {k: float(np.clip(rng.normal(d.mean, d.sd), 0.02, 0.98)) for k, d in self.traits.items()}
        if "low_brightness" in self.situation:
            t["attention_budget"] = max(0.02, t["attention_budget"] - 0.12)
        if "distracted" in self.situation or "interrupted" in self.situation:
            t["attention_budget"] = max(0.02, t["attention_budget"] - 0.10)
            t["patience"] = max(0.02, t["patience"] - 0.10)
        return t


class Screenshot(BaseModel):
    image: str                 # path, relative to the experiment file
    caption: str               # the words printed on the screenshot (what an agent "reads")


class Review(BaseModel):
    title: str
    body: str
    stars: int = 5


class Privacy(BaseModel):
    tracking: list[str] = Field(default_factory=list)     # "Data Used to Track You"
    linked: list[str] = Field(default_factory=list)       # "Data Linked to You"
    not_linked: list[str] = Field(default_factory=list)   # "Data Not Linked to You"
    not_collected: bool = False                           # "Data Not Collected"


class StorePage(BaseModel):
    name: str
    subtitle: str = ""
    developer: str = ""
    icon: Optional[str] = None
    category: str = "Health & Fitness"
    age_rating: str = "4+"
    price: str = "Get"                     # "Get" or "$2.99"
    in_app_purchases: bool = False
    rating: Optional[float] = None
    rating_count: int = 0
    promo_text: str = ""
    screenshots: list[Screenshot]
    description: str = ""
    whats_new: str = ""
    reviews: list[Review] = Field(default_factory=list)
    privacy: Privacy = Field(default_factory=Privacy)


class Variant(BaseModel):
    id: str
    label: str
    note: str = ""
    page: StorePage


class Journey(BaseModel):
    template: Literal["store_page_install"] = "store_page_install"
    goal: str = "Decide whether to download the app"
    # words that tell a visitor "this is the kind of app I searched for" (category relevance),
    # independent of whether it matches their specific need
    search_terms: list[str] = Field(default_factory=list)
    max_steps: int = 12


class SimParams(BaseModel):
    """Global behavior parameters. Calibration fits base_hazard and install_bar."""
    base_hazard: float = 0.18
    install_bar: float = 0.48
    install_sharpness: float = 7.0


class Experiment(BaseModel):
    name: str
    app: str
    journey: Journey = Field(default_factory=Journey)
    segments: list[Segment]
    variants: list[Variant]
    n_per_cell: int = 50
    seed: int = 7
    brain: Literal["heuristic", "claude"] = "heuristic"
    concurrency: int = 8
    params: SimParams = Field(default_factory=SimParams)
    base_dir: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> "Experiment":
        import yaml
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        data["base_dir"] = path.parent.resolve()
        return cls.model_validate(data)

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.base_dir / p)


# ---------- runtime records ----------

ActionKind = Literal["look", "swipe_screenshots", "scroll_down", "expand_description", "tap_get", "leave"]


class Element(BaseModel):
    id: str
    kind: str
    text: str
    x: float
    y: float
    w: float
    h: float
    font_px: float
    salience: float = 0.0


class Observation(BaseModel):
    state_key: str
    frame: str                         # path to the rendered screen (jpg)
    visible: list[Element]
    attended: list[Element]            # what this agent actually took in


class Appraisal(BaseModel):
    value: float          # does this solve my problem (0..1)
    trust: float          # do I believe it (0..1)
    comprehension: float  # do I understand what it is (0..1)
    friction: float       # cost signals: price, account, effort (0..1)
    curiosity: float      # do I want to see more before deciding (0..1)
    concerns: list[str] = Field(default_factory=list)   # category codes, see analysis.CONCERNS
    thought: str = ""
    emotion: str = "neutral"


class Step(BaseModel):
    idx: int
    state_key: str
    frame: str
    attended: list[str]                    # element ids
    attended_boxes: list[tuple[float, float, float, float]]
    action: ActionKind
    hesitation_ms: int
    thought: str
    value: float
    trust: float
    comprehension: float
    friction: float
    p_install: float
    p_abandon: float
    concerns: list[str]
    emotion: str


class AgentSession(BaseModel):
    id: str
    run_id: str
    segment_id: str
    variant_id: str
    traits: dict[str, float]
    outcome: Literal["install", "abandon", "stuck"]
    abandon_step: Optional[int] = None
    abandon_concern: Optional[str] = None
    steps: list[Step]
    cost_usd: float = 0.0
