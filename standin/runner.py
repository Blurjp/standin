"""Runs an experiment: segments x variants x N agents, concurrently.

Paired design: agent i of a segment has the same traits and the same random stream in every
variant, so the variant comparison is within-person and needs far fewer agents for a given
confidence.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .agent import run_agent
from .brain import AgentContext, make_brain
from .models import AgentSession, Experiment, SimParams, Variant
from .render import StoreRenderer


@dataclass
class RunResult:
    run_id: str
    exp: Experiment
    params: SimParams
    sessions: list[AgentSession]
    run_dir: Path
    seconds: float
    brain: str
    n_per_cell: int

    @property
    def cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.sessions), 4)


def new_run_id(exp: Experiment) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(f"{exp.name}{time.time_ns()}".encode()).hexdigest()[:5]
    return f"run-{stamp}-{h}"


def session_id(seg_id: str, i: int, variant_id: str) -> str:
    return f"{seg_id}-{i:03d}-{variant_id}"


def make_context(exp: Experiment, seg_index: int, i: int, variant: Variant) -> AgentContext:
    """Agent i of a segment: same traits and same random stream in every variant (paired design).
    Traits are drawn sequentially per segment, so agent i depends only on the first i+1 draws."""
    seg = exp.segments[seg_index]
    trait_rng = np.random.default_rng([exp.seed, seg_index])
    traits = [seg.sample(trait_rng) for _ in range(i + 1)][-1]
    return AgentContext(segment=seg, traits=traits, page=variant.page, goal=exp.journey.goal,
                        search_terms=exp.journey.search_terms,
                        rng=np.random.default_rng([exp.seed, 7919, seg_index, i]),
                        session_id=session_id(seg.id, i, variant.id))


async def run_experiment(exp: Experiment, out_dir: Path, *, params: Optional[SimParams] = None,
                         n_per_cell: Optional[int] = None, variant_ids: Optional[list[str]] = None,
                         brain=None, run_id: Optional[str] = None,
                         progress: Optional[Callable[[int, int], None]] = None,
                         renderer: Optional[StoreRenderer] = None) -> RunResult:
    params = params or exp.params
    n = n_per_cell or exp.n_per_cell
    run_id = run_id or new_run_id(exp)
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    brain = brain or make_brain(exp.brain)
    variants = [v for v in exp.variants if variant_ids is None or v.id in variant_ids]
    sem = asyncio.Semaphore(exp.concurrency if brain.name == "claude" else 64)
    t0 = time.time()

    async def _go(rend: StoreRenderer) -> list[AgentSession]:
        jobs = [(si, i, v) for si in range(len(exp.segments)) for i in range(n) for v in variants]
        done = 0
        total = len(jobs)

        async def one(si, i, v):
            nonlocal done
            async with sem:
                ctx = make_context(exp, si, i, v)
                s = await run_agent(ctx.session_id, run_id, v.id, ctx, rend, brain, params, exp.journey.max_steps)
            done += 1
            if progress:
                progress(done, total)
            return s

        return list(await asyncio.gather(*(one(*j) for j in jobs)))

    if renderer is not None:
        sessions = await _go(renderer)
    else:
        async with StoreRenderer(exp, run_dir / "frames") as rend:
            sessions = await _go(rend)

    return RunResult(run_id=run_id, exp=exp, params=params, sessions=sessions, run_dir=run_dir,
                     seconds=round(time.time() - t0, 1), brain=brain.name, n_per_cell=n)
