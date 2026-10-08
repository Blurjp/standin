"""Renders a store-page variant in headless Chromium and returns what is on screen in a given state.

States are discrete (scroll page, screenshot carousel position, description expanded), so every
(variant, state) pair is rendered once and cached; hundreds of agents share the same frames.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.async_api import Browser, Page, async_playwright

from .models import Element, Experiment, Variant

VW, VH = 390, 844
SCROLL_STEP = 600  # px per "scroll_down" action (~70% of a screen)

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]))


@dataclass(frozen=True)
class ScreenState:
    scroll: int = 0
    shot: int = 0
    expanded: bool = False

    def key(self, variant_id: str) -> str:
        return f"{variant_id}|s{self.scroll}|c{self.shot}|e{int(self.expanded)}"


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def render_html(exp: Experiment, variant: Variant) -> str:
    p = variant.page
    icon = _data_uri(exp.resolve(p.icon)) if p.icon else None
    shots = [{"src": _data_uri(exp.resolve(s.image)), "caption": s.caption} for s in p.screenshots]
    return _env.get_template("store_page.html.j2").render(p=p, icon=icon, shots=shots)


class StoreRenderer:
    def __init__(self, exp: Experiment, frames_dir: Path):
        self.exp = exp
        self.frames_dir = frames_dir
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._pw = None
        self._browser: Browser | None = None
        self._pages: dict[str, Page] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, tuple[str, list[Element]]] = {}
        self.max_scroll: dict[str, int] = {}
        self.n_shots: dict[str, int] = {}

    async def __aenter__(self) -> "StoreRenderer":
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch()
        for v in self.exp.variants:
            page = await self._browser.new_page(viewport={"width": VW, "height": VH}, device_scale_factor=1)
            await page.set_content(render_html(self.exp, v), wait_until="load")
            height = await page.evaluate("document.body.scrollHeight")
            self.max_scroll[v.id] = max(0, math.ceil((height - 420 - VH) / SCROLL_STEP))
            self.n_shots[v.id] = len(v.page.screenshots)
            self._pages[v.id] = page
            self._locks[v.id] = asyncio.Lock()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def observe(self, variant_id: str, state: ScreenState) -> tuple[str, list[Element]]:
        key = state.key(variant_id)
        if key in self._cache:
            return self._cache[key]
        async with self._locks[variant_id]:
            if key in self._cache:
                return self._cache[key]
            page = self._pages[variant_id]
            await page.evaluate("([a,b,c,d]) => window.__applyState(a,b,c,d)",
                                [state.scroll, state.shot, state.expanded, SCROLL_STEP])
            raw = await page.evaluate(f"window.__extract({VW},{VH})")
            name = hashlib.sha1(key.encode()).hexdigest()[:12] + ".jpg"
            path = self.frames_dir / name
            await page.screenshot(path=str(path), type="jpeg", quality=60)
            elements = [Element(**r) for r in raw]
            self._cache[key] = (str(path), elements)
            return self._cache[key]
