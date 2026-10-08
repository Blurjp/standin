"""Generates PLACEHOLDER screenshots and an icon for the Vilia example.

Replace assets/ with your real App Store screenshots (and their real captions in experiment.yaml)
before trusting any result. Usage: python examples/vilia/make_assets.py
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
OUT = HERE / "assets"

SHOTS = {
    "a": [
        ("Your postpartum companion", "Everything for your fourth trimester", ["Today", "Mood check-in", "Recovery tracker"]),
        ("Track your recovery journey", "Log symptoms, mood and milestones", ["Week 3", "Energy 2/5", "Sleep 4h"]),
        ("AI-powered personalized insights", "Smart guidance tailored to you", ["Insight", "You slept less", "Try a nap"]),
        ("Join a supportive community", "Connect with moms like you", ["Circles", "Night feeds", "C-section recovery"]),
        ("Daily wellness tips for new moms", "Self-care made simple", ["Tip of the day", "Hydrate", "Breathe"]),
    ],
    "b": [
        ("Answers at 3am, one-handed", "Know what's normal after birth in seconds", ["Is this normal?", "Bleeding after 2 weeks", "Normal / Call your doctor"]),
        ("Your health data stays on your phone", "No ads. No tracking. Never sold.", ["Private by default", "Stored on this device", "No account needed"]),
        ("Bleeding, sleep, breastfeeding", "What's normal, what needs a call", ["Warning signs", "Fever over 100.4°F", "Call now"]),
        ("For partners: how to help tonight", "Simple things that make a difference", ["Tonight", "Take the 2am feed", "Bring water"]),
        ("Second baby? Plans that fit a toddler too", "Specific help for round two", ["Sibling prep", "Nap overlap", "Your recovery"]),
    ],
}

TPL = """<html><body style="margin:0;width:430px;height:932px;font-family:-apple-system,Helvetica,Arial,sans-serif;
background:linear-gradient(170deg,{bg1},{bg2});display:flex;flex-direction:column;align-items:center;">
<div style="padding:70px 34px 0;text-align:center;color:#3b2b45">
  <div style="font-size:38px;font-weight:800;line-height:1.12;letter-spacing:-.5px">{title}</div>
  <div style="font-size:21px;margin-top:14px;color:#6b5877">{sub}</div></div>
<div style="margin-top:44px;width:300px;height:560px;border-radius:44px;background:#fff;box-shadow:0 18px 40px rgba(80,40,90,.18);padding:56px 22px 0">
  {cards}</div></body></html>"""
CARD = ('<div style="border-radius:18px;background:{c};padding:18px 16px;margin-bottom:14px;font-size:{fs}px;'
        'font-weight:{fw};color:#3b2b45">{t}</div>')
ICON = """<html><body style="margin:0;width:256px;height:256px;background:#fbf3f1;display:flex;align-items:center;justify-content:center">
<svg width="190" height="190" viewBox="0 0 100 100" fill="none" stroke-width="3.2" stroke-linecap="round">
<path d="M50 86 C20 66 12 46 22 32 C30 22 42 24 46 34" stroke="#d98a8a"/>
<path d="M50 86 C80 66 88 46 78 32 C70 22 58 24 54 34" stroke="#9b86b8"/>
<circle cx="36" cy="22" r="7" stroke="#d98a8a"/><circle cx="64" cy="20" r="7" stroke="#9b86b8"/>
<circle cx="50" cy="50" r="6" stroke="#e8b27a"/><path d="M42 64 C46 70 54 70 58 64" stroke="#e8b27a"/></svg></body></html>"""


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 430, "height": 932})
        palettes = {"a": ("#f7e9ee", "#e9e3f3"), "b": ("#fde8e1", "#efe4f7")}
        for v, shots in SHOTS.items():
            for i, (title, sub, cards) in enumerate(shots):
                body = "".join(CARD.format(c=["#f6eef3", "#f1edf8", "#fbf1ea"][j % 3], fs=22 if j == 0 else 18,
                                           fw=700 if j == 0 else 500, t=t) for j, t in enumerate(cards))
                await page.set_content(TPL.format(bg1=palettes[v][0], bg2=palettes[v][1], title=title, sub=sub, cards=body))
                await page.screenshot(path=str(OUT / f"{v}{i + 1}.png"))
        page2 = await b.new_page(viewport={"width": 256, "height": 256})
        await page2.set_content(ICON)
        await page2.screenshot(path=str(OUT / "icon.png"))
        await b.close()
    print(f"wrote {len(list(OUT.glob('*.png')))} files to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
