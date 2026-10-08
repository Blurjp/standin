# StandIn — simulated users for App Store product pages

MVP of the design doc: before you change your App Store screenshots or copy, run a few hundred
simulated visitors from your real audience segments through both versions and get a paired
comparison, the reasons people leave, and a replay of every journey.

```bash
pip install -r requirements.txt
python -m playwright install chromium        # skip if Chromium is already available
python examples/vilia/make_assets.py         # placeholder screenshots for the example
python -m standin run examples/vilia/experiment.yaml
# -> runs/<run-id>/report.html
```

## What it does

```
experiment.yaml ─► renderer (Playwright, App Store page clone) ─► frames + on-screen elements
                                                                      │
segments ─► sample 7 traits per agent ─► perceive (attention budget) ─┤
                                         appraise (brain) ◄───────────┘
                                         decide: p_install = σ(k·(utility − bar))
                                                 p_leave   = base·patience_decay·(1−value)·friction
                                         explore: swipe / scroll / expand
sessions ─► analysis (paired bootstrap, findings, attention) ─► report.html + standin.db
real funnel ─► calibrate (fit base_hazard, install_bar) ─► fidelity badge
past changes ─► backtest (winner hit-rate) ─► fidelity badge
```

| Module | Role |
| --- | --- |
| `render.py` | Renders each variant as an App Store product page in Chromium; returns the frame and every element on screen for a state (scroll, carousel position, description expanded). States are cached, so 300 agents share ~30 renders. |
| `perception.py` | Salience per element; each agent notices only `2 + 7·attention` of them, small print rarely, especially at low brightness. |
| `brain.py` | `HeuristicBrain` (offline, deterministic, free) and `ClaudeBrain` (sends the frame + noticed elements, gets structured beliefs back via a forced tool call). The brain updates beliefs; it never decides to quit. |
| `agent.py` | Install / leave / explore decision per step with fixed formulas, so LLM politeness cannot inflate conversion. |
| `runner.py` | Segments × variants × N, concurrent. **Paired design:** agent *i* has the same traits and random stream in every variant. |
| `analysis.py` | Segment-weighted install rates, paired bootstrap 95% interval, give-up curve, attention table, findings clustered by exit reason, sanity checks (segments too alike, interval too wide, all-or-nothing rates). |
| `calibrate.py` | Calibration grid fit (regularized; refuses below 10 installs / 300 views), backtest against past real A/B results, and the fidelity badge every report carries. |
| `store.py` | SQLite with the doc's tables (runs, sessions, steps, findings, calibrations, backtests). Swap for Postgres when hosted. |
| `report.py` | Single self-contained HTML: verdict, per-segment table, give-up curve, findings with evidence links, attention heatmap, step-by-step replay with attended-element overlays. |

## Commands

```bash
python -m standin run EXP.yaml [--n 100] [--brain claude] [--uncalibrated]
python -m standin calibrate EXP.yaml --variant A --page-views 4200 --installs 610
python -m standin backtest backtests.yaml     # see format in calibrate.backtest()
python -m standin runs
```

`--brain claude` needs `ANTHROPIC_API_KEY`. Model: `STANDIN_MODEL` (default `claude-sonnet-5-5`, ~$0.009 per
journey; `claude-haiku-5-5` ~$0.0004). Prices per model live in `brain.PRICES`; override with
`STANDIN_PRICE_IN_PER_MTOK` / `STANDIN_PRICE_OUT_PER_MTOK`. Auth and unknown-model errors abort the run;
transient errors fall back to the heuristic for that step and are counted.

Backtest spec:

```yaml
changes:
  - name: "Screenshots v2, Mar 2026"
    experiment: storage_cleaner_v2.yaml      # variant 0 = before, variant 1 = after
    actual:
      A: {page_views: 4200, installs: 610}
      B: {page_views: 3900, installs: 702}
```

## Read this before trusting a number

- **The heuristic brain scores value by matching each segment's `needs` phrases.** A variant whose
  captions use those words wins almost by construction. It is a cheap baseline and a test harness,
  not evidence. Use `--brain claude` for real appraisals, and write `needs` from reviews and
  support emails, not from the variant you hope wins.
- **Absolute install rates are meaningless until calibrated.** Report the difference, and only after
  `calibrate` succeeds. The tool refuses to calibrate on too little data and says so on the badge.
- **The Vilia example is placeholder content**: screenshots, captions, subtitle, description and
  privacy label are invented. Replace them with the real page. Variant B's privacy claims are only
  allowed on the page if they are true.
- Not built yet (doc phase 2): simulator/onboarding journeys, paywall, Web, App Store Connect API
  import, hosted UI, per-agent memory across journeys.

## Tests

```bash
python -m pytest -q
```

## Testing the brain without an API key

`scripts/manual_brain.py` replays chosen agents with recorded appraisals and stops at the first step that has
none, printing exactly what the Claude brain would be sent. Answer it with `scripts/answer.py`, re-run, repeat.
It ends with outcomes next to the heuristic brain's and an estimated cost per journey per model.
Results of the first such test: `docs/brain-test-2026-10-08.md`.
