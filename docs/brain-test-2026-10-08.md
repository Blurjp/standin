# Claude-brain test, 2026-10-08

**Verdict:** the Claude brain is affordable on Sonnet 5.5 (~$0.008 per journey, target ≤ $0.015) and its
appraisals read like people, but this test cannot tell us whether it predicts real behavior. The biggest
threat to fidelity is not the brain; it is personas written by the same person who wrote the winning variant.

## What was tested

No `ANTHROPIC_API_KEY` was available, so the live API was not called. Instead:

1. **Code path** — `ClaudeBrain` against a fake Messages client (`tests/test_claude_brain.py`): request
   shape, tool-output parsing, clipping, unknown concern codes, cost math, fallback on timeouts and malformed
   output, and abort (no silent fallback) on auth / permission / unknown-model errors.
2. **Request size** — the exact request `build_request()` produces for each step, measured offline.
3. **Appraisal quality** — Claude (in the build session) acted as the brain through `ScriptedBrain`, on
   agent 0 of each segment × both variants (6 journeys, 12 appraisals), seeing only the persona, beliefs
   so far, and what the agent noticed. Reproduce with:
   `python scripts/manual_brain.py examples/vilia/experiment.yaml manual_test/answers.json --agents newmom:0 partner:0 secondtime:0`

## Cost

Per step: ~1,528 input tokens (474 forced-tool overhead + 434 image tokens for a 390×844 frame + ~620 text)
and ~90–150 output tokens. Journeys averaged 2.0 steps here, 2.2 across a full heuristic run.

| Model | $/step | $/journey (2.2 steps) | Price source |
| --- | --- | --- | --- |
| claude-sonnet-5-5 | ~$0.0039 | ~$0.009 | $2 / $10 per MTok |
| claude-haiku-5-5 | ~$0.0002 | ~$0.0004 | $0.10 / $0.50 per MTok |
| claude-opus-5-5 | ~$0.0078 | ~$0.017 | $4 / $20 per MTok |

Prices: [platform.claude.com pricing](https://platform.claude.com/docs/en/about-claude/pricing), image
tokens: ⌈w/28⌉ × ⌈h/28⌉ ([vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)).
Sonnet 5.5's forced-tool overhead isn't published; Sonnet 5's 474 was used. Token counts are estimates
(±20%); confirm with `count_tokens` or `usage` on the first real run. The $49 Indie tier (1,000 journeys)
costs ~$9 on Sonnet, ~$0.40 on Haiku.

## Outcomes, same agents, two brains

| Session | Claude as brain | Heuristic brain |
| --- | --- | --- |
| newmom-000-A | install, step 1 | install, step 1 |
| newmom-000-B | install, step 1 | install, step 1 |
| partner-000-A | leave at step 3: generic | leave at step 3: can't tell what it does |
| partner-000-B | install, step 1 | install, step 1 |
| secondtime-000-A | leave at step 3: generic | leave at step 3: generic |
| secondtime-000-B | leave at step 3: not for my situation | leave at step 3: not sure I can trust it |

Same outcome on 6/6 (they share the random stream and the decision formulas, so this is expected), different
exit reason on 2 of 3 exits. The Claude-brain reasons are the more believable ones: the second-time mom on
B left because the page is built for a first baby, which the heuristic has no way to express.

## Problems found and fixed

1. **Agents read captions they couldn't see.** A screenshot cut off at the screen edge gave its full caption.
   Now a screenshot under 80% visible reads "(caption cut off)". Effect on the Vilia run: B's simulated lead
   fell from +42 to +33 pp.
2. **A very-low-attention agent could skip the biggest thing on screen** by random draw. The single most
   salient new element is now always noticed.
3. **The brain prompt cut text at 300 characters,** dropping the privacy sentence at the end of B's
   description. Raised to 900.
4. **Auth or model errors used to fall back to the heuristic on every step,** producing a heuristic run
   labeled "claude". They now abort; transient errors still fall back and are counted.
5. **Price defaults were wrong** ($3/$15). Now per model from the pricing page.

## Problems found, not fixed

1. **Personas leak the answer.** The new-mom persona says "3am" and "wants to know if it's normal"; variant
   B's first caption says "Answers at 3am… know what's normal". Both were written in the same session.
   Any brain will favor B. Fix: generate segments, needs and contexts from App Store reviews before writing
   variants (next build).
2. **The appraiser was not blind.** Claude wrote variant B and the personas, then appraised them. Treat
   the quality check as "plausible", not as evidence. The live API run is still needed, ideally with a
   model call that never sees the variant notes.
3. **Words and actions can disagree.** The second-time mom on A thought "Next." at step 2 but kept browsing
   for another step, because the leave hazard ignores the brain's `curiosity`. Candidate fix: scale the
   hazard by (1.5 − curiosity). Not changed yet; tune it only against backtest data.
4. **Screenshots 4–5 are almost never seen** (heuristic run, n = 180 per variant): screenshot 3 is noticed
   by 19–25% of visitors, 4 by 3–9%, 5 by 0–3%. B's partner and second-baby pitches sit in screenshots 4
   and 5, so they do nothing for those segments. Product suggestion: move segment-specific pitches into
   the subtitle or captions 1–2. This matches common App Store experience but is still model output.

## Next

Run `python -m standin run examples/vilia/experiment.yaml --brain claude --n 20` with a real key and check:
`usage` tokens vs this estimate, fallback count (should be 0), and whether 20 replays read like people.
