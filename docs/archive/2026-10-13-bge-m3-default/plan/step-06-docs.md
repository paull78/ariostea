# Step 06 · README, write-up and artifact

**Milestone:** 3 · **Needs:** 05 · **Covers:** AC-06, AC-07

## Goal
README "Where it stands" and a fourth learning-journey trial describe bge-m3 as the default; `docs/retrieval-tuning.md` records the 256t+64 result and the final decision; the log artifact is republished.

## Files touched
- `README.md`: "Trial 4: the embedding model" in the learning journey (the normalisation finding, the four-model table in hits, the threshold reading, the choice), and "Where it stands" rewritten for the new defaults.
- `docs/retrieval-tuning.md`: the 256t+64 HYBRID result and an updated "Decision on the embedding model"; a short "Default embedding model" note that the default moved again.
- `eval/results/experiment_log.html`: re-rendered.

## Tests to write first
Exception: documentation and a publish step.

## Implementation notes
- Write with the `human-write-time` skill in the neutral register `~/.claude/VOICE.md` sets for README and docs: no first person, no dashes, digits for metrics, durations in words, sentence-case headings, British spelling.
- Numbers come from `eval/results/runs.jsonl` and `docs/retrieval-tuning.md` only.
- Republish `eval/results/experiment_log.html` to https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid with the `Artifact` tool, passing it as `url`.

## Verify
`grep -n "Trial 4" README.md` → one match; the artifact publish returns the same URL with a new version.

## Done when
- README and write-up updated, artifact republished.
- manual: owner reads the new README sections.
