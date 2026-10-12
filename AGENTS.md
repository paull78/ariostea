<!-- project-time:start v2 -->
## Project workflow (project-time)

This repo runs the project-time workflow. Its state lives in `docs/`: the files are the source of truth, the conversation is scratch.

- `docs/memory/`: project memory. Read it; setup creates it and only `project-wrap` changes it.
- `docs/current/`: the active cycle on this branch. `spec.md`, `plan/INDEX.md` with `plan/step-NN-*.md`, and `memory/` (cycle memory).
- `docs/archive/<date>-<slug>/`: finished cycles.

**Session start.** Read `docs/memory/INDEX.md`. If `docs/current/` exists, also read `docs/current/memory/INDEX.md` and `docs/current/plan/INDEX.md`. Open area files only when they relate to the work at hand.

**Memory.** Record in `docs/current/memory/` as soon as it happens, before moving on:
- a decision (with the why and the rejected options), a discovery, a pivot, an issue;
- anything the user asks to pin or remember: a one-liner under `## Pinned` in the cycle memory INDEX, as `- <slug>.<NN> · <the rule>`. With no active cycle, a pin waits for the next cycle: say so.

Entries go in area files by topic, each area listed in one line in the INDEX:

    ### <slug>.<NN> · <decision|discovery|pivot|issue> · <YYYY-MM-DD> · <phase or step>
    <the fact>
    **Why:** <reason; rejected options>

NN is the highest number in the cycle plus one, pins included. Never delete an entry. A pivot adds a new entry marked `Supersedes <id>` and appends `Superseded by <id>` to the old one. If the old one is in project memory, add `- <old> → <new> · <one line>` under `## Overrides` in the cycle INDEX instead. Cycle memory wins over project memory while the cycle is active. Pins from both indexes apply.

**Commits.** The workflow commits at setup, cycle start, gates, each finished build step, spec amendments, plan updates, parallel-run start and record, interim recaps, and wrap, each with only the files that step touched. Messages start with `<slug>:` or `<slug> step NN:`. Never push, and never merge a feature branch into the default branch.

**Sub-agents** never write under `docs/`. They report back; the main agent records.

**To continue work,** run `/project-time`.
<!-- project-time:end -->
