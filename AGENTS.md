<!-- project-time:start v4 -->
## Project workflow (project-time)

This repo runs the project-time workflow. Its state lives in `docs/`: the files are the source of truth, the conversation is scratch.

**Memory has three levels**, from specific to general. Each has an `INDEX.md` (`## Pinned`, then one line per area) and area files.

| Level | Where | Holds | Changed |
|---|---|---|---|
| cycle | `docs/current/memory/` | the feature being built on this branch | as things happen |
| project | `docs/memory/` | what holds for this repo: code, product, working rules, this machine's setup | only by `project-wrap` (and setup, moving old auto-memory in) |
| general | `~/.agents/memory/` | what holds in every project: the user, their tools, lessons | at wrap, each item approved by the user, or when the user asks to remember something for every project |

The rest of `docs/`: `docs/current/` is the active cycle (`spec.md`, `plan/INDEX.md` with `plan/step-NN-*.md`, `memory/`); `docs/archive/<date>-<slug>/` holds finished cycles.

**Session start.** Read `~/.agents/memory/INDEX.md` and `docs/memory/INDEX.md`. If `docs/current/` exists, also read `docs/current/memory/INDEX.md` and `docs/current/plan/INDEX.md`. Open area files only when they relate to the work at hand. Where levels disagree, the more specific wins: cycle, then project, then general. Pins from all three apply.

**Recording.** Write to the cycle level as soon as it happens, before moving on:
- a decision (with the why and the rejected options), a discovery, a pivot, an issue;
- anything the user asks to pin or remember: a one-liner under `## Pinned` in the cycle INDEX, as `- <slug>.<NN> · <the rule>`. With no active cycle, a pin waits for the next cycle: say so.

When the user asks to remember something for every project, write it to the general level at once, as a pin or a line in an area file. Nothing else skips a level.

Entries go in area files by topic, each area listed in one line in the INDEX:

    ### <slug>.<NN> · <decision|discovery|pivot|issue> · <YYYY-MM-DD> · <phase or step>
    <the fact>
    **Why:** <reason; rejected options>

NN is the highest number in the cycle plus one, pins included. Never delete an entry. A pivot adds a new entry marked `Supersedes <id>` and appends `Superseded by <id>` to the old one. If the old one is in project memory, add `- <old> → <new> · <one line>` under `## Overrides` in the cycle INDEX instead.

**Flowing up.** At wrap, each cycle entry is promoted to project memory or archived with the cycle. Then whatever holds beyond this repo is proposed for the general level, one item at a time. Only promote what will be useful at the higher level.

**Claude Code auto-memory is off** in this repo (`.claude/settings.json`): everything goes to one of the three levels.

**Commits.** The workflow commits at setup, cycle start, gates, each finished build step, spec amendments, plan updates, parallel-run start and record, interim recaps, and wrap, each with only the files that step touched. Messages start with `<slug>:` or `<slug> step NN:`. Never push, and never merge a feature branch into the default branch. The general level lives outside the repo and is never committed.

**Sub-agents** never write under `docs/` or in `~/.agents/memory/`. They report back; the main agent records.

**To continue work,** run `/project-time`.
<!-- project-time:end -->
