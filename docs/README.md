# docs

For using Ariostea, start at the [README](../README.md). For how it was built and measured, read [LEARNING.md](../LEARNING.md).

This folder holds the full experiment account and the working records of the project, kept in the open as part of the learning project.

| Path | What it holds |
|---|---|
| [`retrieval-tuning.md`](retrieval-tuning.md) | The full account of the retrieval experiments behind `LEARNING.md`: every figure, and the commands that reproduce it. |
| `images/` | Charts used by `LEARNING.md` and `retrieval-tuning.md`. |
| `design/`, `plans/` | Design notes and implementation plans for the early experiments (eval set, chunking, contextual retrieval, embedding sweep). |
| `memory/` | Project memory: the decisions and discoveries that hold today, with the reason for each and the options rejected. `memory/overview.md` describes the system as it is. |
| `archive/` | One folder per finished development cycle: its spec, step-by-step plan, notes taken as it went, and a recap. |
| `current/` | The cycle in progress, when there is one. It moves to `archive/` when the cycle ends. |

The cycles were run with an AI coding agent, which kept these records as it worked; the root `recaps/` folder holds two earlier session recaps from the same work.
