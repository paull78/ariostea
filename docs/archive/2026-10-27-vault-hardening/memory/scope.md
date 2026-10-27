# Scope

What the cycle fixes and what it leaves out.

### vault-hardening.01 · decision · 2026-10-26 · spec
The cycle fixes nine findings of the 2026-10-26 usability assessment: the three blockers (a missing vault path wipes the index, one unreadable file aborts every run and kills `watch`, a long unbroken string becomes one 8,000-token chunk costing 20 to 25 GB of RAM), plus skipping hidden folders, `.trash/` and Excalidraw by default, headings inside code fences, `get_note` reading the file, removing the unused `openai_compat` embedding provider, progress output, and a README note on the 3.2 GB first-run download.
**Why:** each is small and testable, and together they cover every way a generic vault loses its index, stalls the machine or fills the index with junk. Rejected: blockers only (leaves junk chunks and wrong titles in every technical vault); adding the server issues (MCP reindex blocking, resume, concurrent writers, status time, BOM, config tracebacks) or everything listed (symlinks, unused keys, Desktop docs), which double the cycle for rarer problems.
