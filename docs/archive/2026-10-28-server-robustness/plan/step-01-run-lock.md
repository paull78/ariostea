# Step 01 · RunLock port and adapters

**Milestone:** 1 · **Needs:** none · **Covers:** AC-04, AC-12

## Goal
A `RunLock` driven port with a file adapter on `fcntl.flock` and an in-memory fake, both passing one contract suite, and a cross-process test showing a SIGKILL leaves no lock behind.

## Files touched
- `src/ariostea/ports/run_lock.py`: new, the `RunLock` Protocol.
- `src/ariostea/adapters/lock/__init__.py`, `src/ariostea/adapters/lock/file.py`: new, `FileRunLock(path)`.
- `src/ariostea/adapters/lock/memory.py`: new, `InMemoryRunLock`, with a `handle()` method giving a second holder on the same lock (what another process is to the file adapter).
- `tests/adapters/lock/test_run_lock_contract.py`: new, the contract suite on both adapters.
- `tests/adapters/lock/test_file_run_lock.py`: new, the cross-process tests.

## Tests to write first
Contract suite, parametrised over a fixture `pair` that returns two holders of the same lock (file: two `FileRunLock` on one path; memory: a lock and `lock.handle()`):
- `test_try_acquire_on_a_free_lock_takes_it`: `a.try_acquire()` is True and `a.held()` and `b.held()` are both True.
- `test_try_acquire_on_a_taken_lock_refuses`: after `a.try_acquire()`, `b.try_acquire()` is False at once (under 0.5 s).
- `test_release_frees_it`: after `a.release()`, `held()` is False on both and `b.try_acquire()` is True.
- `test_acquire_waits_for_release`: `a` holds it; a thread releases `a` after 0.3 s; `b.acquire()` returns after at least 0.25 s and `b` then holds it.
- `test_held_is_false_on_a_fresh_lock`.

File adapter only:
- `test_lock_file_is_next_to_the_index`: the lock file exists at the given path after `try_acquire()` and is left in place after `release()`.
- `test_another_process_holding_it_is_seen`: a subprocess (`sys.executable -c` with `FileRunLock(path).acquire()` then printing `held` and sleeping) holds it; once it printed, `try_acquire()` is False and `held()` is True.
- `test_sigkill_releases_it` (AC-04): same subprocess, killed with `SIGKILL`; `try_acquire()` is True within 1 s after `wait()`.

## Implementation notes
- Port (owned by the app, purpose-named, no technology in the signature):
  ```python
  class RunLock(Protocol):
      def try_acquire(self) -> bool: ...   # take it if free; never waits
      def acquire(self) -> None: ...       # wait until free, then take it
      def release(self) -> None: ...
      def held(self) -> bool: ...          # any holder, this process included
  ```
- `FileRunLock`: opens the path with `os.open(path, os.O_RDWR | os.O_CREAT, 0o644)` on take, `fcntl.flock(fd, LOCK_EX | LOCK_NB)` for `try_acquire`, `LOCK_EX` for `acquire`, close the fd on release. `held()` returns True when this instance holds it, else probes with a fresh fd and `LOCK_EX | LOCK_NB`, releasing at once if it got it. flock locks belong to the open file description, so a probe from the same process sees a lock this process holds through another fd. Never delete the lock file (deleting a flock file races with a new holder).
- `InMemoryRunLock`: a shared state object holding a `threading.Lock` and the current holder; `handle()` returns a new `InMemoryRunLock` on the same state.
- Adapters subclass the port (setup.02). The dependency-rule test needs no change: `ports/` imports only `typing`.
- Add `ports.run_lock` to nothing else yet; step 03 wires it.

## Verify
`uv run pytest tests/adapters/lock tests/test_architecture.py -q` → all pass.

## Done when
- Both adapters pass the contract suite; the SIGKILL test passes.
- `uv run ruff check . && uv run ruff format --check .` is clean.
