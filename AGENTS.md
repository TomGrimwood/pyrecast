# pyrecast

Recast's navmesh build (recastnavigation 1.6.0) for Python. `README.md` has the interface.

## Rules

- **Recast as it is.** Build its pinned release unaltered; call its functions in its demo's
  order. What is added sits round it (`cpp/_core.cpp`) and is said in the README.
- **A small, general interface.** NumPy in, NumPy out; Recast's own terms (`Config` mirrors
  `rcConfig` in world units). Nothing of any one user's: no scenes, no file formats.
- **Tested on made-up scenes** (`tests/`): what an agent can and cannot reach, and every
  promise of the interface.
- The Python releases the GIL while Recast works; Recast's failures are RuntimeErrors saying
  which step, bad input ValueErrors.
- Comments and docstrings say plainly what the code does, for a reader with only the file.
- Work on a branch; merge to `main` with `--no-ff` when `make` passes. Commit only when asked.

## Commands

```bash
uv sync                                   # builds the extension
uv sync --reinstall-package pyrecast      # after changing cpp/, then its types:
uv run --with nanobind python -m nanobind.stubgen -m pyrecast._core -o src/pyrecast/_core.pyi
make                                      # lint, format, types, the tests
uv run python scripts/check.py            # what `make` runs, without make (Windows)
```
