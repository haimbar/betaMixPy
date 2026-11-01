# betaMixPy — Claude Instructions

This file defines the mandatory checklist Claude must complete **every time the
package is updated**, before committing or pushing anything.  Work through the
steps in order; do not skip or defer.

---

## Package overview

- **Language**: Python 3
- **Entry point**: `betaMixPy/betaMixPy.py` — all public functions live here.
- **Exports**: `betaMixPy/__init__.py` uses `from .betaMixPy import *`; every
  non-underscore name is exported.
- **Input convention**: `betaMix(M)` expects M as **N×P** (samples in rows,
  variables in columns), matching `np.corrcoef(M, rowvar=False)` and the R
  package.  Never reintroduce a transpose.
- **Return convention**: `betaMix()` returns a **dict** (not a tuple).  Keys
  mirror R's named list: `angleMat`, `z_j`, `m0`, `p0`, `ahat`, `bhat`,
  `etahat`, `ppthr`, `nodes`, `edges`, `bmax`, `N`, `cnt`, `method`.
- **Parallel R package**: `/Users/haim/Documents/GitHub/betaMix` (R).  Python and R
  implementations must stay semantically aligned.

---

## Mandatory update checklist

### 1. Docstrings

- Every public function must have a docstring that describes: purpose,
  parameters (name, type, default, meaning), return value, and any guarantees
  or invariants.
- Private helpers (`_MLEfun`, `_jacmle`, etc.) need at least a one-line
  description of what they compute.
- Do **not** describe *what* the code does line-by-line; describe *why* and
  *what the caller needs to know*.
- Verify that docstrings for modified functions still match the current
  signatures and return types after any change.

### 2. Test suite

- Tests live in `tests/test_betaMixPy.py`.  Run with:
  ```
  cd /Users/haim/Documents/GitHub/betaMixPy && python -m pytest tests/ -v
  ```
- After any change, run the full suite and fix all failures before proceeding.
- For each **new function**, add at least:
  - A smoke test (runs without error on a minimal input).
  - A correctness test (checks a known output or structural invariant).
  - An edge-case test (e.g., isolated nodes, all-zero adjacency, P=2).
- For `greedyBasis`: always assert independence (no two basis nodes adjacent)
  and maximality (every non-basis node has a basis neighbour).
- For `betaMix`: assert the dict contains all expected keys.
- Keep the DrySeeds reference values in the suite:
  `p0≈0.4751`, `ppthr≈0.7767`, `edges=203`.

### 3. Package dependencies

- `betaMixPy` depends on: `numpy`, `scipy`, `pandas`, `matplotlib`,
  `networkx`, `statistics`.
- Do not add new top-level dependencies without updating `setup.py`
  `install_requires` and the dependency list in `doc/betaMixPy.tex`.
- Before adding a dependency, check whether the stdlib or an existing
  dependency already covers the need.

### 4. OS portability

- Use `os.path.join()` or `pathlib.Path` for file paths — never hard-code `/`
  or `\` separators.
- Do not rely on Unix-only shell commands inside Python code.
- All file I/O in example scripts (`code/*.py`) should use relative paths from
  the script's directory or accept a configurable prefix.
- Test that the package imports cleanly on macOS (Darwin) — the primary
  development platform — and that nothing in the code assumes a specific locale
  or filesystem encoding.

### 5. Risky operations

Before committing, scan for and fix:

- **Command injection**: never pass unsanitised user input to `subprocess`,
  `os.system`, or `eval`.
- **Unbounded memory**: operations that allocate an N×P or P×P dense matrix
  where P could be very large.  Ensure `subsamplesize` is respected so the EM
  never allocates more than `O(subsamplesize)` working memory.
- **Silent data loss**: numpy operations that silently broadcast or truncate;
  always check shapes before matrix multiplication.
- **Division by zero**: `clusteringCoef` and any ratio — use `np.errstate` or
  `np.where` guards.
- **Convergence failures**: the MLE solver (`root(..., method="hybr")`) may
  not converge in pathological cases; always check `sol.success or
  np.allclose(sol.fun, 0, atol=1e-6)` and fall back gracefully.
- **Mutable defaults**: never use a mutable object (list, dict) as a default
  argument.

### 6. Documentation

Update **all** of the following — not just the ones closest to the change:

| File | What to update |
|---|---|
| `README.md` | Function tables, quick-start example, What's-new section |
| `NEWS.md` | Add an entry under the current version with breaking changes, new functions, fixes |
| `llms.txt` | Function reference section for any new or changed function |
| `doc/betaMixPy.tex` | Input-convention text, `\showCode` line references, new sections for major features |
| Docstrings | See step 1 |

Do **not** create additional documentation files (avoid README-2.md, NOTES.md,
etc.).  All changelog content goes in `NEWS.md`.

### 7. Lazy / deferred imports

- Import **only** what is needed at the top of the module:
  `numpy`, `scipy`, `pandas`, `matplotlib`, `networkx`, `statistics`.
- Heavy imports used only in plotting functions (`matplotlib.pyplot`,
  `networkx`) should be imported inside the function body if they are not
  already imported at module level, not at the top of the file, if they
  add significant load time.
- Never import an entire subpackage when only one symbol is needed (prefer
  `from scipy.special import psi` over `import scipy`).
- After adding any import, verify there are no circular imports by running
  `python -c "import betaMixPy"`.

### 8. GitHub update

Push to GitHub **only after** steps 1–7 are complete and the test suite passes.

```
git add <specific files>   # never "git add -A" without reviewing what's staged
git commit -m "..."
git push
```

Commit message format:
- First line: concise summary (≤ 72 chars), imperative mood.
- Body (optional): explain *why*, not *what*.
- Always include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

### 9. R package port plan

For **every new feature** added to betaMixPy, write a brief porting plan
describing how to implement the equivalent in the R package at
`/Users/haim/Documents/GitHub/betaMix`.  The plan should cover:

- Which R file(s) to modify (typically `R/betaMix.R` or a new `R/<feature>.R`).
- The R equivalents of any Python-specific constructs used
  (e.g., `np.searchsorted` → `findInterval`; `pd.DataFrame` → `data.frame`;
  `scipy.optimize.brentq` → `uniroot`; `np.argmin` → `which.min`).
- Whether the function should be exported in `NAMESPACE` (add
  `@export` roxygen tag).
- Any new R package dependencies (`Imports:` in `DESCRIPTION`).
- Suggested roxygen2 `@examples` block.

Write this plan as a code comment block at the top of the new Python function,
or as a separate `doc/port_plan_<feature>.md` file for larger features.

---

## Project layout

```
betaMixPy/
├── betaMixPy/
│   ├── __init__.py          # re-exports everything from betaMixPy.py
│   └── betaMixPy.py         # all functions
├── code/
│   ├── Simulated.py         # example: simulated data
│   ├── Tomatos.py           # example: metabolomics case study
│   └── compare_R_Python.py  # validation/benchmark vs R
├── data/                    # CSV datasets (N×P, samples in rows)
├── doc/
│   ├── betaMixPy.tex        # main LaTeX documentation
│   └── generated/           # runcode cache (do not edit manually)
├── figures/                 # output figures referenced by the LaTeX doc
├── tests/
│   └── test_betaMixPy.py    # pytest test suite
├── CLAUDE.md                # this file
├── llms.txt                 # LLM-friendly function reference
├── NEWS.md                  # version changelog
├── README.md                # user-facing README
└── setup.py                 # package metadata and version
```

## Parallel R package

R source: `/Users/haim/Documents/GitHub/betaMix`

When behaviour should match R exactly, validate against the reference values
recorded in `code/compare_R_Python.py`.  Known acceptable discrepancies:
- SixHourImbibed: < 1% edge difference due to `nleqslv` (R) vs `hybr`
  (Python) in a flat-likelihood region — both are valid solutions.
