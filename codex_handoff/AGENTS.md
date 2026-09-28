# AGENTS.md — working rules for building APABM

Place this file and `SPEC.md` at the root of a new (private) repository. `SPEC.md` is the
single source of truth.

## Before coding
- Read `SPEC.md` completely. **§0.3 (Revision v2) is authoritative**: where it conflicts with later sections, §0.3 wins. Build milestones M0 → M8 in order (SPEC §14).
- After each milestone: run `make lint test`, then append a dated entry to `PROGRESS.md`
  (what was built, test status, open issues).
- When the spec is ambiguous, choose the simplest option consistent with it and record the
  choice in `DECISIONS.md`. Never silently change semantics.

## Hard rules
1. **Data**: never commit anything under `data/raw`, `data/interim` or `data/processed`.
   Commit only `data/MANIFEST.json`. Do not attempt NSHAP/HRS/SHARE/CHARLS/ELSA downloads.
   NSHAP R3 (DS1 core, DS3 networks, Stata) is placed manually in `data/raw/nshap/r3/`; if present,
   use it via `NshapSource` (SPEC §0.3.2); if absent, the pipeline must still run (flag "pre-NSHAP").
2. **Verify variable names** against downloaded codebooks. Mappings live in
   `apabm/data/*_varmap.yaml`.
3. **Hidden vs observable**: controllers and baselines (except the oracle C10) receive only
   `Observation` objects. They must never import `sim.state` internals.
4. **Randomness** only through named streams from `apabm.rng`. No global `np.random`, no
   unseeded `random`.
5. **LLM calls** only through `apabm.llm.client.call`. No other module talks HTTP. Everything
   is logged.
6. **Caching**: replay only within the same run (condition + seed + config hash). Never reuse
   LLM outputs across conditions or seeds.
7. **Seeds**: tune only on dev seeds (0–9; legacy 42/100/200). Eval seeds 1000–1029 and
   legacy holdout 300/400/500/600 are for final runs only.
8. **No result-shaping**: do not tune dynamics, preference mappings or prompts to make any
   condition win. The only calibration targets are those in SPEC §8.
9. **Mock backend** is for tests and smoke runs only. The reporting code must refuse
   manifests with `backend=mock`.
10. Tests in SPEC §12 are mandatory and must never be skipped or weakened to pass.

## Style
- Python 3.11, type hints, dataclasses/pydantic v2, small pure functions for the dynamics
  and rules (easy to unit-test and audit).
- Config via YAML + pydantic. Every run writes a resolved config and a manifest.
- `ruff` clean; docstrings on public functions state units and ranges.

## Handy commands (to implement)
```
make env            # create conda env from environment.yml
make test           # pytest -n auto
make smoke          # E1 tiny (N=40, 4 weeks, 2 seeds) with mock backend
python -m apabm.data.download --sources brfss atus
python -m apabm.population.build --config apabm/experiments/configs/population.yaml
python -m apabm.calibration.virtual_rct --config ...
bash scripts/serve_vllm.sh persona   # GPU0 :8000
bash scripts/serve_vllm.sh diag      # GPU1 :8001
python -m apabm.experiments.run --exp E1 --backend vllm --resume
python -m apabm.analysis.report --exp E1
```
