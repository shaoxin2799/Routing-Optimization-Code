# APABM — Auditable Preference-Aware Policy Adaptation in Agent-Based Social Care

Full build specification for a from-scratch implementation. Target: an AAMAS/IJCAI-grade
paper extending our MABS paper *"Separating Diagnosis from Control: Auditable Policy
Adaptation in Agent-Based Simulations with LLM-Based Diagnostics"*.

Read this whole document before writing code. `AGENTS.md` contains the working rules.

---

## 0. Research context (what the code must support)

### 0.1 Prior work (MABS version, to be re-implemented as a baseline)
- ABM of an elderly-care facility, N=30 agents, T=200 days.
- Agent state: loneliness ℓ∈[0,1], frailty f, stress s, energy e; dynamic social network.
- Policy levers: social-event intensity θs∈[0.8,1.5]; home-visit eligibility threshold
  θt∈[0.4,0.6] (visit agents with ℓ>θt); visit probability θp∈[0.15,0.5].
- Every 7 days an LLM diagnoses agents with ℓ>0.6 and returns JSON (risk level, primary
  driver, priority_social, priority_visit). Aggregates: r (high-risk share), ps, pv (mean priorities).
- Deterministic control rules (MABS Eq. 2–3):
  - Δθs = min(0.05, 0.1·ps) if r>0.40 and ps>0.75, else 0
  - Δθt = −0.02 if pv>0.75 and θt>0.4, else 0
  - Δθp = +0.05 if pv>0.75 and θp<0.5, else 0
  - every |Δθ| ≤ cap (default 0.05), then clip to ranges.
- Conditions: Baseline (none), Fixed (θs=1.0, θt=0.6, θp=0.3), LLM Mapping (θs=1.2 if r>0.4
  else 1.0), Closed-loop (Eq. 2–3), Black-box LLM (LLM outputs θ directly).
- Published holdout results (seeds 300/400/500/600), mean final loneliness:
  Baseline 0.717, Fixed 0.674, LLM Mapping 0.680, Closed-loop 0.607, Black-box 0.687,
  Max policy 0.595 (with 2014 visits vs 1343 for closed-loop).

**Known defects of the MABS version that must NOT be reproduced:**
1. The update cap had no effect (sensitivity rows for cap 0.03/0.08 were identical).
2. The social-event rule (Eq. 2) never fired; sensitivity to r was exactly zero.
3. The text said only agents with ℓ>0.6 were diagnosed (8–12/cycle), but logs show ~28/cycle.
4. Comparisons were not resource-matched (closed-loop used ~70% more visits than black-box).
5. The black-box prompt did not state the objective, the budget, or the history (a straw man).
6. Only 4 seeds, and t-tests with n=4.
7. Diagnosis outputs were cached and replayed in a way that was ambiguous for a closed loop.

### 0.2 New paper: story and research questions
Residents have **heterogeneous hidden preferences**: some want more visits, some find visits
intrusive, some enjoy group activities, some dislike them. The policy adapts to **residents'
natural-language feedback**. The LLM **understands people** (it parses feedback into structured
signals and screens silent residents). **Explicit, auditable rules weigh people** (they aggregate
preferences under a stated welfare criterion, apply bounded updates, and respect budgets).
Evaluation always uses **hidden ground-truth utilities**, never an LLM judgement.

| RQ | Question | Hypothesis |
|---|---|---|
| RQ1 Alignment | At equal resources, does feedback-driven adaptation raise true utility? | Ours > static Pareto frontier, > state-only closed loop (MABS), > black-box LLM |
| RQ2 Voice bias | Does response-driven adaptation neglect silent high-need residents? Can it be corrected? | Naive aggregation hurts the silent and worst-off groups; IPW + silent-risk screening corrects this |
| RQ3 Aggregation & audit | Explicit welfare rules vs black-box: fairness, stability, traceability | Explicit rules make the fairness trade-off visible, with near-100% decision reproducibility |
| RQ4 Robustness | Noisy or exaggerated feedback | The separated architecture degrades boundedly, and less than black-box |
| RQ5 Need for LLM | Free text + LLM vs structured survey vs keyword parser | The LLM parses indirect feedback better and finds silent risk better |

---

## 1. Scope of this build round

### 1.1 Build and run now
- The whole code base: simulation, LLM layer, controllers, baselines, experiments, analysis.
- Population from **directly downloadable public data**: BRFSS and ATUS (see §3). No login is
  needed for either.
- A **synthetic population generator** with the identical schema, used for tests, for
  development, and as a fallback.
- Experiments E1–E7, E9, E10, and E8 with BRFSS as the reference distribution.

### 1.2 Deferred (needs a manual download with login; do NOT attempt now)
- NSHAP (ICPSR 20541 / 34921 / 36873): network roster, UCLA-3 items, social participation, longitudinal data.
- HRS, SHARE, CHARLS, ELSA.
- For these, implement **adapter stubs with interfaces only** (§3.5), so that dropping in NSHAP
  later changes configuration, not code.
- PPO baseline (C11): optional, last milestone.

### 1.3 Definition of done for this round
1. `pytest` passes, including every test in §12.
2. `make smoke` runs E1 end to end with the mock LLM backend in under 10 minutes on CPU.
3. The population is built from BRFSS+ATUS (or from the synthetic fallback, clearly flagged).
4. The legacy MABS replication (§8.1) has been run and reported.
5. E1–E7, E9 and E10 have been run with real vLLM backends. `results/` holds metrics, figures
   and tables. `RESULTS.md` summarizes them.
6. `PROGRESS.md` and `DECISIONS.md` are up to date.

---

## 2. Environment

- Server 2: 2× NVIDIA A100 80GB, Linux, CUDA 12.x.
- Python 3.11. Package name `apabm`. Install with `pip install -e .[dev]`.
- Core dependencies: numpy, pandas, pyarrow, networkx, scipy, scikit-learn, pydantic>=2,
  pyyaml, httpx, openai (client only), tqdm, matplotlib, seaborn, statsmodels, pyreadstat
  (for SAS XPT use `pandas.read_sas`), rich.
- LLM serving: `vllm` (a recent version with OpenAI-compatible server, guided JSON, TP).
- Dev tools: pytest, pytest-xdist, ruff, mypy (lenient).
- Provide `environment.yml`, `pyproject.toml`, `Makefile`, and `scripts/serve_vllm.sh`.
- Secrets: `HF_TOKEN` is optional, needed only for gated models such as Llama. Read it from
  env; never commit it.

### 2.1 Repository layout
```
apabm/
  config/            pydantic config models + YAML loading + config hashing
  rng.py             named RNG streams (SeedSequence-based)
  data/
    download.py      BRFSS + ATUS downloaders (manifest, checksums)
    brfss.py         load + harmonize BRFSS (65+)
    atus.py          load + derive time-use features (65+)
    fusion.py        ATUS->BRFSS statistical matching (PMM)
    sources/nshap.py STUB (interface only)
  population/
    schema.py        PersonRecord / Population dataclasses (single schema)
    synthetic.py     synthetic generator (same schema)
    clustering.py    GMM persona clustering (BIC)
    preferences.py   preference mappings M0/M1/M2/M_rand
    persona_cards.py deterministic persona text
    build.py         end-to-end population build CLI
  sim/
    state.py         hidden state vs Observation (strict separation)
    network.py       initial network + dynamics
    dynamics.py      daily updates (loneliness, stress, frailty)
    interventions.py events, visits, declines, budgets
    utility.py       hidden utility + gold preference labels
    voice.py         who speaks this week
    screening.py     periodic noisy loneliness screening
    env.py           BudgetedEnv and LegacyEnv (MABS semantics)
  llm/
    client.py        async OpenAI-compatible client (vLLM/Ollama) + MockBackend
    schemas.py       JSON schemas + pydantic validators
    prompts/         *.jinja templates (persona, parse, screen, legacy_diag, blackbox_*)
    persona.py       persona feedback generation
    diagnosis.py     feedback parsing, silent screening, legacy diagnosis
    cache.py         record/replay (same-run replay only)
  control/
    policy.py        Policy dataclasses (LegacyPolicy, BudgetPolicy)
    mabs_rules.py    MABS Eq. 2-3 (C3) + LLM Mapping
    aggregation.py   segment aggregation, IPW, welfare criteria
    bounded.py       bounded update + budget projection
    allocation.py    within-segment visit ranking/allocation
    audit.py         audit log records
  baselines/
    static.py        C0, C1, C2 grid (incl. Max)
    survey.py        C4 structured survey signals
    keyword.py       C5 keyword parser
    blackbox.py      C8, C9 (and legacy black-box)
    oracle.py        C10 myopic oracle
    ppo.py           C11 (optional)
  calibration/
    virtual_rct.py   effect-size calibration
  experiments/
    registry.py      condition registry C0..C11
    run.py           CLI runner (parallel seeds, GPU endpoints)
    configs/         E1..E10 YAML + legacy.yaml + smoke.yaml
  analysis/
    metrics.py       all metrics (§10)
    stats.py         paired tests, Holm, bootstrap, hypervolume
    plots.py         figures (§11)
    report.py        tables + RESULTS.md
tests/
scripts/serve_vllm.sh
data/                (gitignored except data/MANIFEST.json)
results/             (gitignored except results/**/summary*.csv and figures)
```

---

## 3. Data

### 3.1 Data policy
- `data/raw/`, `data/interim/` and `data/processed/` are **gitignored**. Only
  `data/MANIFEST.json` is committed. It records the URL, download time, SHA256, size and
  detected variables.
- Downloads are scripted: `python -m apabm.data.download --sources brfss atus`.
- BLS (bls.gov) rejects default Python user agents. Send the header
  `User-Agent: apabm-research/0.1 (contact: <set via env APABM_CONTACT_EMAIL>)`.
  Retry 4 times with backoff.
- **Variable names below are expected names. Verify each one against the downloaded
  codebook or data dictionary.** When a name differs, map it in `apabm/data/*_varmap.yaml`
  and note the change in `DECISIONS.md`. Never guess silently.
- If a download fails after retries, fall back to the synthetic population, set
  `population.source = "synthetic_fallback"` in every run manifest, and continue.

### 3.2 BRFSS (CDC): person-level attributes, including loneliness and support
- Index pages: `https://www.cdc.gov/brfss/annual_data/annual_2022.html` (and 2023, 2024).
  Expected file: `https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip` (SAS XPT).
  Also download the codebook and any module "version" files. Parse the index page for the
  actual hrefs rather than hard-coding them.
- Why BRFSS: the 2022 **Social Determinants & Health Equity (SD/HE) optional module** asks
  about loneliness and emotional support. Only the states that adopted the module have these
  items. For 2023 and 2024, detect whether the items exist and include those years if they do.
- Sample: age ≥ 65 (`_AGE80` ≥ 65 or `_AGEG5YR` ≥ 10), non-missing loneliness item.
- Expected variables (verify):

| Concept | Expected var | Harmonized field | Coding |
|---|---|---|---|
| Loneliness | `SDLONELY` | `lonely_freq` | 1 Always … 5 Never → ℓ̄ map below |
| Emotional support | `EMTSUPRT` | `support` | Always=1.0 … Never=0.0 |
| Life satisfaction | `LSATISFY` | `life_sat` | 0–1 |
| Stress | `SDHSTRE1` (verify) | `stress0` | 0–1 |
| Age | `_AGE80` / `_AGEG5YR` | `age` | years (top-coded 80) / band |
| Sex | `SEXVAR` | `female` | 0/1 |
| Marital | `MARITAL` | `marital` | married/widowed/divorced-separated/never/partner |
| Household adults | `NUMADULT` / `HHADULT` (verify) | `hh_adults` | int |
| Children in HH | `CHILDREN` | `hh_children` | int |
| Living alone | derived | `lives_alone` | hh_adults==1 & hh_children==0 |
| Education | `_EDUCAG` | `educ` | 4 levels |
| General health | `GENHLTH` | `gen_health` | 1–5 |
| Mentally unhealthy days | `MENTHLTH` | `ment_days` | 0–30 |
| Depression diagnosis | `ADDEPEV3` | `depression` | 0/1 |
| Difficulty walking | `DIFFWALK` | `mobility_limit` | 0/1 |
| Difficulty dressing | `DIFFDRES` | `selfcare_limit` | 0/1 |
| Difficulty with errands alone | `DIFFALON` | `errands_limit` | 0/1 |
| Cognitive difficulty | `DECIDE` | `cog_limit` | 0/1 |
| Weight | module weight if provided, else `_LLCPWT` | `w` | float |

- Baseline loneliness map: Always→0.85, Usually→0.70, Sometimes→0.50, Rarely→0.30,
  Never→0.15, plus Uniform(−0.07, 0.07) jitter from the `population` RNG stream.
- Frailty index f0 = mean(mobility_limit, selfcare_limit, errands_limit, (gen_health−1)/4,
  1[age≥80]), clipped to [0,1].

### 3.3 ATUS (BLS): time use and social activity for people 65+
- Index: `https://www.bls.gov/tus/data.htm`. Multi-year files are listed on pages like
  `https://www.bls.gov/tus/data/datafiles-0325.htm` (2003–2025) or `...-0324.htm`. Pick the
  latest page that exists. Download the Respondent, Roster, Who, Activity Summary and
  ATUS-CPS zip files, plus the data dictionaries (PDF).
- Sample: years 2015+ and age ≥ 65 (`TEAGE`). Use the final weight (`TUFINLWGT`).
- Derived per-respondent features (verify codes against the lexicon):
  - `soc_min`: minutes in socializing & communicating (major category 12), attending social
    events, religious activities (14), and volunteering (15).
  - `alone_min`: minutes coded "alone" in the Who file. `with_others_min`: minutes with
    non-household persons. Exclude sleep and personal care from the denominator.
  - `sa` (social-activity propensity) = weighted percentile rank of `soc_min` among 65+.
  - `tw` (time with others share) = with_others_min / waking minutes.
- Shared covariates with BRFSS (harmonize): age band (65–69, 70–74, 75–79, 80+), sex,
  marital (4 levels), lives_alone (from Roster: household size 1), education (4 levels),
  physical-disability flag from ATUS-CPS disability items (`PEDISPHY` / `PEDISOUT`; verify
  availability, else drop).

### 3.4 Fusion: ATUS → BRFSS (statistical matching)
- Fit `HistGradientBoostingRegressor` for `sa` and `tw` on the shared covariates with ATUS
  weights, using 5-fold CV. Report R² in `data/processed/fusion_report.json`. Low R² is
  expected and acceptable.
- Impute onto BRFSS persons with **predictive mean matching**: for each BRFSS person, take
  the 5 ATUS donors with the nearest predicted value and draw one donor's observed value
  (stream `fusion`). This preserves the marginal variance.
- Output: `data/processed/persons_65plus.parquet` (harmonized BRFSS + `sa`, `tw`, `w`).

### 3.5 Deferred source interface (NSHAP)
```python
class PopulationSource(Protocol):
    def persons(self) -> pd.DataFrame: ...           # harmonized schema §4.1
    def network_stats(self) -> NetworkStats | None:  # degree dist, tie strength, contact freq
    def longitudinal(self) -> LongitudinalStats | None  # for alpha calibration
```
Implement `BrfssAtusSource`, `SyntheticSource`, and `NshapSource` (a stub that raises
`NotImplementedError("download NSHAP from ICPSR; see README")`). Until NSHAP arrives,
`network_stats()` uses the literature-parameterized defaults in §5.3, and `alpha` uses the
defaults and sensitivity range in §5.2. Label every result "pre-NSHAP".

### 3.6 Literature parameters (not from data)
| Parameter | Default | Range for sensitivity | Source / note |
|---|---|---|---|
| Target visit effect (standardized mean difference after a 12-week virtual RCT) | d_v = 0.20 | 0.10–0.40 | Loneliness-intervention meta-analyses (e.g. Masi et al. 2011, PSPR) report small effects. Verify the exact values from the paper before writing them into the text |
| Target event effect (for attenders with μ>0) | d_e = 0.15 | 0.05–0.30 | same |
| Loneliness reversion α | 0.02/day | {0.01, 0.02, 0.05} | to be calibrated from NSHAP later |
| Mean core network size | 3.5 | 2.5–4.5 | to be replaced by NSHAP roster |

---

## 4. Population

### 4.1 Harmonized person schema (the only schema downstream code may use)
`pid, age, female, marital, lives_alone, educ, gen_health, ment_days, depression,
mobility_limit, selfcare_limit, errands_limit, cog_limit, lonely_freq, l_base, support,
life_sat, stress0, f0, sa, tw, w, cluster (added later)`

### 4.2 Synthetic generator
- Produces the same schema with plausible marginals: l_base ~ Beta(2.2, 2.8);
  mobility_limit ~ Bern(0.3); lives_alone ~ Bern(0.3); support ~ Beta(3,1.5); sa ~ U(0,1)
  correlated −0.3 with l_base; and so on.
- Deterministic given the seed. Used by tests and smoke runs, and as a fallback.

### 4.3 Persona clustering
- Features (standardized): l_base, support, life_sat, f0, mobility_limit, lives_alone, age,
  depression, sa, tw.
- Draw a weighted resample of 20,000 persons (probability ∝ w). Fit `GaussianMixture` for
  K = 3..7 with 5 initializations each and pick K by BIC. Save the model, cluster
  proportions, per-cluster means, and a human-readable cluster summary in
  `data/processed/personas.json`.
- Name the clusters deterministically from their feature profile, e.g. "socially active",
  "isolated & limited", "independent & private", "supported family-oriented". Codex writes
  the naming rules; they are not LLM-generated.

### 4.4 Sampling a simulated population
For a given seed and N, draw N persons with probability ∝ w from `persons_65plus.parquet`
(stream `population`). Assign each person their cluster, then derive preferences (§4.5) and
a persona card (§4.6).

### 4.5 Hidden preference mapping (assumption M; reported and sensitivity-tested)
Let λ_max = 0.10 and σ(x) = logistic.
- **μ_i ∈ [−1,1] (group-activity preference)**:
  `μ = clip( 2(sa−0.5) + 0.8·mobility_limit·(1−sa) + N(0, 0.15), −1, 1 )`.
  Rationale: low participation without physical limitation suggests a genuine preference for
  less group activity; low participation with limitation suggests the person is constrained
  but still willing.
- **λ_i ∈ [0, λ_max] (intrusion aversion to visits)**:
  `λ = λ_max·σ( 2.5(support−0.5) + 1.0(1−lives_alone) − 2.0(l_base−0.5) + N(0,0.3) )`.
- **τ_i ∈ {0,1,2,3} (comfortable visits/week)**: `τ = round(3·(1 − λ/λ_max))`.
- **cog_i** = cog_limit.
- Variants for E9: **M1** = all coefficients ×0.5; **M2** = μ from sa only, λ from support only;
  **M_rand** = randomly permute (μ, λ, τ) across agents, which breaks the data link.
- Sanity report (not a tuning target): the share of agents for whom a 2nd weekly visit has
  negative marginal utility. Log it; do not tune it.

### 4.6 Persona card (deterministic template)
```
You are a {age}-year-old {woman/man}, {marital phrase}, who {lives alone / lives with family}.
Your general health is {excellent..poor}. {You have difficulty walking.|}{You need help with errands.|}
You {rarely/sometimes/often} take part in group or social activities.
{You have people you can rely on for emotional support.|You rarely have someone to confide in.}
About you: {μ>0.3: "you enjoy group activities and company";
            μ<-0.3: "you prefer quiet, one-to-one company and your privacy";
            else: "you like some company but not too much"}.
{λ high: "You value your independence and do not like people dropping by too often."}
Communication style: {one of: brief; polite and indirect; talkative; reserved; tends to complain}.
```
The style is drawn from the `persona` stream at population build. The card is **hidden** from
controllers.

---

## 5. Simulation model

### 5.1 Time, scale and budgets
- Daily steps. Weekly decision cycle (every 7 days). Main runs: T = 182 days (26 weeks),
  N = 200. Scale runs: N ∈ {200, 500, 1000}.
- Budget per week: visits B_v = 0.2·N (40 at N=200); events B_e = 3. Both scale with N.
- E2 budget levels (B_v/N, B_e): (0.05,1), (0.1,2), (0.2,3), (0.3,4), (0.5,5).
- Warm-up: 14 days with no intervention before the first decision.

### 5.2 Daily dynamics (all parameters in YAML; defaults shown)
```
ℓ_i ← clip( ℓ_i + α(ℓ̄_i − ℓ_i) − β_soc·c_i − β_e·h(μ_i)·a_i − β_v·v_i
            + κ_s(s_i − 0.3) + σ_ℓ·ε , 0, 1 )
s_i ← clip( s_i + α_s(s0_i − s_i) + 0.1·κ_ls(ℓ_i − 0.5) + s_intr·(λ_i/λ_max)·x_i − β_vs·v_i , 0, 1 )
f_i ← clip( f_i + φ_f(1 + s_i) , 0, 1 )
```
- c_i = min(1, informal interactions today / 3). Interactions come from network ties and
  household members (§5.3).
- a_i = 1 if the agent attended an event today; h(μ) = clip(0.5 + 0.5μ, 0.1, 1).
- v_i = 1 if an accepted visit happened today; x_i = 1 if that visit exceeds τ_i this week.
- Defaults: α=0.02, β_soc=0.01, β_v and β_e from calibration (initial 0.04 and 0.03),
  σ_ℓ=0.01, κ_s=0.02, α_s=0.05, κ_ls=0.2, s_intr=0.05, β_vs=0.02, φ_f=0.0003.

### 5.3 Network
- Initial degree ~ NegBin(mean 3.5, dispersion 2), truncated to [0, 8], via a configuration
  model with homophily rewiring on (cluster, ℓ). Tie strength w_ij ~ Beta(2,2).
  (NSHAP will replace this later.)
- Household: if not lives_alone, add a virtual household contact with daily interaction
  probability 0.8. This contact is not an agent.
- Daily interaction per tie: p = 0.15·w_ij·(1 − 0.5·max(f_i,f_j)).
- Tie formation at events: each pair of co-attendees forms a tie with probability
  0.02·sim_ij, where sim_ij = exp(−|ℓ_i−ℓ_j|/0.2)·(1 + 0.5·1[same cluster])/1.5; new w = 0.3.
- Decay: a tie with no interaction for 30 days is removed with probability 0.02/day.

### 5.4 Interventions (BudgetedEnv)
- **Events**: n_e ∈ {0..B_e} per week, on fixed weekdays. Attendance per event:
  `P = σ(−0.5 + 2.0μ_i − 1.5f_i + 0.8·n_e/B_e)`. The last term is staff encouragement: more
  events means more pressure, so reluctant introverts sometimes attend.
- **Visits**: the controller allocates at most B_v visit slots per week, at most `v_max = 2`
  per person. Each allocated visit may be declined:
  `P(decline) = clip(0.05 + 0.4·λ_i/λ_max + 0.3·1[V_i^w ≥ τ_i], 0, 0.95)`.
  A declined visit still consumes its slot. Declines are observable.
- Budgets must never be exceeded; there is a test for this.

### 5.5 Hidden utility (the only evaluation ground truth)
Weekly, per agent:
```
u_i^w = − mean_{days in week} ℓ_i²  − λ_i·max(0, V_i^w − τ_i)  + c_A·μ_i·A_i^w
```
V = accepted visits this week, A = events attended this week, c_A = 0.02. The squared
loneliness term makes a reduction worth more for lonelier people, so need-based targeting
is justified by the utility itself.

**Gold preference labels** (for evaluating parsers and alignment):
- Let k = (1/7)·Σ_{d=0..6}(1−α)^d be the carry-over factor of a one-off loneliness reduction
  onto next week's mean ℓ (derived from the reversion dynamics).
- `true_visit_pref` = sign of the marginal utility of one more accepted visit next week:
  `m_v = 2·β_v·k·ℓ_i − λ_i·1[V_i^w + 1 > τ_i]`. Label "more" if m_v > ε, "less" if
  m_v < −ε, otherwise "same" (ε = 0.002).
- `true_event_pref` from `m_e = P(attend)·(2·β_e·h(μ_i)·k·ℓ_i + c_A·μ_i)`, with the same thresholds.
- `true_satisfaction` = min-max-normalized u_i^w across the population-week.

### 5.6 Voice (who gives feedback this week)
`P(speak) = v0·(1 − s_ℓ·ℓ_i)·(1 − s_f·f_i)·(1 − s_c·cog_i)`, with v0=0.5, s_ℓ=0.6 (the E3 sweep
uses {0, 0.3, 0.6, 0.9}), s_f=0.3, s_c=0.3. Stream `voice`. Record `v_i^0`, agent i's voice
propensity at t=0, to define the "silent" group (bottom quartile).

### 5.7 Screening (observable need)
Every 28 days, every agent has a staff screening. Observed `ℓ_obs = clip(ℓ + N(0, 0.10))`
(stream `screen`). Screening results are observable to all controllers.

### 5.8 Observation vs hidden state (enforced)
- Controllers receive an `Observation` object only: pid, age band, female, lives_alone,
  mobility_limit, cog_limit, segment id, latest ℓ_obs and its change, attendance history,
  visits allocated/accepted/declined history, weeks since last feedback, this week's feedback
  text (or None), and the budget.
- **Hidden**: ℓ, s, f exact values, μ, λ, τ, cluster, persona card, voice propensity, network.
- `Observation` is a frozen dataclass built by `env.observe()`. There is a test that
  controllers cannot reach the hidden state.

### 5.9 LegacyEnv (MABS semantics, for replication and C3-legacy)
N=30, T=200, homogeneous synthetic agents (the synthetic source with preference
heterogeneity switched off: μ=0.5, λ=0), no budget, levers θs/θt/θp as in §0.1: each day
eligible agents (ℓ>θt) receive a visit with probability θp/7, and events have an effect ∝ θs.
Diagnosis starts after a 28-day warm-up. Holdout seeds 300/400/500/600; dev seeds 42/100/200.

---

## 6. LLM layer

### 6.1 Backends
- `OpenAICompatBackend`: async httpx/openai client to a vLLM server (or Ollama `/v1`).
  Configurable concurrency (default 256), timeout, and retries.
- `MockBackend`: deterministic, rule-based responses from templates (for tests and smoke
  runs). The mock persona uses canned sentences keyed by felt signals. The mock parser
  reads the hidden gold label with noise. **The mock must never be used in reported results.**
- Every call goes through `llm.client.call(role, messages, schema, params, meta)`. It logs
  model id, prompt hash, params, response text, parse status, fallback flag, tokens and latency
  to `llm_calls.jsonl.gz`.

### 6.2 Serving (scripts/serve_vllm.sh)
- GPU0: persona model on port 8000. GPU1: diagnosis + black-box model on port 8001.
- Structured output: use vLLM guided JSON (`response_format={"type":"json_schema",...}`,
  or `extra_body={"guided_json": schema}` depending on the vLLM version). Still parse and
  validate everything.
- For Qwen3 models, disable thinking: `extra_body={"chat_template_kwargs":{"enable_thinking": false}}`.
- Default models (HF ids; persona and diagnosis must be **different families**):

| Role | Default | Fallback if no HF_TOKEN |
|---|---|---|
| Persona (LLM-A) | `Qwen/Qwen3-8B` | same |
| Diagnosis + black-box (LLM-B) | `meta-llama/Llama-3.1-8B-Instruct` (gated) | `microsoft/phi-4` |
| E7 extra diagnosis | `Qwen/Qwen3-14B`, `mistralai/Mistral-Small-24B-Instruct-2501` | — |
| E7 large | `Qwen/Qwen2.5-72B-Instruct-AWQ` (TP=2) or `meta-llama/Llama-3.3-70B-Instruct` if token | — |

When running a 72B model with TP=2, run the persona 8B model on GPU0 with
`--gpu-memory-utilization 0.3`, and the 72B-AWQ model with 0.6 per GPU. Record exact model
revisions, the vLLM version, and quantization in the manifest.
- Sampling: persona T=0.7, max_tokens=150; parser T=0.1, max_tokens=200; screening T=0.1,
  max_tokens=200; black-box T=0.1, max_tokens=600. Seed each request deterministically
  from (run seed, week, pid, role).

### 6.3 Prompts (templates in `llm/prompts/`, versioned; the prompt hash goes in the manifest)
**persona_feedback** (LLM-A). Input: persona card, week summary, felt signals. Output
`{"message": str}`.
```
[system] You are role-playing a resident of a community aged-care service. Stay in character.
[user] {persona_card}
This week: you had {V} home visit(s) and attended {A} of {n_e} group activities offered.
How you feel (do not quote these words literally): {felt_signals}
Write a short message (1–3 sentences) to the care staff about how things are going for you.
Speak naturally in your communication style; you may be indirect. Return JSON {"message": "..."}.
```
Felt signals are derived from the true state and preferences. Examples: "quite lonely this
week", "the visits felt like too much", "you would welcome more visits", "the group activity
was uncomfortable", "you would enjoy more activities". **Fidelity noise** η ∈ {0, 0.2, 0.4}:
each signal is dropped or flipped with probability η (stream `feedback_noise`). **Exaggerators**
(a fraction x ∈ {0, 0.1, 0.3}, fixed per agent) always claim high loneliness and want more visits.

**feedback_parse** (LLM-B). Input: message plus observable segment descriptors. Output:
```json
{"satisfaction": 0-1, "visit_pref": "more|same|less", "event_pref": "more|same|less",
 "urgency": 0-1, "confidence": 0-1}
```
Instruction: use only evidence in the message; when it is unclear, answer "same" with low confidence.

**silent_screen** (LLM-B). Only for residents without feedback this week, at most every 2
weeks per resident. Input: observable record summary (age band, lives alone, mobility,
last ℓ_obs and its change, attendance and visits accepted/declined over the last 4 weeks,
weeks since last feedback). Output `{"risk": "low|medium|high", "priority_visit": 0-1, "reason": str}`.

**legacy_diagnosis** (C3 / LLM Mapping). Reproduce the MABS appendix prompt: agent state,
7-day interactions, network position. Output `risk_loneliness, risk_frailty, primary_driver,
priority_social, priority_visit, confidence`.

**blackbox_legacy**: the MABS appendix prompt (aggregate r, ps, pv and current θ → θ).
Used only in the legacy replication.

**blackbox_informed** (C8/C9). States the objective explicitly: maximize residents'
well-being (lower loneliness) while respecting their preferences, not over-visiting people
who do not want visits, and being fair across groups. Also gives the budget, per-segment
summaries (n_g, response rate, counts of more/same/less for visits and events, mean
satisfaction, mean ℓ_obs, declines, silent high-risk flags), the last 4 weeks of policy and
aggregates, and up to 30 feedback messages sampled stratified by segment. For C9, it also
states the step limit. Output:
```json
{"n_events": int, "segment_shares": {"<segment_id>": float, ...}, "rationale": str}
```

### 6.4 Validation, fallback and caching
- Validate with pydantic. On failure, retry once. If it fails again, apply the fallback:
  parser → `same/same, satisfaction 0.5, confidence 0`; screen → `risk=low`; black-box →
  previous policy. Log the parse status and the fallback flag.
- **Record/replay** is allowed only to replay *the same run* (same condition, seed, config hash)
  for the E6 audit. **Never** reuse LLM outputs across conditions or across seeds. There is a
  test for this.

---

## 7. Controllers and conditions

### 7.1 Policy objects
- `LegacyPolicy(theta_s, theta_t, theta_p)`.
- `BudgetPolicy(n_events: int, segment_shares: dict[seg, float], v_max: int = 2)`.
  Shares sum to 1.

### 7.2 Segments (observable)
lives_alone × mobility_limit × age≥80, giving 8 cells. Merge any cell with fewer than 5
members into its nearest neighbor (fixed merge order documented).

### 7.3 Within-segment visit allocation (shared by all budgeted controllers except the oracle)
Segment quota q_g = round(π_g·B_v), with largest-remainder rounding so that Σq_g = B_v.
Rank members by priority:
```
p_i = ℓ_obs_i + γ_d·d̂_i + γ_r·risk_i − 1.0·block_i
```
- d̂_i ∈ {−1,0,1}: latest parsed visit_pref, which decays linearly to 0 over 4 weeks.
- risk_i: high=1, medium=0.5, low=0.
- block_i = 1 if the agent said "less" within the last 2 weeks or declined ≥ 2 times in the last 4 weeks.
- γ_d = 0.2, γ_r = 0.2.

Assign visits to the top of the ranking, at most v_max per person. Unused quota flows to the
next segment by the global ranking.

### 7.4 C7: ours (LLM parse + silent screening + IPW + explicit welfare rule + bounded update)
Per week:
1. Parse all feedback (LLM-B). Screen silent residents (LLM-B).
2. Per segment g: members n_g, responders R_g, response rate r̂_g = EMA(|R_g|/n_g, 0.3),
   floored at 0.05.
3. Signals: d_i (visit), e_i (event) ∈ {−1,0,1}; sat_i. Silent residents flagged high or
   medium are imputed as d=+1 with sat 0.2 (high) or 0.4 (medium).
4. **IPW / Hájek segment estimates**: D_g = mean of d over responders ∪ flagged;
   ŝ_g = mean of sat over the same set. Population-level quantities weight segments by n_g
   (not by the number of responses).
5. **Welfare criterion** (a config value; three variants):
   - utilitarian: π*_g ∝ n_g·max(0.05, 1 + D_g)
   - rawlsian: π*_g ∝ n_g·max(0.05, 1 + D_g)·exp(−5(ŝ_g − min_h ŝ_h))
   - nash: π*_g ∝ n_g·max(0.05, 1 + D_g)/max(ŝ_g, 0.05)
6. **Bounded update**: π_g ← π_g + clip(π*_g − π_g, −δ, δ) with δ = 0.05, then project onto
   the simplex (iterate clip+renormalize until it converges, or use exact projection).
7. **Events**: E = Σ_g n_g·Ē_g / N. If E > 0.15, n_e += 1; if E < −0.15, n_e −= 1. Clip to [0, B_e].
8. Write an audit record for every change: inputs (per-segment D, ŝ, r̂, counts), the rule
   id, pre and post values, and the clip events.

### 7.5 Condition registry

| ID | Name | Env | LLM | Definition |
|---|---|---|---|---|
| C0 | none | Budgeted | – | no events, no visits |
| C1 | fixed | Budgeted | – | n_e = ceil(B_e/2); shares ∝ n_g; full visit budget; ranking uses ℓ_obs only |
| C2 | static grid | Budgeted | – | n_e ∈ {0..B_e} × budget use ∈ {0, .25, .5, .75, 1} × targeting ∈ {need-ranked, random}. **Max** = (B_e, 1, need-ranked) |
| C3 | MABS closed loop | Budgeted (+Legacy) | legacy_diagnosis | MABS Eq. 2–3 on θ. Budgeted mapping: n_e = round(1 + (B_e−1)(θs−0.8)/0.7); each agent with ℓ_obs > θt receives one visit this week with probability θp; if this exceeds B_v, keep the highest ℓ_obs |
| C4 | structured survey | Budgeted | – | responders give Likert or more/same/less answers drawn from the gold labels with the same fidelity noise; no screening LLM (silent flag = ℓ_obs > 0.6 and ℓ_obs change > 0.05); otherwise identical to C7 |
| C5 | keyword parser | Budgeted | persona only | same free text as C7, parsed by keyword/regex rules (documented lexicon); screening by the rule in C4; otherwise C7 |
| C6 | ours, naive | Budgeted | ✓ | LLM parse, **no IPW, no silent screening**: π*_g ∝ |R_g|·max(0.05, 1+D_g) (attention follows voices); events weighted by responses |
| C7 | ours, full | Budgeted | ✓ | §7.4; variants C7u, C7r, C7n |
| C8 | black-box informed | Budgeted | ✓ | blackbox_informed sets n_e and the shares; allocation §7.3 (with the same parsed signals) |
| C9 | black-box + cap | Budgeted | ✓ | C8 with the step limit stated in the prompt and enforced after the call (|Δπ| ≤ δ, |Δn_e| ≤ 1) |
| C10 | myopic oracle | Budgeted | – | sees hidden state. Visits: greedy on true marginal utility m_v (§5.5), respecting B_v and v_max. Events: choose n_e by a one-week lookahead (2 rollouts, common random numbers) |
| C11 | PPO (optional) | Budgeted | – | policy on aggregated observables → (n_e, shares); trained on dev seeds only |
| L-* | legacy set | Legacy | per MABS | L-Baseline, L-Fixed, L-Mapping, L-ClosedLoop (C3), L-BlackBox (legacy prompt), L-BlackBox-informed, L-Max |

---

## 8. Calibration and legacy replication

### 8.1 Legacy replication (sanity check, not a headline result)
- Implement LegacyEnv (§5.9). On **dev seeds only** (42, 100, 200), tune α, β_soc, β_v and β_e
  so that the L-Baseline final mean ℓ ≈ 0.72 and L-Fixed ≈ 0.67 (the published values).
- Then run all L-* conditions on holdout seeds 300/400/500/600 **and** on 30 fresh seeds.
- Report: final loneliness, visits, events, rule-firing counts (each MABS rule branch), cap
  clip counts, and the number of agents diagnosed per cycle.
- Expected: ordering similar to MABS. Deviations are reported as they are, not tuned away.

### 8.2 Effect calibration by virtual RCT (BudgetedEnv, dev seeds 0–9)
- Visits: treatment gets 1 visit/week (never declined in the trial), control gets none,
  12 weeks, N=400 split 1:1. Choose β_v by bisection so that the standardized difference in
  final ℓ equals d_v.
- Events: the same design, restricted to agents with μ>0 and forced attendance of 1/week,
  giving β_e from d_e.
- Freeze the values in `configs/calibrated.yaml` along with the calibration report.
  **All later experiments load this file.**

---

## 9. Experiments

Seeds: dev = 0–9 (tuning only); eval = 1000–1029. **Common random numbers**: within a seed,
all conditions share the initial population and the exogenous streams (population, network0,
dynamics noise, screening noise). Condition-dependent draws use separate named streams so that
paired comparisons are valid.

| Exp | RQ | Design | Conditions | Seeds |
|---|---|---|---|---|
| E1 main | RQ1, RQ3 | N=200, default budget, s_ℓ=0.6, η=0.2, x=0 | C0–C10 (C7 × u, r, n) | 30 |
| E2 budget/Pareto | RQ1 | 5 budget levels | C2 grid, C3, C7r, C8, C10 | 20 |
| E3 voice bias | RQ2 | s_ℓ ∈ {0, 0.3, 0.6, 0.9} | C4, C5, C6, C7r, C8 | 20 |
| E4 robustness | RQ4 | η ∈ {0, 0.2, 0.4} × x ∈ {0, 0.1, 0.3} | C7r, C8, C9 | 20 |
| E5 welfare criteria | RQ3 | taken from E1 | C7u, C7r, C7n | (E1) |
| E6 audit | RQ3 | (a) replay the same run from its LLM log → the policy must be identical; (b) 5 fresh re-samples of the LLM at T=0.1 and at T=0.7 → policy variance; (c) leave-one-feedback-out counterfactual influence | C7r, C8 | 20 |
| E7 models | generality | persona ∈ {Qwen3-8B, Llama-3.1-8B/phi-4} × diagnosis ∈ {Llama-3.1-8B/phi-4, Qwen3-14B, Mistral-Small-24B}, plus 72B diagnosis | C7r, C8 | 10 |
| E8 persona validity | validity | (a) at t=0, the persona LLM answers the BRFSS loneliness item ("How often do you feel lonely?" 5 options); compare with source BRFSS by cluster (Wasserstein, KS); (b) parser accuracy vs gold labels (C5 keyword vs LLM parse) including indirect-style personas; (c) persona consistency over 4 weeks | persona + parse | 5 |
| E9 sensitivity | robustness | mapping ∈ {M0, M1, M2, M_rand} × effects d_v ∈ {0.1, 0.2, 0.4} × α ∈ {0.01, 0.02, 0.05} (one-at-a-time around the default); also δ ∈ {0.02, 0.05, 0.1} and the event threshold, **reporting rule-firing counts** | C2(Max), C7r, C8, C10 | 10 |
| E10 scale | scalability | N ∈ {200, 500, 1000} | C7r, C8 | 5 |
| L legacy | replication | §8.1 | L-* | 4 + 30 |

Estimated load: about 1,800 LLM runs, each about 1–2 minutes at N=200 with batched vLLM.
Run 8–16 seeds concurrently. The total is well under 2 days on 2×A100. The runner must
support `--resume` (skip completed run dirs), `--only C7r,C8`, `--seeds 1000-1009` and
`--backend mock|vllm`.

---

## 10. Metrics (computed from logs; LLM never scores)
Per run (and per-agent tables saved for pooling):
- **Mean welfare** W = mean_i mean_w u_i^w.
- **Worst-group welfare** W_min = min over *hidden persona clusters* of the cluster-mean
  welfare. Also report per-segment values.
- **Silent welfare** W_silent: welfare of the bottom quartile of v_i^0.
- Final loneliness (mean ℓ at T) and mean loneliness over time.
- **Burden** = Σ_i Σ_w λ_i·max(0, V_i^w − τ_i); declined visits.
- **Resources**: visits allocated and accepted, events, LLM calls and tokens, wall time.
- **Fairness**: group welfare gap (max − min over clusters); Gini of final ℓ.
- **Alignment**: the share of agent-weeks with true_visit_pref ≠ same where the sign of the
  next-week change in V_i matches true_visit_pref.
- **Parser accuracy** (E8): macro-F1 of the parsed visit_pref and event_pref vs gold.
- **Audit**: replay identity (bool); policy variance across LLM re-samples (mean L1 distance
  of the share vectors, plus |Δn_e|); the share of policy changes with a complete audit
  record; leave-one-out influence distribution.
- **Robustness**: relative drop in W vs η=0, x=0.

## 11. Statistics, figures, tables
- Paired by seed: Wilcoxon signed-rank tests for the key contrasts (C7r vs each of C1, C2-best-
  at-equal-visits, C3, C6, C8, C9; C6 vs C7r on W_silent), with Holm correction. Report the
  paired rank-biserial r and bootstrap 95% CIs (10k resamples).
- **Resource-matched comparison**: for each method, take the C2 static configuration with the
  closest accepted-visit count (±5%) on the same seed, and compare against it.
- Pareto: accepted visits (x) vs W (y); the static frontier = upper hull of C2. Report the
  hypervolume with reference point (max visits, W of C0).
- Figures: (1) architecture (manual); (2) Pareto (E2); (3) voice-bias curves W_silent and
  W_min vs s_ℓ (E3); (4) welfare-criterion trade-off W vs W_min (E5); (5) robustness curves
  (E4); (6) policy trajectories with ±SD; (7) E8 persona-validity histograms.
- Tables: main E1 (mean ± CI for all metrics); resource-matched contrasts; audit (E6); model
  grid (E7); sensitivity (E9); legacy replication (L).
- `analysis/report.py` writes `results/<exp>/summary.csv`, the figures (PDF+PNG) and `RESULTS.md`.

## 12. Tests (must exist and pass)
1. `test_cap_enforced`: MABS rules with cap=0.03 produce Δθp = 0.03 (not 0.05).
2. `test_mabs_rules_fire`: synthetic aggregates trigger each branch of Eq. 2–3.
3. `test_budget_never_exceeded`: property test over random policies.
4. `test_hidden_isolation`: controllers receive only `Observation`; accessing hidden fields fails.
5. `test_crn_initial_population_identical`: same seed ⇒ identical population, network0 and screening noise across conditions.
6. `test_determinism_mock`: same config + seed + mock ⇒ identical outputs.
7. `test_replay_identity`: record then replay ⇒ identical policy trajectory for C7.
8. `test_no_cross_condition_cache`: the cache refuses a key from another condition or seed.
9. `test_schema_fallback`: invalid JSON ⇒ retry ⇒ fallback, and the flag is logged.
10. `test_ipw_unbiased`: toy data with segment-dependent response rates ⇒ the IPW/Hájek
    estimate is unbiased and the pooled naive estimate is biased.
11. `test_population_schema`: synthetic and BRFSS+ATUS builders emit the identical schema.
12. `test_metrics_toy`: hand-computed toy example for every metric.
13. `test_simplex_projection` and `test_bounded_update`.
14. `test_gold_labels`: the marginal-utility sign matches a finite-difference simulation on toy agents.

## 13. Run manifest and outputs
`results/<exp>/<condition>/<seed>/`:
- `config.yaml` (resolved), `manifest.json` (git SHA, config hash, prompt hashes, model ids and
  revisions, vLLM version, GPU, population source, calibrated.yaml hash, start/end time),
- `weekly.parquet` (per-week aggregates and policy), `agents.parquet` (per-agent-week: ℓ, s, f,
  V, A, declines, u, spoke, gold labels, parsed labels),
- `audit.jsonl`, `llm_calls.jsonl.gz`, `metrics.json`.

## 14. Milestones (in order; each ends with green tests and a PROGRESS.md entry)
- **M0** Scaffolding: pyproject, env, Makefile (`make test`, `make smoke`, `make lint`), config system, RNG streams.
- **M1** Simulation core with the synthetic population; C0/C1/C2/C10; tests 3–6, 12–14.
- **M2** LLM layer: backends (mock, OpenAI-compatible), schemas, prompts, logging, record/replay; tests 7–9.
- **M3** LegacyEnv + L-* conditions; tests 1–2; legacy replication with a real LLM (§8.1).
- **M4** Feedback loop: voice, persona, parse, screen, aggregation/IPW, C4–C9, audit log; test 10.
- **M5** Data: download BRFSS+ATUS, harmonize, fuse, cluster, preferences, persona cards; test 11; fusion report.
- **M6** Virtual-RCT calibration → `configs/calibrated.yaml`.
- **M7** Experiment configs E1–E10, runner (parallel, resume), analysis, plots; smoke run with the mock.
- **M8** Real runs: legacy, then E8 → E1 → E2 → E3 → E4 → E6 → E7 → E9 → E10; `RESULTS.md`.
- **M9 (optional)** C11 PPO; the NSHAP adapter once the data are provided.

## 15. Rules that protect validity
- Tune parameters only on dev seeds (0–9, 42/100/200). Never on eval seeds.
- Do not tune preference mappings or dynamics to favor any method. Calibration targets are
  only those in §8.
- All controllers see the same observables; only the oracle sees hidden state.
- The mock backend is never used for reported numbers; the report refuses runs whose
  manifest says `backend=mock`.
- Every deviation from this spec goes into `DECISIONS.md` with a reason.
