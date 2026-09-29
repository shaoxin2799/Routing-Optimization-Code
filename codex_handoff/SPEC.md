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


### 0.3 Revision v2: authoritative, supersedes any conflicting text below

**Headline of the paper:** *Adaptive policies observe voices, not preferences.* When feedback is
selective, adaptation can over-represent those who speak. We separate (i) who speaks,
(ii) what is said and how it is interpreted, (iii) how evidence is weighted, and (iv) which
welfare criterion is applied, under a fixed budget. This makes the representation gap
measurable and controllable.

#### 0.3.1 Frozen variable layers and information boundary
Every variable belongs to exactly one layer. `Observation` may contain only variables marked
**O**. Evaluator-only (**E**) and hidden (**H**) variables must never reach a controller except
C10 (oracle).

| Layer | Variables | Visibility |
|---|---|---|
| L0 Data | BRFSS person attributes X_i; ATUS activity profile A_i; NSHAP network statistics ψ, UCLA-3 distribution, long-run persistence r5; literature effect ranges | build-time only |
| L1 Latent state Z_i | ℓ_i(t), s_i, f_i, ties/network, A_i, preferences (μ_i, λ_i, τ_i), response propensity, silence regime parameters, exaggerator flag, persona card | **H** |
| L2 Expression | R_{i,w} (spoke or not), message y_{i,w} | R and y: **O**; felt signals: **H**; gold labels (true_visit_pref, true_event_pref, true_satisfaction): **E** |
| L3 Observable evidence O_t | age band, sex, lives_alone, mobility_limit, cog_limit, segment id, screening ℓ_obs, attendance, visits allocated/accepted/declined, weeks since feedback, parsed signals (derived), screening flags (derived) | **O** |
| L4 Aggregation | r̂_g, weights w_g, D_g, ŝ_g, welfare criterion, weight cap | **O** (derived) |
| L5 Policy | n_e, π_g, allocations, budget B | **O** |
| Evaluation | u_i^w, W, W_min, W_silent, burden, alignment, parse accuracy | **E** |

A test (`test_information_boundary`) walks every controller input and asserts that only
L3–L5 fields are present.

#### 0.3.2 Data roles (replaces §3.3–3.6 where they conflict)
- **BRFSS = population backbone.** Wording in code, docs and the paper: "older adults in
  jurisdictions that fielded the SD/HE module". Never "nationally representative loneliness".
  Record which states and years are included.
- **ATUS = activity behaviour, not preference.** Rename `sa` → `activity_profile` (vector: social
  minutes, religious/civic/volunteer minutes, share of time alone, share of time with
  non-household others).
  - ATUS directly calibrates **behaviour**: the baseline event-attendance propensity
    b_i = logit of the fused social-participation percentile (it enters attendance, §5.4).
  - **Preferences are latent**: μ_i = f(A_i, X_i; φ). φ has regimes **weak / medium / strong**
    (coefficients ×0.5 / ×1 / ×1.5 of M0 in §4.5) plus **random** (permutation). Report all
    regimes (E9).
  - Attendance becomes `P = σ(b_i + κ_μ·μ_i − 1.5 f_i + 0.8 n_e/B_e)` with κ_μ = 1.0.
- **NSHAP = social structure + long-run persistence** (when the R3 files are provided; the
  pipeline must still run without them):
  - **Network**: estimate ψ from the R3 social-network roster: roster size distribution,
    kin/non-kin shares, contact-frequency distribution per tie type, and closeness. Generate
    a **synthetic** network G ~ 𝒢(ψ). Wording: "network statistics calibrated to NSHAP",
    never "the network is taken from NSHAP". Tie types set w_ij and daily contact
    probabilities (map the reported contact frequency to a daily probability).
  - **Loneliness construct**: UCLA-3 (lack companionship, left out, isolated) → score
    distribution by strata (age band × lives alone × functional limitation). Used in E8b.
  - **Persistence**: with R1–R3 (downloaded later), estimate the 5-year correlation r5 of the
    UCLA-3 score. Interpret r5 as the stable-trait variance share. Set the initial
    decomposition ℓ_i(0) = ℓ̄_i + transient, with Var(ℓ̄)/Var(ℓ) = r5, and set σ_ℓ so that
    the stationary transient variance equals (1−r5)·Var(ℓ). **The daily reversion α is NOT
    identified by NSHAP.** Keep α as a sensitivity parameter {0.01, 0.02, 0.05}. Before R1/R2
    arrive, use r5 = 0.5 and flag it.
  - Keys: `SU_ID` links rounds; the weight is `WEIGHT_ADJ` (round-specific).
- **Intervention effects** stay as literature ranges: β_visit ∈ {low, mid, high}, via virtual-RCT
  targets d_v ∈ {0.1, 0.2, 0.4}. Main conclusions must be checked across all three (E9).

#### 0.3.3 Silence as a controlled mechanism (replaces §5.6)
`P(R_{i,w}=1) = σ(α0 + α_g[seg_i] + α_L·ℓ_i + α_F·f_i + α_C·cog_i + α_P·dissat_i)`, where
dissat_i = clip(−(u_i^{w−1} − u_i^{w−2})/0.05, −1, 1) (a recent drop in own utility).
α0 is solved per regime so that the mean response rate is 0.5 at t=0.

| Regime | Setting | Meaning |
|---|---|---|
| S0 MAR-segment | α_g ~ spread of ±0.8 across segments; α_L=α_F=α_C=α_P=0 | response depends only on the observable group (IPW is correct) |
| S1 need-dependent (default) | α_L = −κ, α_F = −0.5κ, α_C = −0.5κ, with silence strength κ ∈ {0, 1, 2, 3} | lonelier/frailer people speak less (MNAR) |
| S2a dissatisfied-silent | α_P = −1.5 | unhappy residents disengage |
| S2b dissatisfied-loud | α_P = +1.5 | complainers dominate |

The paper must call these *controlled missingness mechanisms*, not estimates of real response rates.

#### 0.3.4 IPW safeguards (extends §7.4)
- Stabilized weights w_g = (p̄ / r̂_g), where p̄ is the overall response rate, capped at
  w_max ∈ {3, 5, ∞}; the default is 5.
- Log the effective sample size per segment. If fewer than 3 responders and no flags,
  shrink D_g toward the population mean D̄ with weight k/(k+3), where k is the evidence count.

#### 0.3.5 New and renamed experiments (replace the E8 row; add E4b; extend E3)
- **E3** now crosses silence regime {S0, S1(κ=0..3), S2a, S2b} × {C4, C5, C6, C7r, C8}.
- **E4b Interpretation × weighting (new)**: inject parse errors (flip each parsed label with
  probability ε_p ∈ {0, 0.1, 0.2, 0.3}) × weighting {naive (C6), IPW w_max=∞, IPW w_max=5,
  IPW w_max=3} × silence {S0, S1(κ=2), S2a} → metrics W_silent, W_min, policy error
  ‖π − π_oracle-share‖₁. Question: does IPW amplify interpretation error?
- **E8 Persona state-expression fidelity** (renamed; replaces "persona validity"):
  - E8a state recovery: hidden state → persona text → parser → recovered state. Report
    Spearman ρ and monotonicity of recovered loneliness vs hidden ℓ, and macro-F1 of the
    preference labels.
  - E8b distributional calibration: the synthetic UCLA-3-style score (parser reads a persona's
    answer to the three UCLA-3 items) vs the NSHAP R3 distribution, stratified by X. Before NSHAP
    is available, use the BRFSS single item.
  - E8c cross-model matrix: generator ∈ {Qwen3-8B, Llama-3.1-8B (or phi-4), Mistral-Small-24B} ×
    interpreter ∈ the same set, so that the pipeline is never read back only by its own family.
  - Wording: "tests whether persona-generated feedback preserves the intended latent-state
    distributions across models and strata". Never "persona validity" or "realistic residents".


#### 0.3.6 Alignment with `plans/EXPERIMENT_PLAN.md`
The scientific protocol (gates G0–G5, `prereg-v1` freeze, primary endpoints, E1b factorial
ablation, the extra conditions C7r-noIPW / C7r-noScreen, IPW weight-cap variants, the
representation ratio ρ_rep, manipulation gain, policy error) is defined in
`plans/EXPERIMENT_PLAN.md`; implement them as defined there. Eval seeds are reserved as 1000–1049.


#### 0.3.7 Controller map "M1": from observable evidence to policy target (authoritative; replaces §7.3–7.4)

All quantities below are computed from L3 observables only. Weeks are indexed by w; the
evidence window is the last H = 4 weeks, with age weights a_0..a_3 = (1, 0.75, 0.5, 0.25).

**Step 0: planning constants (known to the planner, fixed before runs).**
β̂_v = the calibrated *mid*-regime visit effect (so the planner is misspecified in the
low/high effect regimes; this is intended); k = (1/7)Σ_{d=0..6}(1−α̂)^d with α̂ = 0.02;
c_I = λ_max/2 (the prior intrusion cost). Planner constants never read the simulator's true values.

**Step 1: response rates (window + shrinkage).**
p̄_w = (number of speakers over the last H weeks) / (H·N).
r̂_g = (S_g + 2·p̄_w) / (H·n_g + 2), where S_g = number of (resident, week) responses in
segment g over the window. Floor r̂_g ≥ 0.05. Stabilized weight w_g = p̄_w / r̂_g.

**Step 2: per-resident loneliness estimate ℓ̂_i.**
- Base: ℓ_obs_i, the latest 4-weekly screening.
- Responders (spoke in the window): ℓ̂_i = clip(ℓ_obs_i + γ_s·Σ_a a·conf·(0.5 − sat) / Σ_a a, 0, 1),
  with γ_s = 0.2 (this uses the parsed satisfaction and confidence of their messages).
- Silent-risk uplift (C7 only): if the LLM screen flags i, ℓ̂_i ← min(1, ℓ̂_i + Δ_r), where
  Δ_r = 0.15 (high) or 0.07 (medium).

**Step 3: per-resident intrusion-risk estimate q̂_i ≈ P(one more visit is unwanted).**
Segment prior q̄_g (Step 4). Evidence counts over the window, with age weights:
L_i = Σ a·(#"less" visit_pref + #declines); M_i = Σ a·(#"more" visit_pref + 0.5·#accepted visits).
q̂_i = (2·q̄_g + L_i) / (2 + L_i + M_i). For a second visit in the same week:
q̂_i^(2) = 1 − (1 − q̂_i)².

**Step 4: segment prior with non-response weighting (this is where IPW and the cap act).**
q_i^resp = L_i / (L_i + M_i) for residents with L_i + M_i > 0 in the window (set R_g).
Raw segment mean q̄_g^raw = mean_{i∈R_g} q_i^resp (Hájek). Population mean
q̄_pop = Σ_g n_g q̄_g^raw / Σ_g n_g (**population-weighted**, not voice-weighted).
Shrinkage factor s_g = min(1, w_max·r̂_g / p̄_w) · |R_g| / (|R_g| + 3).
q̄_g = s_g·q̄_g^raw + (1 − s_g)·q̄_pop. Uncapped IPW: w_max = ∞ (only the evidence term remains).
If |R_g| = 0: q̄_g = q̄_pop. Global prior when nobody has spoken yet: q̄_pop = 0.3.

**Step 5: predicted marginal welfare of a visit (observable utility prediction).**
m̂_i^(1) = 2β̂_v·k·ℓ̂_i·(1 − q̂_i) − c_I·q̂_i
m̂_i^(2) = 2β̂_v·k·ℓ̂_i·(1 − q̂_i^(2)) − c_I·q̂_i^(2)
(the same functional form as the hidden gold-label marginal utility §5.5, with hidden
quantities replaced by observable estimates).
Predicted welfare level: û_i = −ℓ̂_i² − c_I·q̂_i·V_i^{w−1}; segment level û_g = mean_{i∈g} û_i.

**Step 6: welfare criterion → priority weights ω_i** (the only place the criterion enters)
- utilitarian: ω_i = 1
- Nash: ω_i = 1 / max(0.05, 1 + û_i)
- Rawlsian (soft-min over segments): ω_i = exp(−κ_R (û_{g(i)} − min_h û_h)), with κ_R = 20.

**Step 7: target shares and usage (Lipschitz by construction).**
Z = Σ_j Σ_{v=1,2} [ω_j m̂_j^(v)]_+ .
If Z < Z_min (= 1e−4): keep the previous shares (log `rule=hold_low_signal`). Otherwise
π*_g = Σ_{i∈g} Σ_v [ω_i m̂_i^(v)]_+ / Z.
Target usage u* = min(1, #{(i,v): m̂_i^(v) > 0} / B_v). The budget is a cap, not an obligation.

**Step 8: bounded update.** π ← Proj_simplex(π + clip(π* − π, −δ, δ)), with δ = 0.05;
u ← u + clip(u* − u, −0.1, 0.1).

**Step 9: allocation.** Slots = round(u·B_v). Segment quotas come from largest-remainder
rounding of π·slots. Within each segment, candidate (i,v) pairs are ranked by ω_i m̂_i^(v)
(ties by pid), and **only pairs with m̂ > 0 are assigned**; at most v_max = 2 per resident.
Unfilled quota goes to the global pool (same ranking); still unfilled slots stay unused.
Visits are placed Monday to Friday in rank order.

**Step 10: events.** ē_g = the Hájek mean of the parsed event_pref over the window (same
shrinkage as Step 4). Population demand Ê = Σ_g n_g ē_g / N. If Ê > 0.15 then n_e += 1; if
Ê < −0.15 then n_e −= 1; clip to [0, B_e].

**Step 11: audit.** Log per week: r̂_g, w_g, s_g, q̄_g, all ω-weighted segment totals, π*,
π, u, slots, rule ids (`hold_low_signal`, `cap_clip`, `event_up`/`event_down`), and per
allocated visit (i, v, ω_i, m̂_i^(v)).

**Variants (all use the same code path with switches):**

| Variant | Switches |
|---|---|
| C7u / C7r / C7n | ω per Step 6; IPW on (w_max = 5); screening on |
| C7r-noIPW | q̄_pop and Ê are **voice-weighted** (pooled over all responders); s_g uses the evidence term only |
| C7r-noScreen | Δ_r = 0 |
| C7r-IPW∞ / IPW3 | w_max = ∞ / 3 |
| **C6 (response-only, pooled)** | noIPW + noScreen |
| **C6q (request-driven service)** | only residents with ≥ 1 "more" request in the window are candidates, ranked by the latest urgency; no screening. This is a realistic "visits on request" baseline |
| C4 / C5 | same M1 with the parser replaced (survey answers / keyword rules) |
| C8/C9 | the black-box LLM outputs (n_e, π, u); Step 9 allocation is reused with its π |

**Consequences for the propositions:** P2 holds for π* (Step 7). π* is Lipschitz in (ℓ̂, q̂)
on {Z ≥ Z_min}, with a constant bounded by (max ω)·(2β̂_v k + c_I)/Z_min, and Step 8 caps any
change at δ. The within-segment ranking in Step 9 is discontinuous; P2 is stated for the
target and the shares, not for individual assignments.
**The screening interval** (default 4 weeks) is an E9 factor {2, 4, 8}: sparse screening
makes voice matter more.


#### 0.3.8 Frozen language channel and parallel execution (authoritative; supersedes §6.4 caching and §9 run order)

**A. Message bank (persona expression).**
- Cell key = (hidden persona cluster, communication style, lives_alone, mobility_limit,
  felt-signal combination). Felt signals after noise/exaggeration are discrete:
  loneliness {high, mid, low} × trend {better, worse, none} × visits {too_many, want_more, none}
  × events {uncomfortable, want_more, too_many, none} = 108 combinations.
- **Enumerate the full product** (not only "reachable" cells). Generate m = 6 messages per cell
  with the persona model (T = 0.7, a per-cell stable seed); cells × m ≈ 7×5×2×2×108×6 ≈ 90k
  short generations (≈ 5M tokens).
- **Bank A** = primary generator family (Qwen3-8B). **Bank B** = second family (Llama-3.1-8B or
  phi-4), for validation and E7.
- At runtime a message is selected by keyed RNG (seed, pid, week, "bank"). **A missing cell is
  a hard error**; online generation is never used as a fallback in bank mode.
- UCLA-3 answers for E8b are banked in the same way (cluster × style × loneliness level).

**B. Parse bank.** Every bank message is parsed once by every interpreter family (T = 0.1, a
fixed seed). Store the raw output, the validated fields, and the retry/fallback flags. At
runtime, parsing is a lookup. E4b perturbs these stored labels offline.

**C. Screening and black-box calls (online).**
- The screen input is discretized (ℓ_obs to 0.05, changes to 0.05, integer counts, booleans)
  and content-addressed: key = hash(model revision, prompt template hash, discretized input,
  sampling params, sampling seed).
- One **shared, single-flight cache** (SQLite in WAL mode or Redis): concurrent requests for the
  same key wait for one call.
- C8/C9 prompts are also content-addressed; with the sampled messages taken from the bank,
  their keys repeat across seeds rarely, so they are effectively online.
- **Reproducibility is guaranteed by the frozen cache artifact**, which is shipped with the
  results, not by re-running inference (vLLM is not batch-invariant).

**D. Caching rule (replaces AGENTS rule 6).**
- **Allowed**: the message and parse banks built before the freeze; content-addressed
  memoization where the key contains every input that affects the output.
- **Forbidden**: returning an output for a different input; replaying outputs recorded under
  another key; filling bank misses online.

**E. Gate G6: language-channel validation (dev seeds 0–9, before the freeze).**
Run C6, C7r and C8 twice: in **live mode** (online persona generation plus online parsing) and
in **bank mode**. G6 passes if:
1. the sign of ΔW_silent(C7r − C6) and ΔW(C7r − C8) agrees in both modes, and the bank-mode
   estimate lies inside the live-mode 95% CI;
2. parse macro-F1 by style differs by ≤ 0.05 between modes;
3. Bank A and Bank B give the same sign on these contrasts.

If G6 fails, enlarge m or add key dimensions and repeat. **Do not freeze before G6 passes.**
The G0–G6 gate report requires a **human sign-off** before the tag `prereg-v1` is created.

**F. Execution model.**
- Job = (experiment, condition, cell-params, seed). Jobs are idempotent and resumable. The
  output directory is `results/<exp>/<cond>/<cellhash>/<seed>/`, written to a temp dir and
  atomically renamed. A job manifest records status, host, timing and cache hit rates.
- **Randomness**: a counter-based generator (numpy Philox), keyed by
  (seed, stream, agent, day). Common random numbers (CRN) therefore hold regardless of execution
  order or process.
- **Hot loop**: numpy arrays and a sparse adjacency matrix; networkx only for build and metrics.
- **Job classes and priority**:
  - A = CPU-only: C0, C1, C2, C10, E4b offline, metrics, bootstrap;
  - B = light GPU: bank-mode C3–C7 with screening calls;
  - C = heavy GPU: C8/C9, E6 re-sampling, live validation, E7 black-box.
  - GPU queue priority: primary E1/E3 > primary E2/E4b > secondary > E7/E9. Class C gets a
    concurrency cap so it cannot starve the short screening calls.
- **GPU router**: after the banks are built, every GPU runs a general inference replica (the
  interpreter/screen model and the black-box model share replicas when memory allows). The
  client does least-outstanding-requests routing across replicas. No fixed "one GPU per role".
- **E7**: interpreter-family generality = re-parse the bank offline with each family (batch),
  then CPU-only closed loops. Generator-family generality = Bank B. Only C8/C9 with other
  families need online model swaps; schedule them on whichever GPU frees first. The 72B-AWQ
  model (TP = 2) runs last.
- **Concurrency is measured, not hard-coded**: a benchmark job runs 16 dev trajectories and
  records CPU-seconds per trajectory, RSS, GPU requests/s and cache hit rate. The scheduler
  then sets `world_workers = min(cores / threads_per_job, 0.8·RAM / RSS, GPU-limited rate)`
  and logs the decision. Start around 64–96 and allow up to 128 when measurements permit.

**G. Critical path and expected wall-clock time** (4×A100 + 256 CPU; if 2 GPUs, the
GPU-bound phases take roughly ×2):
1. 0–1.5 h in parallel: tests; data build and fusion (CPU); virtual-RCT calibration and the
   C2 grid on dev seeds (CPU); bank A/B generation (GPU), pipelined with parsing.
2. 1.5–3 h: dev pilot, gates G0–G6 (including live-vs-bank validation) → human sign-off →
   `prereg-v1`.
3. 3–7 h: all eval experiments sharded in parallel (E1, E2, E3, E4, E4b, E6 snapshots, E9, E10).
4. 7–10 h: E6 re-sampling, E7 online C8/C9 swaps, retries, independent metric recomputation,
   bootstrap, `export_tex.py` with all checks.

Target: core primary data in ~4–6 h after the start, and the full suite in ~8–12 h. This is
verified by the benchmark job, not assumed.

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
| Loneliness reversion α | 0.02/day | {0.01, 0.02, 0.05} | not identified by NSHAP (5-year spacing); sensitivity only. NSHAP constrains r5 (§0.3.2) |
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
| E4b interp × weighting | RQ4/RQ2 | see §0.3.5 | C6, C7r variants | 20 |
| E4 robustness | RQ4 | η ∈ {0, 0.2, 0.4} × x ∈ {0, 0.1, 0.3} | C7r, C8, C9 | 20 |
| E5 welfare criteria | RQ3 | taken from E1 | C7u, C7r, C7n | (E1) |
| E6 audit | RQ3 | (a) replay the same run from its LLM log → the policy must be identical; (b) 5 fresh re-samples of the LLM at T=0.1 and at T=0.7 → policy variance; (c) leave-one-feedback-out counterfactual influence | C7r, C8 | 20 |
| E7 models | generality | persona ∈ {Qwen3-8B, Llama-3.1-8B/phi-4} × diagnosis ∈ {Llama-3.1-8B/phi-4, Qwen3-14B, Mistral-Small-24B}, plus 72B diagnosis | C7r, C8 | 10 |
| E8 state-expression fidelity | validity | see §0.3.5 (E8a state recovery, E8b distributional calibration vs NSHAP/BRFSS, E8c cross-model matrix) | persona + parse | 5 |
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
15. `test_information_boundary`: every controller input contains only L3–L5 fields (§0.3.1).
16. `test_ipw_weight_cap`: stabilized weights never exceed w_max; shrinkage is applied when evidence < 3.
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

---

# Appendices: operational details (use these verbatim unless verification shows otherwise)

## A. Models: where to get them and how to download

All models come from Hugging Face. Pages: `https://huggingface.co/<model_id>`.

| Key | HF model id | Page | Gated? | Approx. disk (bf16) | Role |
|---|---|---|---|---|---|
| qwen3-8b | `Qwen/Qwen3-8B` | https://huggingface.co/Qwen/Qwen3-8B | no | ~16 GB | persona (default) |
| llama31-8b | `meta-llama/Llama-3.1-8B-Instruct` | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | **yes** (accept licence on the page, then use HF_TOKEN) | ~16 GB | diagnosis/black-box (default) |
| phi4 | `microsoft/phi-4` | https://huggingface.co/microsoft/phi-4 | no | ~29 GB | diagnosis fallback |
| qwen3-14b | `Qwen/Qwen3-14B` | https://huggingface.co/Qwen/Qwen3-14B | no | ~28 GB | E7 |
| mistral-24b | `mistralai/Mistral-Small-24B-Instruct-2501` | https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501 | check page | ~47 GB | E7 |
| qwen25-72b-awq | `Qwen/Qwen2.5-72B-Instruct-AWQ` | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ | no | ~41 GB | E7 large (TP=2) |
| llama33-70b | `meta-llama/Llama-3.3-70B-Instruct` | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct | **yes** | ~140 GB (use FP8 via `--quantization fp8`) | E7 large alternative |

Download (on server 2; models go to `$HF_HOME`, default `~/.cache/huggingface`; put it on a large disk):
```bash
pip install -U "huggingface_hub[cli]"
export HF_HOME=/data/hf            # adjust to a disk with >= 300 GB free
huggingface-cli login              # only needed for gated Llama models (paste HF token)
for m in Qwen/Qwen3-8B microsoft/phi-4 Qwen/Qwen3-14B \
         mistralai/Mistral-Small-24B-Instruct-2501 Qwen/Qwen2.5-72B-Instruct-AWQ \
         meta-llama/Llama-3.1-8B-Instruct; do
  huggingface-cli download "$m" --exclude "*.pth" "original/*" || echo "SKIP $m (gated or unavailable)"
done
```
(`hf download <id>` is the newer equivalent CLI; either is fine.) Record `git rev` of each model
(`huggingface_hub.model_info(id).sha`) in the run manifest.

If a gated model is unavailable, `configs/models.yaml` falls back automatically: llama31-8b → phi4,
llama33-70b → qwen25-72b-awq. Log the substitution in the manifest and in DECISIONS.md.

Legacy option (only for the MABS replication, optional): Ollama `llama3:8b` via
`ollama pull llama3:8b`; the OpenAI-compatible endpoint is `http://localhost:11434/v1`.

## B. vLLM serving commands (`scripts/serve_vllm.sh`)
```bash
pip install vllm          # record the installed version (e.g. `python -c "import vllm;print(vllm.__version__)"`)

# persona on GPU0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8000 \
  --served-model-name persona --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 --seed 0

# diagnosis + black-box on GPU1
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 \
  --served-model-name diag --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 --seed 0

# E7 large: 72B-AWQ across both GPUs + persona squeezed on GPU0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8000 --served-model-name persona \
  --gpu-memory-utilization 0.30 --max-model-len 4096
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --port 8001 \
  --served-model-name diag --tensor-parallel-size 2 --gpu-memory-utilization 0.60 \
  --max-model-len 8192 --quantization awq
```
Health check: `curl localhost:8000/v1/models`. The client reads endpoints from `configs/models.yaml`.

**Structured output (version-dependent; implement both and auto-detect at startup):**
- newer vLLM: `extra_body={"structured_outputs": {"json": <schema>}}`
- older vLLM: `extra_body={"guided_json": <schema>}`
- fallback: `response_format={"type":"json_schema","json_schema":{"name":"out","schema":<schema>}}`
Qwen3 thinking off: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
Per-request seed: `extra_body={"seed": int}` (derive with `hash((run_seed, week, pid, role)) % 2**31`
via a stable hash, e.g. blake2b, not Python `hash`).

## C. Environment file
```yaml
# environment.yml
name: apabm
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
      - numpy>=1.26
      - pandas>=2.2
      - pyarrow>=15
      - networkx>=3.2
      - scipy>=1.12
      - scikit-learn>=1.4
      - statsmodels>=0.14
      - pydantic>=2.6
      - pyyaml>=6
      - httpx>=0.27
      - openai>=1.30
      - tenacity>=8.2
      - jinja2>=3.1
      - tqdm
      - rich
      - matplotlib>=3.8
      - seaborn>=0.13
      - beautifulsoup4     # parse data index pages
      - pytest
      - pytest-xdist
      - hypothesis         # property tests
      - ruff
      - huggingface_hub[cli]
# vLLM is installed separately into the same env: pip install vllm  (needs CUDA 12 driver)
```

## D. Data: exact sources and download procedure

### D.1 BRFSS (no login)
| Item | URL |
|---|---|
| Documentation hub | https://www.cdc.gov/brfss/data_documentation/index.htm |
| 2022 data & docs | https://www.cdc.gov/brfss/annual_data/annual_2022.html |
| 2022 main XPT (64 MB zip, 326 vars) | https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip |
| 2022 multiple-questionnaire version files (optional modules) | https://www.cdc.gov/brfss/annual_data/2022/llcp_multiq.html |
| 2022 questionnaire (SD/HE module text) | https://www.cdc.gov/brfss/questionnaires/pdf-ques/2022-BRFSS-Questionnaire-508.pdf |
| 2023 data & docs | https://www.cdc.gov/brfss/annual_data/annual_2023.html |
| 2024 data & docs | https://www.cdc.gov/brfss/annual_data/annual_2024.html |

Procedure (`apabm/data/download.py::brfss`):
1. GET the year's index page; collect every href ending in `.zip` whose text or name contains
   `XPT` and `LLCP`; also the codebook (`codebook*.html`/`.zip`/`.pdf`) and the variable layout.
2. GET the `llcp_multiq.html` page; download the version XPT files (names like `LLCP22V1_XPT.zip`,
   `LLCP22V2_XPT.zip`; verify) — optional modules such as SD/HE may live there together with
   version-specific weights (`_LCPWTV1`, `_LCPWTV2`, …; verify).
3. Load with `pandas.read_sas(path, format="xport")`. Search all loaded files for `SDLONELY` and
   `EMTSUPRT`; use the file(s) where they are non-missing, with the matching weight.
4. Write `data/interim/brfss_<year>.parquet` and append to `data/MANIFEST.json`.
Reference implementation ideas for parsing BRFSS/ATUS: https://asdfree.com/behavioral-risk-factor-surveillance-system-brfss.html
and https://asdfree.com/american-time-use-survey-atus.html (R code; use only as a guide).

### D.2 ATUS (no login; needs User-Agent with contact e-mail)
| Item | URL |
|---|---|
| ATUS data hub | https://www.bls.gov/tus/data.htm |
| Multi-year 2003–2024 page | https://www.bls.gov/tus/data/datafiles-0324.htm |
| Multi-year 2003–2025 page (use if present) | https://www.bls.gov/tus/data/datafiles-0325.htm |
| Activity-summary data dictionary (example) | https://www.bls.gov/tus/dictionaries/atussmcodebk0524.pdf |
| Interview data dictionary (example) | https://www.bls.gov/tus/dictionaries/atusintcodebk0324.pdf |

Expected file names on the multi-year page (0324 shown; 0325 analogous):
`atusresp-0324.zip` (respondent), `atusrost-0324.zip` (roster), `atuswho-0324.zip` (who),
`atussum-0324.zip` (activity summary), `atuscps-0324.zip` (CPS), `atusact-0324.zip` (activity).
Resolve the actual hrefs by parsing the page (do not hard-code the path prefix). Each zip holds a
CSV (`.dat`) plus SAS/SPSS/Stata readers.

Key fields (verify in dictionaries): `TUCASEID` (join key), `TUYEAR`, `TEAGE`, `TESEX`,
`TUFINLWGT` (final weight), roster `TULINENO`/`TERRP` (household composition → lives alone when
only the respondent is listed), who-file `TUWHO_CODE` (codes for "alone" = 18/19 — verify),
activity-summary columns `t12xxxx` (socializing, relaxing, leisure — keep socializing/communicating
`t1201xx` and attending events `t1202xx`, verify), `t14xxxx` (religious), `t15xxxx` (volunteer),
CPS disability items `PEDISPHY`, `PEDISOUT` (available 2008+; verify).

### D.3 What is deliberately NOT downloaded this round
NSHAP (https://www.icpsr.umich.edu/sites/icpsr/view/collections/706), HRS
(https://hrs.isr.umich.edu/data-products), SHARE (https://share-eric.eu/data/data-access),
CHARLS (https://charls.pku.edu.cn/en), ELSA (https://www.elsa-project.ac.uk/accessing-elsa-data)
— all require login/registration. Only the `NshapSource` stub is built.
Known NSHAP facts (from the ICPSR collection page) for the later adapter: person id `SU_ID` links
respondents across rounds; each round has a cross-sectional weight `WEIGHT_ADJ` (round-specific);
for longitudinal analyses use the Round 2 `WEIGHT_ADJ` until a panel weight exists. Study pages:
R3 https://www.icpsr.umich.edu/web/ICPSR/studies/36873/versions/V9 (public DS1 core, DS3 social
networks, DS11 COVID; Stata format), R2 https://www.icpsr.umich.edu/web/ICPSR/studies/34921/versions/V5.
Round 4 (study 39511) is restricted-only and out of scope.

## E. Algorithms (pseudocode)

### E.1 Run loop
```
init: pop = sample_population(seed); net = build_network(pop, seed); state = init_state(pop)
policy = controller.initial_policy()
for day in 0..T-1:
    if day >= warmup and day % 7 == 0:            # decision point (week w)
        obs = env.observe(state, history)           # Observation only
        if env.feedback_enabled:
            speakers = voice.sample(state, rng["voice"])
            msgs = persona.generate(speakers, state, week_summary, rng["feedback_noise"])  # LLM-A
            state.history.feedback[w] = msgs          # gold labels saved separately (hidden)
        policy, audit = controller.step(obs, msgs_visible)   # LLM-B inside if needed
        schedule = interventions.schedule_week(policy, obs, budget, rng["alloc"])
    env.step_day(state, schedule[day % 7], rng)     # order below
    if day % 28 == 0: screening.update(state, rng["screen"])
compute weekly utility, gold labels, metrics
```
Daily step order (fixed): (1) events of the day → attendance draws → tie formation among attendees;
(2) visits of the day → decline draws → accepted visits; (3) informal interactions over ties and
household; (4) state updates ℓ, s, f (§5.2) using today's a_i, v_i, x_i, c_i; (5) tie decay; (6) log.

### E.2 Named RNG streams
`streams = SeedSequence(seed).spawn` keyed by a stable hash of the name. Shared across conditions
(common random numbers): `population, network0, dynamics, screen, attendance, decline, voice,
feedback_noise, network_dyn`. Condition-specific: `alloc, controller`. Each draw consumes from its
own stream only, so different policies do not shift the other streams.

### E.3 Largest-remainder quota + allocation (§7.3)
```
raw = π_g * B_v; q_g = floor(raw); give remaining slots to largest (raw - q_g), ties by segment id
for g in segments (fixed order): rank members by p_i desc (ties by pid); assign up to v_max each
leftover slots → global ranking over all residents not yet at v_max
visits are placed on weekdays round-robin (Mon..Fri) in rank order
```

### E.4 Bounded update + simplex projection
```
Δ = clip(π* - π, -δ, δ); π' = π + Δ
π' = project_to_simplex(π')     # Duchi et al. 2008 sort-based Euclidean projection
if max|π' - π| > δ + 1e-9: scale (π' - π) down uniformly so the cap holds, re-project; log "cap_clip"
```

### E.5 IPW / Hájek estimates (C7) vs naive (C6)
```
r̂_g ← 0.3 * |R_g|/n_g + 0.7 * r̂_g ; r̂_g = max(r̂_g, 0.05)
S_g = R_g ∪ Flagged_g
D_g = mean_{i∈S_g} d_i ; ŝ_g = mean_{i∈S_g} sat_i          (segment means)
population event demand E = Σ_g n_g * Ē_g / N                (weights by population)
naive: pool all responders, weights by |R_g| (voices), no flagged set
```
(The Horvitz–Thompson form Σ_{i∈R_g} y_i / r̂_g / n_g is logged too, for the IPW unit test.)

### E.6 Persona-cluster GMM
```
X = standardize(features) on a 20k weighted resample (rng "population", fixed seed 12345)
for K in 3..7: fit GaussianMixture(K, covariance_type="full", n_init=5, random_state=0); record BIC
K* = argmin BIC; if two BICs within 1%, pick smaller K
assign each person to argmax posterior; save means, covariances, proportions, BIC table
```

### E.7 Predictive mean matching (ATUS → BRFSS)
```
model_y = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300)
fit on ATUS 65+ with sample_weight = TUFINLWGT; 5-fold CV R² reported
ŷ_atus = cross-fitted predictions; ŷ_brfss = model_y.predict(BRFSS covariates)
for each BRFSS person: donors = 5 ATUS records with nearest ŷ_atus to ŷ_brfss; draw one (rng "fusion")
y_brfss = donor's observed y     (do this separately for sa and tw, using the SAME donor for both
                                   to keep their joint distribution: match on ŷ_sa, ŷ_tw jointly by
                                   Mahalanobis distance)
```

### E.8 Virtual-RCT calibration (β_v; β_e analogous)
```
target d_v; lo, hi = 0.0, 0.2
repeat 30 times (bisection):
    β = (lo+hi)/2
    for s in dev seeds 0..9: simulate 12 weeks, N=400, 1:1 arms, treatment = 1 accepted visit/week
                             (declines disabled in the trial); d_s = (mean ℓ_ctrl - mean ℓ_trt)/pooled SD
    d = mean_s d_s ; if d < target: lo = β else hi = β
freeze β_v = (lo+hi)/2 with report (d per seed, final d)
```

### E.9 Myopic oracle (C10)
```
visits: repeat B_v times: pick resident with max m_v (§5.5, true values, current planned V) with
        m_v > 0 and V < v_max; tie by pid; update planned V
events: for n in 0..B_e: copy env, simulate 7 days with the visit plan and n events using
        cloned RNG streams (2 rollouts), score Σ_i u_i^w; pick best n
```

### E.10 Counterfactual influence (E6c)
For C7: for each week and each received message j, recompute the controller output with message j
removed (parser outputs from the log; no new LLM calls) → influence_j = ||π(with) − π(without)||₁ +
|Δn_e|. For C8, influence needs new LLM calls: sample 20 messages/run, re-query with the message
removed (logged as extra calls).

### E.11 Statistics
- Wilcoxon signed-rank (`scipy.stats.wilcoxon`, paired by seed, two-sided); Holm across the family of
  contrasts in one table; rank-biserial r = (W+ − W−)/(W+ + W−).
- Bootstrap: resample seeds with replacement 10,000 times; percentile 95% CI of the mean paired diff.
- Hypervolume (2D, maximize W, minimize visits): sort non-dominated points; sum rectangles to
  reference point (max visits over all runs, W of C0).

### E.12 PPO (optional C11)
stable-baselines3 PPO; observation = per-segment [n_g/N, r̂_g, D_g, ŝ_g, mean ℓ_obs, declines] + current
policy + week/T; action = Box → softmax shares + discrete n_e via rounding; reward = weekly mean u
(observable proxy is NOT allowed to be the hidden u at deploy time — train on dev seeds with hidden
reward, evaluate on eval seeds; state this caveat in the paper). 2e5 env steps, default hyper-params.

## F. Remaining concrete definitions

### F.1 Felt-signal thresholds (persona input)
| Signal | Condition (true state) | Phrase |
|---|---|---|
| lonely_high | mean ℓ this week > 0.6 | "you felt quite lonely this week" |
| lonely_mid | 0.4–0.6 | "you felt a little lonely at times" |
| lonely_low | < 0.4 | "you felt fairly connected" |
| lonely_trend | Δ mean ℓ vs last week < −0.03 / > +0.03 | "a bit better than last week" / "a bit worse than last week" |
| visits_too_many | V^w > τ | "the visits felt like too much / intrusive" |
| visits_want_more | true_visit_pref = more and V^w ≤ τ | "you would welcome more visits" |
| visits_fine | otherwise | (omit) |
| events_uncomfortable | A^w > 0 and μ < −0.2 | "the group activity was uncomfortable" |
| events_want_more | true_event_pref = more | "you would enjoy more activities" |
| events_too_many | true_event_pref = less and n_e ≥ 2 | "there is too much pressure to join activities" |
Noise: each included signal is dropped with prob η/2 and flipped to its opposite with prob η/2.

### F.2 Keyword lexicon (C5), case-insensitive regex
- visit_more: `\b(more (visits|company|visitors)|come (by|round|over) more|nobody (visits|comes)|would (love|like) (a )?visit|lonely)\b`
- visit_less: `\b(too many visits|(leave|let) me (alone|be)|don'?t need (the )?visits|(stop|fewer) visits|privacy|intrusive|too much)\b`
- event_more: `\b(more (activities|events|outings|groups)|enjoy(ed)? the (group|activity)|love the (group|activities))\b`
- event_less: `\b(don'?t like (the )?(group|activities)|not (a|much of a) (group|social) person|too many activities|pressure to join)\b`
- satisfaction: 0.8 if only positive matches, 0.2 if only negative, 0.5 otherwise.
Conflicts (both more and less matched) → "same".

### F.3 Full prompt: blackbox_informed (C8/C9)
```
[system] You are the policy planner of a community aged-care service. You decide next week's
service plan. Return only JSON matching the schema.
[user]
Goal: maximise residents' well-being — mainly lower loneliness — while respecting what residents
want (do not over-visit people who do not want visits; offer activities people value), and be fair
across groups (do not neglect groups with low satisfaction or residents who do not speak up).
Weekly budget: {B_v} home-visit slots (max {v_max} per resident) and up to {B_e} group events.
{C9 only: You may change each segment share by at most {delta} and the number of events by at most 1 per week.}
Segments (observable groups):
{table: segment_id | description | n | response rate | visit more/same/less | event more/same/less |
 mean satisfaction | mean screening loneliness | declines last week | silent high-risk flags}
Last 4 weeks: {table: week | n_events | shares | mean satisfaction | mean screening loneliness}
Current plan: n_events={n_e}, shares={shares}
Sample of resident messages this week (by segment):
{segment_id: "message" …  (≤30 total)}
Return JSON: {"n_events": int in [0,{B_e}], "segment_shares": {segment_id: float ≥0, summing to 1},
"rationale": "≤ 60 words"}
```
Post-processing: renormalise shares; missing segments get 0; invalid → retry once → previous plan.

### F.4 Full prompt: feedback_parse
```
[system] You are the assessment module of a community aged-care service. Return only JSON.
[user] Resident group: {segment description}.
Message from the resident: "{message}"
Infer from the message only:
- satisfaction (0 = very unhappy, 1 = very happy with the service),
- visit_pref: does the resident want more, the same, or fewer home visits?
- event_pref: more, same, or fewer group activities?
- urgency (0–1): how urgently should staff follow up?
- confidence (0–1).
If the message gives no evidence for a field, use "same" and a low confidence.
```

### F.5 Full prompt: silent_screen
```
[system] You are the screening module of a community aged-care service. Return only JSON.
[user] This resident has not given feedback for {k} week(s). Service record:
age band {age_band}; lives alone: {yes/no}; mobility difficulty: {yes/no};
latest loneliness screening {l_obs:.2f} (0–1, higher = lonelier), change since previous {dl:+.2f};
group activities attended in last 4 weeks: {att}; visits accepted/declined in last 4 weeks: {acc}/{dec}.
Assess the risk that this resident is lonely and would benefit from a visit.
Return {"risk": "low|medium|high", "priority_visit": 0–1, "reason": "≤ 25 words"}.
```

### F.6 legacy_diagnosis and blackbox_legacy
Copy the two templates from the MABS paper, Appendix C, verbatim (reproduced here):
```
You are a diagnostic module for an elderly-care simulation.
Assess the following agent state and return only valid JSON.
Agent state: {loneliness, frailty, stress, energy}
Recent interactions: {interaction_count_7d, social_event_count_7d}
Network position: {degree, relative_degree, isolation_flag}
Required fields: risk_loneliness: low|medium|high; risk_frailty: low|medium|high;
primary_driver: short string; priority_social: number in [0,1]; priority_visit: number in [0,1];
confidence: number in [0,1]
```
```
You control policy parameters in an elderly-care simulation.
Given aggregate statistics {r, p_s, p_v} and current parameters {theta_s, theta_t, theta_p},
return only valid JSON with: theta_s: number in [0.8,1.5]; theta_t: number in [0.4,0.6];
theta_p: number in [0.15,0.5]. Do not include any additional text.
```
Legacy aggregates: diagnosed set H = agents with ℓ > 0.6 at the decision day (log |H|);
r = |{i∈H: risk_loneliness = high}| / N (also log the ratio over |H|); ps, pv = means over H;
if H is empty, no update.

### F.7 Segment merge order
Segment id = `A{lives_alone}{mobility_limit}{age80}` e.g. `A101`. If a cell has < 5 members, merge
it into the cell that differs only in `age80`; if still < 5, into the cell differing only in
`mobility_limit`; ids of merged cells are joined with `+`. Merging is fixed at t=0 per seed.

### F.8 Example experiment config (`apabm/experiments/configs/E1.yaml`)
```yaml
exp: E1
env: {type: budgeted, N: 200, T_days: 182, warmup_days: 14, budget: {visits_per_capita: 0.2, events: 3}}
population: {source: brfss_atus, mapping: M0}          # synthetic | brfss_atus
dynamics: {file: configs/calibrated.yaml}
voice: {v0: 0.5, s_l: 0.6, s_f: 0.3, s_c: 0.3}
feedback: {eta: 0.2, exaggerators: 0.0}
controller_defaults: {delta: 0.05, event_threshold: 0.15, gamma_d: 0.2, gamma_r: 0.2, v_max: 2}
models: {persona: qwen3-8b, diag: llama31-8b}
conditions: [C0, C1, C2, C3, C4, C5, C6, C7u, C7r, C7n, C8, C9, C10]
seeds: {range: [1000, 1029]}
parallel: {max_concurrent_runs: 12}
backend: vllm
```
