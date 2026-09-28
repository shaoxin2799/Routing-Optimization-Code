# Experimental Plan: Selective Feedback and Auditable Preference-Aware Policy Adaptation

Consolidated scientific protocol (v2). It is consistent with `codex_handoff/SPEC.md`, which holds
the implementation details (§0.3 there is authoritative for code). This document defines **what
we test, why, how, and what counts as evidence**, to the standard expected of an AAMAS
main-track paper.

---

## 0. Thesis, research questions and hypotheses

**Thesis.** Adaptive policies observe voices, not preferences. When feedback is selective,
adaptation over-represents those who speak. Separating **participation** (who speaks),
**interpretation** (what is said), **evidence weighting** and **welfare choice**, with bounded,
budget-feasible, logged updates, makes the representation gap measurable and controllable.

| RQ | Question | Hypothesis (directional, pre-registered) | Primary test |
|---|---|---|---|
| RQ1 Representation | Under selective feedback, whose preferences drive adaptation? Can explicit weighting and silent handling correct this? | H1a: response-only aggregation (C6) under-serves silent residents (representation ratio ρ_rep < 1, lower W_silent). H1b: IPW + silent screening (C7r) raises W_silent and moves ρ_rep toward 1 under S0/S1. H1c (expected failure, reported): under dissatisfaction-dependent silence (S2a), IPW alone does not correct the bias | E3 |
| RQ2 Efficiency at equal resources | Is any welfare gain more than just "more visits"? | H2: at matched accepted visits, W(C7r) ≥ W(best static C2*), and C7r ≥ C3 and C8 | E1, E2 |
| RQ3 Interpretation × weighting | Does IPW amplify LLM interpretation errors? | H3: with parse errors, uncapped IPW increases policy error and variance in low-response segments; capped weights reduce this at a small bias cost (the cap trade-off is exploratory) | E4b |
| RQ4 Robustness and manipulation | Noisy or strategic (exaggerated) feedback | H4: the welfare degradation slope for C7r is smaller than for C8; the manipulation gain of exaggerators is bounded under C7r | E4 |
| RQ5 Audit | Are decisions reproducible and attributable? | H5: C7r replay identity = 100%; policy variance under LLM re-sampling is lower for C7r than for C8 | E6 |
| RQ6 Need for an interpreter | Free text + LLM vs keyword parsing vs structured survey | H6: LLM parsing beats keyword parsing on indirect language (F1) and on W_silent; the structured survey is an idealized reference | E1b, E8a |
| Validity | Does persona text preserve the intended hidden states? | Gate: recovered-vs-hidden loneliness Spearman ρ ≥ 0.6 (dev); cross-model F1 gap ≤ 0.10 | E8 |
| Robustness of conclusions | Do the conclusions depend on assumptions? | Conclusions of H1/H2 hold (same sign) across preference-mapping regimes, effect sizes, α, r5, budgets | E9 |

**Primary endpoints** (fixed before the eval runs):
- **P-1** ΔW_silent(C7r − C6) under S1 (κ=2);
- **P-2** ΔW(C7r − C2*_matched), where C2*_matched is the best static policy within ±5% accepted visits on the same seed;
- **P-3** policy variance ratio C8/C7r under re-sampling.

Everything else is secondary or exploratory and labelled as such.

---

## 1. Design principles
1. **Controlled social simulation.** We study mechanisms under explicit, varied assumptions. We
   do not predict real-world effect sizes.
2. **Hidden-state evaluation.** All outcomes are computed from simulator states and hidden
   utilities. No LLM judges outcomes.
3. **Information boundary.** Five layers (latent state → expression → observable evidence →
   aggregation → policy). Controllers see only observable layers; this is enforced by a test.
4. **Common random numbers (CRN) and pairing.** Within a seed, every condition shares the
   population, the initial network and the exogenous noise streams. All comparisons are paired by seed.
5. **Dev/eval separation and a freeze.** Everything is tuned on dev seeds. Configs and analysis
   code are then frozen (git tag `prereg-v1`) before any eval seed runs.
6. **Resource accounting.** Every welfare number is reported together with the resources used.
7. **No result-shaping.** Calibration targets are fixed in advance (§3.4). Preference mappings,
   dynamics and prompts are not tuned to make any method win.

---

## 2. Data-generating design (three layers)

| Layer | Source | Constrains | Must NOT be claimed |
|---|---|---|---|
| **Population grounding** | BRFSS (2022 SD/HE module; 2023–24 where present), age 65+ | joint distribution of age, sex, marital status, living alone, education, general health, functional limitations, depression, emotional support, loneliness (single item), weights | "nationally representative loneliness"; say "older adults in jurisdictions fielding the module" |
| **Social grounding** | ATUS 2015+ (65+); NSHAP R3 network roster | ATUS: **activity profiles** (social / religious-civic / volunteer minutes, time alone, time with others), fused onto BRFSS by predictive mean matching; they calibrate baseline attendance behaviour. NSHAP: roster size, kin/non-kin shares, contact frequency, closeness → synthetic network generator ψ | ATUS measures preferences; the network "is" NSHAP's |
| **Dynamic calibration** | NSHAP R1–R3 UCLA-3 (5-year spacing); intervention literature | the long-run persistence r5 (stable-trait variance share); the UCLA-3 score distribution by strata; intervention effect ranges via virtual RCT | NSHAP identifies daily dynamics; our effect sizes are causal estimates |

**Latent preferences.** μ_i (group-activity preference), λ_i (intrusion aversion) and τ_i
(comfortable visits/week) are generated as f(A_i, X_i; φ), where φ ∈ {weak, medium, strong,
random}. They are reported as an assumption and swept in E9.

**Pre-NSHAP mode.** The whole pipeline runs without NSHAP: a parametric network, r5 = 0.5, and
the BRFSS item for E8b. Results are flagged, and the main results are re-run once NSHAP R3 is in.

### 2.1 Build products and checks
- `persons_65plus.parquet`: harmonized fields, weights, included states and years.
- `fusion_report.json`: CV R² and donor diagnostics for the ATUS → BRFSS match.
- `personas.json`: GMM clusters, with K chosen by BIC.
- `network_stats.json`: ψ, fitted vs NSHAP moments.
- `calibrated.yaml`: β_v and β_e per effect regime, σ_ℓ from r5.

Every check is reported in the supplementary material.

---

## 3. Simulation model (final)

### 3.1 Setting
Community aged-care service, N = 200 (scale runs: 500 and 1000). Daily steps; weekly decisions;
T = 26 weeks after a 2-week warm-up. Budget per week: B_v = 0.2N visit slots (at most 2 per
person) and B_e = 3 group events.

### 3.2 State and dynamics
- ℓ (loneliness), s (stress), f (frailty), a weighted network, and household contact.
- Loneliness: reversion to the individual baseline ℓ̄_i at rate α, minus the effects of informal
  contact, events and visits, plus a stress term and noise. The variance split between ℓ̄ and
  the transient part follows r5.
- **Attendance** `σ(b_i + κ_μ μ_i − 1.5 f_i + 0.8 n_e/B_e)`: b_i comes from the ATUS profile
  (behaviour); μ_i is the latent preference.
- **Visits** can be declined: `P = 0.05 + 0.4 λ_i/λ_max + 0.3·1[V ≥ τ]`. A declined visit uses its slot.

### 3.3 Hidden utility and gold labels
`u_i^w = −mean ℓ² − λ_i·max(0, V − τ_i) + c_A μ_i A`. The squared loneliness term makes a
reduction worth more for lonelier people. Gold labels (evaluation only) are the signs of the
marginal utilities of one more visit or event.

### 3.4 Calibration targets (fixed in advance)
- **Virtual-RCT effect sizes**: visits d_v ∈ {0.1, **0.2**, 0.4}; events d_e ∈ {0.05, **0.15**, 0.3}
  for attenders with μ > 0.
- **Legacy replication**: L-Baseline ≈ 0.72 and L-Fixed ≈ 0.67 (dev seeds only).
- **r5 → σ_ℓ.** Nothing else is calibrated.

### 3.5 Silence (controlled missingness mechanisms)
`P(speak) = σ(α0 + α_g[seg] + α_L ℓ + α_F f + α_C cog + α_P dissat)`, with α0 set so that the
mean response rate at t=0 is 0.5.

| Regime | Setting |
|---|---|
| S0 MAR-segment | response depends only on the observable segment (α_g spread ±0.8) |
| S1(κ) need-dependent | κ ∈ {0, 1, 2, 3}; default κ = 2 |
| S2a dissatisfied-silent | α_P = −1.5 |
| S2b dissatisfied-loud | α_P = +1.5 |

### 3.6 Expression
The persona LLM writes 1–3 sentences from the persona card, the week summary and the felt
signals. Two perturbations:
- **fidelity noise** η ∈ {0, 0.2, 0.4}: signals are dropped or flipped;
- **exaggerators** x ∈ {0, 0.1, 0.3}: a fixed subset always claims high need.

### 3.7 Observable evidence (the controller's view)
Demographic subset, segment, 4-weekly noisy screening ℓ_obs, attendance, visits
allocated/accepted/declined, whether the resident spoke, and the message text. Parsed signals
and screening flags are derived from these.

---

## 4. Methods under test

### 4.1 Our method (C7): a modular pipeline
**Exact controller map ("M1"): see SPEC §0.3.7. It is the authoritative definition of Steps 1–11 below.**
1. **Interpretation**: an LLM parses each message into {satisfaction, visit_pref, event_pref,
   urgency, confidence}.
2. **Silent handling**: an LLM screens non-responders' service records into {risk, priority}.
   Flagged residents are imputed as evidence.
3. **Weighting**: segment Hájek estimates; segments weighted by population n_g; stabilized
   weights capped at w_max (default 5); shrinkage when evidence < 3.
4. **Welfare choice**: utilitarian (C7u), Rawlsian soft-min (C7r, the primary variant), or Nash (C7n).
5. **Bounded update**: |Δπ_g| ≤ δ = 0.05, |Δn_e| ≤ 1, then simplex projection and budget projection.
6. **Allocation**: within-segment ranking by ℓ_obs, parsed preference and risk, respecting
   "less" requests and declines.
7. **Audit log**: inputs, rule id, pre/post values and clips for every change.

### 4.2 Baselines and references

| ID | Role |
|---|---|
| C0 none | lower bound |
| C1 fixed | status quo |
| C2 static grid (incl. Max) | resource frontier; C2*_matched is the resource-matched comparator |
| C3 MABS closed loop | our prior, state-only method |
| C4 structured survey + rules | idealized, no-LLM reference |
| C5 keyword parser + rules | no-LLM interpretation |
| C6 response-only (LLM parse, no IPW, no screening; voice-weighted pooling) | the representation-bias baseline |
| C6q request-driven ("visits on request") | realistic service-model baseline |
| C8 informed black-box LLM (objective, budget, history and sample messages are given; it outputs n_e and shares) | end-to-end alternative |
| C9 C8 + the same step bounds | isolates structure vs bounds |
| C10 myopic oracle (sees hidden state) | upper reference |
| C11 PPO on observables (optional) | learning baseline |

### 4.3 Component ablation (E1b): a factorial around C7r
Factors: interpretation {LLM, keyword, survey} × weighting {naive, IPW-uncapped, IPW-capped} ×
silent handling {none, rule, LLM} × bound {on, off}. A fractional design keeps it tractable
(about 16 cells): all two-factor interactions involving weighting and silent handling, plus the
single-factor removals from C7r.

---

## 5. Experiments

Seeds: dev 0–9; eval seeds are reserved as 1000–1049 (experiments use the first n they need, e.g. 1000–1029 for n=30). Eval seeds are used only after `prereg-v1`.

| Exp | Purpose | Factors | Conditions | Seeds | Output |
|---|---|---|---|---|---|
| **E0 Gates** | validity of the pipeline (§7) | – | all | dev | gate report |
| **E1 Main** | RQ2 + overview | S1(κ=2), η=0.2, x=0, default budget | C0–C10 (C7u/r/n) | 30 | Table 1 |
| **E1b Ablation** | RQ6 + components | §4.3 factorial | C7r variants | 20 | Table 2 |
| **E2 Frontier** | RQ2 | 5 budget levels | C2 grid, C3, C6, C7r, C8, C9, C10 | 20 | Fig. 3 |
| **E3 Representation** | RQ1 | silence {S0, S1 κ=0..3, S2a, S2b} | C4, C5, C6, C7r, C7r-noScreen, C7r-noIPW, C8 | 20 | Fig. 2 |
| **E4 Robustness & manipulation** | RQ4 | η {0, .2, .4} × x {0, .1, .3} | C6, C7r, C8, C9 | 20 | Fig. 4b, Table 3 |
| **E4b Interpretation × weighting** | RQ3 | ε_p {0, .1, .2, .3} × weighting {naive, IPW∞, IPW5, IPW3} × silence {S0, S1κ2, S2a} | C7r variants | 20 | Fig. 4a |
| **E5 Welfare criteria** | stakeholder trade-off | from E1 and E3 | C7u, C7r, C7n | (E1/E3) | Fig. S (W vs W_min) |
| **E6 Audit** | RQ5 | replay; re-sampling at T=0.1/0.7 (5×); leave-one-message-out influence | C7r, C8, C9 | 20 | Table 4 |
| **E7 Model generality** | generality | interpreter ∈ {Llama-3.1-8B or phi-4, Qwen3-14B, Mistral-Small-24B, 72B-AWQ}; persona ∈ {Qwen3-8B, Llama-3.1-8B or phi-4} | C6, C7r, C8 | 10 | Table S |
| **E8 State-expression fidelity** | validity | E8a recovery; E8b distribution vs NSHAP/BRFSS by strata; E8c 3×3 generator × interpreter matrix; styles incl. indirect | persona + parse | 5 | Table S, Fig. S |
| **E9 Sensitivity** | robustness of conclusions | one at a time: φ {weak, medium, strong, random}; d_v {.1, .2, .4}; α {.01, .02, .05}; r5 {.3, .5, .7}; screening interval {2, 4, 8} weeks; δ {.02, .05, .1}; w_max {3, 5, ∞}; event threshold {.1, .15, .25} | C2*, C6, C7r, C8, C10 | 10 | Table S (sign-consistency matrix) |
| **E10 Scale & cost** | practicality | N {200, 500, 1000} | C7r, C8 | 5 | Table S (calls, tokens, wall time) |
| **L Legacy** | continuity with prior work | MABS setting | L-* | 4 + 30 | Supp. |

**Manipulation gain (E4)**: the mean utility gain of exaggerators relative to their honest
counterfactual twins (the same agent with x=0 under CRN), expressed as a share of the
per-person resource cap.

---

## 6. Metrics (all from hidden states; LLMs never score)

| Metric | Definition |
|---|---|
| W | mean over agents and weeks of u_i^w |
| W_min | minimum over hidden persona clusters of the cluster-mean welfare |
| W_silent | mean welfare of the bottom quartile of t=0 response propensity |
| **ρ_rep** (representation ratio) | (share of accepted visits going to the silent quartile) / (share of total positive true marginal visit need held by the silent quartile). 1 = proportional; < 1 = under-served |
| Influence–need correlation | Spearman correlation between an agent's counterfactual policy influence (E6) and their true need |
| Burden | Σ λ_i · excess visits; declines |
| Resources | allocated and accepted visits, events, LLM calls, tokens, wall time |
| Alignment | share of agent-weeks where the next-week change in V_i matches the sign of the gold visit preference |
| Policy error | ‖π − π_oracle-share‖₁, where the oracle share is the allocation implied by C10's visit plan |
| Parse F1 | macro-F1 of parsed vs gold preference labels (overall and by style) |
| Audit | replay identity; policy variance under re-sampling (mean ‖Δπ‖₁ and |Δn_e|); trace completeness; influence concentration (Gini of influence) |
| Manipulation gain | defined in §5 |
| Degradation slope | dW/dη and dW/dx, fitted over the grid |

---

## 7. Gates (go/no-go before any eval run; all on dev seeds)

| Gate | Criterion | If it fails |
|---|---|---|
| G0 | all tests pass (including the information-boundary, budget, cap, replay and IPW tests) | fix the code |
| G1 legacy | L-* ordering is qualitatively consistent with MABS; rule-firing and cap-clip counts are non-zero where expected | report the deviations; do not tune to match |
| G2 calibration | virtual-RCT d within ±0.02 of the target for each regime | fix the calibration code |
| G3 fidelity | E8a Spearman(recovered ℓ, hidden ℓ) ≥ 0.6 and preference macro-F1 ≥ 0.6 for the default models | revise the persona/parse prompts **on dev only**; document the change |
| G4 non-degeneracy | share of agents for whom a 2nd weekly visit has negative marginal utility is between 5% and 60%; the response rate is 0.5 ± 0.05 | report; revisit the φ mapping on data grounds only |
| G5 power | pilot on dev seeds: paired d_z of the primary contrasts. If d_z < 0.55, raise E1 to 50 seeds and E3 to 30 | adjust the seed counts before the freeze |

After all gates pass: tag `prereg-v1` (configs, prompts, analysis scripts, primary endpoints) → eval runs.

---

## 8. Statistical analysis plan
- Unit = seed. Paired two-sided Wilcoxon signed-rank tests for the pre-specified contrasts:
  - Primary: P-1, P-2, P-3.
  - Secondary: C7r vs C1, C3, C6, C8, C9 on W, W_min and W_silent; C7r-noIPW vs C7r;
    C7r-noScreen vs C7r.
- Holm correction within each family (primary; each secondary table).
- Effect sizes: paired rank-biserial r and the median paired difference, with bootstrap 95% CIs
  (10k resamples over seeds).
- **Equivalence claims** ("matches the best static policy") need TOST with a margin of 25% of
  the (W(C10) − W(C1)) gap. Without it, only "no significant difference" may be said.
- Frontiers: the upper hull of C2; hypervolume with reference (max visits, W(C0)); the distance
  of each method to the hull at matched visits.
- Slopes (E4, E4b): mixed-effects regression, W ~ factor levels × method + (1 | seed).
- Sensitivity (E9): a sign-consistency matrix. A conclusion is "robust" if the sign of the
  primary contrast holds in every cell and is significant in at least 80% of the cells.
- Report n (seeds), the mean ± CI and the resources for every number. No p-values without effect sizes.

## 9. Result-contingent interpretation (decided in advance)
- **H2 fails but H1 holds** → the paper leads with representation (RQ1/RQ3). Efficiency is
  reported honestly: "matches or trails the frontier".
- **H1c confirmed (IPW fails under S2a)** → this is a finding, not a failure. It motivates the
  explicit welfare choice and screening, and it is discussed as a limit of any response-weighting scheme.
- **H6: keyword ≈ LLM** → claim that the architecture is parser-agnostic. The LLM's value then
  lies in indirect language and screening (E8a by style).
- **C9 ≈ C7r on W** → supports "structure (bounds) matters". C7r still wins on audit (P-3) and
  on explicitness of the trade-off.
- **Any conclusion not robust in E9** → state the condition under which it holds.

## 10. Threats to validity and mitigations
| Threat | Mitigation |
|---|---|
| Persona text is not human text | hidden-state evaluation; E8 fidelity; "controlled simulation" framing; a human study is future work |
| The preference mapping is an assumption | φ regimes including random (E9) |
| Effect sizes are not causal estimates | literature ranges, virtual-RCT calibration, E9 |
| NSHAP is ego-centric; 5-year spacing | a synthetic network from statistics; r5 constrains persistence only; α is swept |
| Same-family generator and interpreter | E8c/E7 cross-model matrix |
| Result-shaping | dev/eval split, gates, `prereg-v1` freeze, fixed calibration targets |
| Resource confounding | resource-matched comparator and frontier; resources reported with every number |
| Implementation errors | tests, legacy replication, oracle and none bounds, an independent code review of controllers |

## 11. Compute budget and run order
- Two vLLM servers on 2× A100 80GB (persona on GPU0, interpreter on GPU1; the 72B-AWQ model
  with TP=2 only for E7). About 1–2 minutes per LLM run at N=200; 12 runs are executed concurrently.
- Approximate LLM runs: E1 390, E1b 320, E2 700, E3 700, E4 720, E4b 960 (can reuse parsed
  logs with injected errors, which needs no new parsing calls, but the persona text must be
  regenerated per run), E6 300, E7 240, E8 small, E9 600, E10 30 → about 5,000 runs, **3–5 days of GPU time**.
- Order: E0 gates → freeze → E8 → E1 → E3 → E2 → E4b → E4 → E6 → E1b → E7 → E9 → E10 → L.
  Re-run E1 and E3 in NSHAP mode once R3 is integrated (the tables report both, or NSHAP mode
  as primary if it is available before the freeze).

## 12. Reproducibility package
Anonymous code repository; configs; `prereg-v1` tag; seeds; model ids and revisions; vLLM
version; prompts with hashes; the LLM call logs (compressed); derived population tables where
the licences allow it (BRFSS and ATUS derivatives yes; NSHAP-derived statistics only, no
microdata); an ODD protocol description; a script to rebuild every figure and table from the
logs.

## 13. Mapping to the paper
Fig. 1 concept; **Fig. 2 ← E3**; **Fig. 3 ← E2**; **Fig. 4 ← E4b (+E4)**; Table 1 ← E1;
Table 2 ← E1b; Table 3 ← E4 manipulation; Table 4 ← E6. Supplementary: E5, E7, E8, E9, E10, L,
data build reports.
