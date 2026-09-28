# Writing Plan: AAMAS Main-Track Paper

Consolidated writing plan (v2). It supersedes `paper_handoff/PAPER_BRIEF.md` where they differ.
Use it together with `plans/EXPERIMENT_PLAN.md`. The writer (human or AI) follows this plan section
by section and never states a result that is not in the frozen results files.

---

## 0. Target and standard
- **Venue**: AAMAS main track. ACM `sigconf` template; 8 pages of content + references;
  double-blind; supplementary material allowed (check the current CFP for page limits and
  supplementary rules).
- **Timing**: AAMAS 2027 (full paper 8 Oct 2026) is not realistic for this scope. Recommended:
  **IJCAI 2027** (≈ mid-January; check the CFP) or **AAMAS 2028**. Plan: submission-ready at T0 + 12 weeks.
- **What reviewers at this level check** (write to each explicitly):
  1. **Significance and relevance to MAS**: heterogeneous agents with private types, selective
     information revelation, strategic misreporting, and preference aggregation (social choice)
     inside an adaptive loop.
  2. **Originality**: a new problem framing (selective feedback in adaptive policy) plus a new
     decomposition, not just a new pipeline.
  3. **Soundness**: an explicit information boundary, hidden-state evaluation, resource-matched
     baselines, a strong black-box baseline, propositions with proofs, pre-registered endpoints,
     paired statistics, and sensitivity analysis.
  4. **Clarity**: one headline, consistent terminology, figures that carry the argument.
  5. **Reproducibility**: code, configs, prompts, seeds, model revisions, logs.
  6. **Honest limitations and ethics.**

## 1. Core message
**Headline.** *Adaptive policies observe voices, not preferences.*

**Elevator paragraph** (the abstract and introduction expand this):
> Care policies increasingly adapt to residents' feedback. But feedback is selective: the
> loneliest and frailest often say least, and what is said is indirect. A policy that adapts to
> what it hears can therefore systematically over-represent those who speak. We make this
> *representation gap* explicit by separating four steps that end-to-end LLM controllers
> conflate: who speaks (participation), what is meant (interpretation), how evidence is weighted,
> and which welfare criterion is applied. An LLM is used only to interpret; weighting, welfare
> choice and budget-feasible bounded updates are explicit and logged. In a data-grounded
> simulation of community aged care, evaluated against hidden ground-truth utilities, we show
> [RQ1 result], [RQ2 result] and [RQ3 result].

**Invariant distinctions** (use them consistently; they are the conceptual contribution):
Need ≠ Voice ≠ Policy influence; participation ≠ interpretation; evidence weighting ≠ welfare choice.

## 2. Contributions (final form; the numbers come from the results)
1. **Problem**: preference-aware policy adaptation under *selective feedback*. It is formalized
   with an explicit information boundary (latent state → expression → observable evidence →
   aggregation → policy) and evaluated on hidden utilities, which avoids LLM-judges-LLM circularity.
2. **Method**: a decomposition into participation, interpretation, weighting and welfare choice.
   An LLM interpreter and silent-risk screen feed explicit IPW-weighted aggregation under a
   stated welfare criterion, with bounded, budget-feasible, fully logged updates.
3. **Analysis**:
   - P1 replay and attribution;
   - P2 bounded sensitivity to interpretation error;
   - P3 IPW consistency under segment-MAR non-response, and the explicit bias of response-only aggregation;
   - P4 IPW variance inflation under interpretation noise, and the bias–variance role of weight caps.
4. **Evidence**: a controlled study grounded in BRFSS, ATUS and NSHAP, covering:
   - representation distortion across silence mechanisms;
   - resource-matched welfare frontiers;
   - interpretation × weighting interaction;
   - robustness to noise and manipulation;
   - audit metrics;
   - generality across LLM families;
   - state-expression fidelity;
   - sign-consistency under assumption sweeps.

## 3. Claims ledger (a claim is allowed only with its evidence)

| # | Claim (strongest allowed wording) | Evidence | Condition for this wording | Fallback wording |
|---|---|---|---|---|
| K1 | "Response-only adaptation under-serves silent residents" | E3: ρ_rep < 1 and W_silent(C6) < W_silent(C7r) | P-1 significant (Holm) under S1 | "tends to", with CI |
| K2 | "IPW plus silent screening reduces the representation gap under segment-MAR and need-dependent silence" | E3 (S0, S1) | significant in both | state the regime in which it holds |
| K3 | "No response-weighting scheme corrects dissatisfaction-dependent silence" | E3 (S2a) + P3 discussion | observed failure | "IPW did not correct…" |
| K4 | "At equal resources, our method matches or exceeds the best static policy" | E2, P-2 | exceeds: CI > 0; matches: TOST passes | "trails by X at matched visits" |
| K5 | "Uncapped IPW amplifies interpretation errors; caps trade bias for variance" | E4b + P4 | interaction significant | "we observe…" |
| K6 | "Decisions are exactly reproducible and attributable" | E6 replay + P1 | replay 100% | none (must hold, or it is a bug) |
| K7 | "End-to-end LLM control is less stable" | E6 variance ratio, E4 slopes | P-3 significant | "was more variable in our setting" |
| K8 | "Findings hold across LLM families and assumption regimes" | E7, E9 sign matrix | robust by the §8 rule of the experiment plan | list the exceptions |
| K9 | "Persona feedback preserves intended latent states" | E8a/b/c | gates met; report the numbers | never "realistic residents" |

**Forbidden claims**: real-world effect sizes; the personas are real or valid people; "full
auditability" (say *control-layer auditability*); one welfare criterion is correct; NSHAP gives
daily dynamics; ATUS measures preferences; nationally representative loneliness.

## 4. Structure, page budget and paragraph plan (8 pages)

### Abstract (≈180 words)
Problem (selective feedback) → gap (end-to-end adaptation conflates four steps) → method
(decomposition; the LLM interprets only; explicit weighting, welfare and bounded updates) →
evaluation (data-grounded, hidden-state, resource-matched) → 3 headline numbers (K1, K4, K5 or
K6) → one closing sentence.

### 1 Introduction (1.0 page, 6 paragraphs + Fig. 1)
1. **Hook**: adaptive care policy and loneliness (health burden; cite Holt-Lunstad et al. 2015
   and the WHO 2025 Commission). Adaptation increasingly relies on feedback, and LLMs make
   free-text feedback usable.
2. **Problem**: *policies observe voices, not preferences*. Selective participation (non-response
   correlates with need; Groves 2006) and indirect expression. Consequence: a representation gap.
3. **Why the obvious solution is insufficient**: an end-to-end LLM that reads feedback and sets
   policy conflates participation, interpretation, weighting and welfare choice. The trade-off
   becomes implicit, and it is non-reproducible and unbounded.
4. **Our approach**: the decomposition, and which component is an LLM and which is explicit. One
   sentence on the information boundary and hidden-state evaluation.
5. **Evidence preview**: the key results in three sentences with numbers (K1, K4, K5/K6).
6. **Contributions** (bulleted, from §2) + one sentence distinguishing this work from the
   prior workshop version (in the third person).

### 2 Related Work (0.7 page, 4 paragraphs; each ends with "we differ by…")
1. **LLM-augmented agent-based simulation and its validity**: Park et al. 2023; Gao et al. 2024;
   Larooij & Törnberg (validation). → We do not rely on behavioural realism; evaluation is on hidden states.
2. **LLMs as simulated humans**: Argyle et al. 2023; Aher, Arriaga & Kalai 2023; Santurkar et al.
   2023; Wang, Morgenstern & Dickerson 2025. → Personas only *express* hidden states; we test
   state-expression fidelity.
3. **AI for policy design and preference aggregation**: Zheng et al. 2022 (AI Economist); Koster
   et al. 2022 (Democratic AI); Bakker et al. 2022; Tessler et al. 2024; Fish et al. (generative
   social choice); Conitzer et al. 2024; Caragiannis et al. 2019; Brandt et al. 2016. → We study
   *selective* feedback inside an adaptive loop, with an explicit welfare rule and non-response correction.
4. **Interpretable and safe adaptive control; non-response**: Rudin 2019; Milani et al. 2024;
   Alshiekh et al. 2018; García & Fernández 2015; Horvitz–Thompson/Hájek; Little & Rubin; Groves
   2006; the prior workshop paper. → Bounded, logged updates with sensitivity guarantees, and IPW
   whose failure modes we characterize.

**Reference policy**: verify every entry (authors, venue, year, pages) against the publisher or
DBLP. Unverified → `[CITE?]` and a flag. Never fabricate.

### 3 Problem Formulation (0.8 page)
- **Agents and hidden types**: Z_i = (ℓ, s, f, ties, activity profile A_i, preferences μ_i, λ_i, τ_i).
- **Utility** u_i^w and the welfare family W_φ (utilitarian, Rawlsian soft-min, Nash) over hidden utilities.
- **Participation**: R_{i,w} ~ Bernoulli(p_i(·)); the silence mechanisms S0/S1/S2 are introduced as a *controlled family*.
- **Expression**: y_{i,w} ~ Persona(Z_i, week).
- **Observation** O_t, and the **information boundary**: Table 1 (a small table: variable → layer → visible to controller?).
- **Budget** C(a_t) ≤ B, and the policy class.
- **Objective**: choose a policy map O_t → a_t that maximizes a declared W_φ subject to the
  budget, while being replayable and attributable.
- **Definitions**: representation ratio ρ_rep; control-layer auditability (P1 properties).

### 4 Method (1.3 pages, Algorithm 1)
- 4.1 Interpretation (LLM parse schema; validation; fallback).
- 4.2 Silent handling (LLM screen of service records; imputation rule).
- 4.3 Evidence weighting (Hájek per segment; population weights; stabilized, capped weights; shrinkage).
- 4.4 Welfare choice (the three target maps; why the choice is exposed to stakeholders).
- 4.5 Bounded, budget-feasible update (step bounds; simplex and budget projection; allocation within segments).
- 4.6 Audit log (record schema).
- **Algorithm 1**: the weekly cycle, ≤ 15 lines.
- A short paragraph: *what the LLM never does* (it does not set weights, choose welfare, or act).

### 5 Properties (0.6 page; full proofs in the supplementary material)
- **P1 Replay and attribution.** The control map is deterministic given the logged
  interpretations; each Δπ decomposes into rule-level terms. *Sketch*: composition of deterministic maps.
- **P2 Bounded sensitivity.** If ‖ŝ − s‖_∞ ≤ ε, then ‖Δπ − Δπ'‖ ≤ min(δ, Lε) per step, and ≤ H·min(δ, Lε)
  over H steps. Event rules flip only if demand is within ε of the threshold. *Sketch*: Lipschitz
  target maps + clipping + non-expansive projection (Duchi et al. 2008).
- **P3 Non-response.** Under segment-MAR, the Hájek segment means with population weights are
  consistent. Response-only pooling has bias Σ_g (n_g r_g / Σ_h n_h r_h − n_g/N) m_g, which is
  non-zero whenever response rates co-vary with segment means. Under MNAR within segments,
  consistency fails (this motivates screening and E3-S2).
- **P4 Variance inflation under interpretation noise.** With a label-flip rate ε_p, the segment
  signal is attenuated by (1−2ε_p) (the same in all segments). The variance of the population
  estimate is Σ_g (n_g/N)² σ_g² / (n_g r_g): low-response segments inflate variance ∝ 1/r_g.
  Capping weights at w_max bounds this term at the cost of bias toward responders. *Sketch*:
  variance of a Hájek mean with a Bernoulli response, plus capped weights.
- One sentence per proposition: *what it means for practice*.

### 6 Experimental Setup (0.9 page)
- 6.1 **Data-grounded population**: the three-layer design (a small figure or a table with
  dataset → role → not claimed); PMM fusion; GMM personas; the preference mapping φ as an assumption.
- 6.2 **Dynamics and calibration**: the virtual RCT; r5; what is swept.
- 6.3 **Conditions**: Table 2 (IDs, what each isolates).
- 6.4 **Metrics** (W, W_min, W_silent, ρ_rep, burden, resources, policy error, audit).
- 6.5 **Protocol**: seeds, CRN, dev/eval, gates, pre-registered endpoints, statistics, models, compute.

### 7 Results (2.0 pages; each subsection opens with the answer in one sentence)
- 7.1 **Who gets represented?** (RQ1) Fig. 2 + text on S0/S1/S2 (K1–K3).
- 7.2 **Is it just more visits?** (RQ2) Fig. 3 + Table 3 main results with resources (K4).
- 7.3 **When interpretation errs, does weighting amplify it?** (RQ3) Fig. 4a (K5).
- 7.4 **Robustness and manipulation** (RQ4) Fig. 4b / a small table (manipulation gain).
- 7.5 **Audit and stability** (RQ5) Table 4 (K6, K7).
- 7.6 **Is an interpreter needed? Generality and fidelity** (RQ6, E1b, E7, E8, short) (K8, K9).
- Sensitivity: a one-paragraph summary of the E9 sign matrix, with the details in the supplementary material.

### 8 Discussion, Limitations, Ethics (0.5 page)
- **Implications**: design feedback-driven systems around participation *and* interpretation;
  expose the welfare choice; bounds and logs as governance tools.
- **Limitations**:
  - controlled simulation and persona text;
  - the preference mapping is an assumption;
  - effects come from literature ranges;
  - NSHAP spacing (α is swept);
  - the IPW limits under MNAR;
  - US data only.
- **Ethics**: public-use data only; no personal data; feedback is sensitive in deployment;
  choosing the welfare criterion requires stakeholder participation; screening assists staff and does not replace them.

### 9 Conclusion (0.2 page)
Restate the headline, and add one sentence on future work (a human-in-the-loop study, learning
the rules, other domains).

## 5. Figures and tables (specifications)

**Fig. 1 — "Selective feedback separates population need from observed voice."** (full width, top of page 2)
- Four regions, left→right:
  1. heterogeneous residents (4 glyphs with need/preference bars; one speaks, one is silent "…";
     a faint grounding bar "BRFSS · ATUS · NSHAP");
  2. selective feedback (a funnel "who speaks", then the LLM "what is said");
  3. explicit aggregation (structured signals → IPW / silent handling → welfare rule; W, W_min, W_silent; C(a) ≤ B);
  4. budgeted intervention (visit / group / outreach) looping back to residents.
- **Solid lines** = observable/control path; **dashed lines** = simulation-only state.
  Footnote: "outcomes are evaluated on hidden simulator states".
- It must not look like the workshop architecture figure. Colour-blind safe palette, vector PDF,
  readable at column width.

**Fig. 2 (E3)**: two panels, W_silent and ρ_rep vs silence strength κ (S1), with S0/S2a/S2b as
side markers. Lines: response-only (C6), IPW only, screening only, IPW + screening (C7r),
black-box (C8). Shaded 95% CI.

**Fig. 3 (E2)**: accepted visits (x) vs W (y). Grey dots and the upper hull for the static grid;
coloured markers for the methods at each budget. An inset or second panel shows W_silent at matched visits.

**Fig. 4 (E4b / E4)**: (a) policy error vs parse-error rate ε_p, with lines for naive / IPW∞ /
IPW5 / IPW3 under S1; (b) degradation of W vs η and exaggeration x for C7r, C8 and C9.

**Tables**:
1. the information boundary (in §3);
2. conditions (in §6);
3. main results E1 (W, W_min, W_silent, ρ_rep, burden, accepted visits, events, LLM calls; mean ± 95% CI; Holm-marked);
4. audit E6.

Ablation E1b, model grid E7, fidelity E8 and sensitivity E9 go to the supplementary material,
with one sentence each in the text.

All figures are generated by scripts from the frozen results (no manual numbers).

## 6. Notation and terminology (glossary; use it verbatim)
| Term | Meaning |
|---|---|
| need | the true positive marginal utility of a service (hidden) |
| voice / participation | whether a resident gives feedback in a week (R) |
| interpretation | mapping text to structured signals (LLM) |
| evidence weighting | how signals are combined across residents and segments (IPW etc.) |
| welfare choice | the declared criterion W_φ |
| representation gap / ratio | mismatch between the resources received and the true need of silent residents (ρ_rep) |
| silent residents | the bottom quartile of t=0 response propensity |
| control-layer auditability | P1: exact replay given logged interpretations, plus rule-level attribution |
| controlled missingness mechanism | the silence regimes S0/S1/S2; not estimates of real rates |
| activity profile | ATUS-derived behaviour (never "preference") |

Symbols: follow EXPERIMENT_PLAN §3 and SPEC §0.3.1; one notation table in the supplementary material.

## 7. Handling the prior workshop paper (double-blind)
- Cite it in the third person as prior work, and list the differences in Related Work:
  1. selective feedback and silence;
  2. heterogeneous hidden preferences;
  3. explicit welfare choice and IPW;
  4. budgets and resource-matched evaluation;
  5. a data-grounded population;
  6. propositions;
  7. a much larger evaluation.
- It appears as baseline C3. No reused text or figures.
- Check the venue's policy on prior archival workshop papers; add a disclosure if required.

## 8. Writing process and timeline (relative to T0 = start of the build)

| Week | Writing | Depends on |
|---|---|---|
| 1–2 | Related work (verified references); problem formulation; notation; Fig. 1 draft | SPEC §0.3 |
| 3 | Method + Algorithm 1; proposition statements and proofs (P1–P4); supplementary ODD | build M4 |
| 4 | Setup section (data design, calibration, conditions, protocol); intro skeleton with placeholders | gates G0–G5 |
| 5–6 | (Runs in progress) Figure scripts on dev data; results-section skeleton; limitations/ethics | E-runs |
| 7 | Fill in results from the frozen outputs; choose the result-contingent branches (EXPERIMENT_PLAN §9); abstract | all primary runs |
| 8 | Full draft; page fitting; supplementary material (proofs, prompts, parameters, extra results, reproducibility checklist) | – |
| 9 | **Internal review round 1**: two "reviewer" passes (§9), one by a co-author and one by an AI with the AAMAS review criteria | – |
| 10 | Revisions; a re-check of every number against `summary.csv`; figure polish | – |
| 11 | **Round 2**: supervisor review (M. Witbrock); the anonymity check; the reference check | – |
| 12 | Final proofreading; CFP compliance; submit | – |

## 9. Internal review protocol (simulate the AAMAS review)
Each reviewer pass fills in: summary; strengths; weaknesses; questions; scores for relevance,
significance, originality, soundness, clarity, reproducibility; and a recommendation. The pass
must explicitly try to break these claims:
- "Is the gain just more resources?" → Fig. 3, matched comparator.
- "Is the black-box a straw man?" → the C8 prompt (objective, budget, history, messages) + C9.
- "LLM judging LLM?" → hidden-state evaluation; E8 cross-model matrix.
- "Are the personas realistic?" → we do not claim it; state-expression fidelity only.
- "Where do the preferences come from?" → the φ mapping + regimes + random.
- "Why is this MAS / AAMAS?" → private types, selective revelation, strategic misreporting,
  social-choice aggregation, and a multi-agent feedback loop.
- "Is IPW standard?" → yes. The contribution is the decomposition and the characterization of
  when it helps or fails (P3, P4, E3, E4b).
- "Novelty over the workshop paper?" → the list in §7.
- "Statistics with few seeds?" → 30–50 paired seeds, pre-registered endpoints, Holm, TOST.
- "Daily dynamics from NSHAP?" → not claimed; α swept.
- "Is the rule design ad hoc?" → it comes from the stated welfare family; the parameters are
  swept (δ, w_max, thresholds); PPO/C11 is an optional learned baseline.

Keep the answers in `REBUTTAL_NOTES.md` for the rebuttal phase.

## 10. Style rules
- Every paragraph has one job. Every claim cites a figure, table or proposition. Numbers always
  come with a CI, n and resources.
- Prefer "we find / we observe" to "we prove" unless it is a proposition. Use "significant" only statistically.
- No hype adjectives. Keep terms consistent (glossary). Use active voice. Define an abbreviation
  once. Use `\cref` and booktabs tables.
- Captions are self-contained: what is plotted, the conditions, n, the CI type, and the takeaway in one clause.
- Anonymity: no names, no identifying URLs (use an anonymous repository), and third-person self-citation.

## 11. Supplementary material
- A. Full proofs (P1–P4).
- B. ODD protocol description of the ABM.
- C. The data build: variables, fusion diagnostics, GMM selection, the network fit, r5.
- D. All parameters with sources and ranges.
- E. Prompts (full text + hashes) and schemas.
- F. Extra results: E1b, E5, E7, E8, E9 (sign matrix), E10, legacy replication.
- G. The reproducibility checklist and compute.
- H. Ethics and data licences.

## 12. Submission checklist
- [ ] Every number in the text and tables is produced by scripts from frozen results; no placeholders.
- [ ] Primary endpoints reported first and exactly as pre-registered; deviations disclosed.
- [ ] A resource-matched comparison is in the main paper; resources are shown next to every welfare number.
- [ ] The black-box baseline prompt is shown in the supplementary material; the C9 variant is included.
- [ ] P1–P4 are stated precisely, with proofs in the supplementary material.
- [ ] Limitations cover the persona text, the φ mapping, effect sizes, NSHAP spacing, MNAR, and US data.
- [ ] The workshop paper is cited in the third person, with its differences stated; no reused text.
- [ ] References verified; no `[CITE?]` left.
- [ ] Anonymized PDF, metadata stripped; page limit met; template unmodified.
- [ ] The anonymous code and supplementary archive builds all figures and tables.
