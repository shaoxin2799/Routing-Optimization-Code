# Paper Writing Brief: Auditable, Preference-Aware Policy Adaptation

This brief is for the AI assistant that will **write the paper**. It fixes the storyline, the
argument, the claims we are allowed to make, and how each claim is backed by an experiment.
Experiments are being run by a separate build (see `codex_handoff/SPEC.md`). Where results are
not in yet, write with placeholders like `[E1: W(C7r)=?]`, and follow the **result-contingent
branches** in §9.

Authors (camera-ready only; the submission is anonymous): Shaoxin Zhong, Yuchen Su, Michael
Witbrock (University of Auckland).
Target venue: **AAMAS main track** (ACM `sigconf`, 8 pages + references, double-blind).
Fallbacks: IJCAI (7+2 pages) or JAAMAS (journal).

---


## 0. Revision v2: framing and wording (authoritative; supersedes conflicting text below)

**Headline:** *Adaptive policies observe voices, not preferences.*
> When feedback is selective, policy adaptation can systematically over-represent those who
> speak. We separate participation (who speaks), interpretation (what is said), evidence
> weighting, and welfare choice, which makes this representation gap measurable and
> controllable under fixed resources.

This replaces "LLMs understand, rules decide" as the spine. Keep that idea as a design
principle inside the method section, not as the headline.

**The core distinction to carry through the paper:** Need ≠ Voice ≠ Policy influence;
participation ≠ interpretation.

**Data wording (use exactly):**
- BRFSS: "public-use microdata of older adults in jurisdictions that fielded the Social
  Determinants and Health Equity module", **not** "nationally representative loneliness".
- ATUS: "activity profiles" or "revealed social participation", **never** "preferences". Latent
  intervention preferences are generated conditionally on activity profiles through a mapping φ
  with weak/medium/strong/random regimes. Sentence to include:
  *"We do not equate observed activity with stated intervention preference; ATUS constrains
  heterogeneous activity profiles, and latent preferences are generated conditionally on these
  profiles and varied in sensitivity analyses."*
- NSHAP: "network statistics calibrated to NSHAP" (a synthetic network from ego-network roster
  statistics), **not** "the NSHAP network". The UCLA-3 construct calibrates the loneliness distribution.
  R1–R3 (5-year spacing) constrain **long-run persistence only**. Short-term dynamics are a
  modelling assumption chosen to be compatible with it. Never claim NSHAP gives daily dynamics.
- Intervention effects: literature ranges (low/mid/high), with conclusions checked across all three.
- Silence: "controlled missingness mechanisms" (MAR-by-segment, need-dependent,
  dissatisfaction-dependent), **not** estimates of real response rates.
- Present the data design as three layers: population grounding (BRFSS) → social grounding
  (ATUS activity profiles + NSHAP network) → dynamic calibration (NSHAP persistence + intervention
  literature). Do not present it as a list of datasets.

**Persona check wording:** E8 is "persona state-expression fidelity". It tests whether
persona-generated feedback preserves the intended latent-state distributions across models and
population strata (E8a state recovery, E8b distributional calibration, E8c cross-model
generator × interpreter matrix). Never write "the personas are valid" or "realistic residents".

**Figure plan (replaces §11 order):**
- **Fig. 1**: *"Selective feedback separates population need from observed voice."* It is not
  an architecture diagram and must not resemble the MABS figure. Four regions left→right:
  1. heterogeneous residents (glyphs with need/preference bars; one speaks, one is silent "…";
     a faint grounding bar "BRFSS · ATUS · NSHAP");
  2. selective feedback (a funnel "who speaks", then the LLM "what is said");
  3. explicit aggregation (structured signals → IPW / silent handling → welfare rule; notation
     W, W_min, W_silent; C(a) ≤ B);
  4. budgeted intervention (visit / group / outreach icons) looping back to residents.
  **Solid lines** = observable/control path; **dashed lines** = simulation-only state (latent
  preference, true loneliness, counterfactual welfare). Footnote in the figure: "outcomes are
  evaluated on hidden simulator states".
- **Fig. 2**: representation distortion: W_silent and W_min vs silence strength and regime, for
  response-only, IPW, screen, and IPW+screen (E3). This is the first result.
- **Fig. 3**: resource-matched welfare frontier: resource use (x) vs W (y); the static hull vs
  the methods (E2). The main table must also report visits beside every welfare number, so the
  "is it just more visits?" question is answered on the first results page.
- **Fig. 4**: interpretation × weighting: does IPW amplify LLM parse errors? (E4b), plus the
  weight-cap effect.
- Welfare-criterion trade-off, audit, model grid and sensitivity go to tables or supplementary material.

**Honesty points that must appear:**
- IPW corrects between-segment non-response only. Under need- or dissatisfaction-dependent
  silence (MNAR) it can fail; report where it fails. Silent screening and the explicit welfare
  choice are the mitigations, and they are measured, not assumed.
- IPW can amplify interpretation errors in low-response groups. Weight caps are a
  bias–variance choice, and we report it (E4b).

**Revised contribution list:**
1. Formulation of preference-aware adaptation under **selective feedback**, with an explicit
   information boundary (latent state / expression / observable evidence / aggregation / policy)
   and evaluation on hidden states.
2. A separation of participation, interpretation, weighting and welfare choice. The LLM
   interprets; explicit rules weight, aggregate and act within a budget, and every step is logged.
3. Properties: replay/attribution, bounded sensitivity, IPW under MAR-by-segment and its failure under MNAR.
4. A data-grounded controlled study: representation distortion, resource-matched frontiers,
   interpretation × weighting interaction, state-expression fidelity, and sensitivity to the
   preference mapping, effect sizes and silence regimes.

---

## 1. One-sentence thesis
> When a care policy must adapt to people with **diverse and partly unspoken preferences**, the
> LLM should be used to **understand people** (turn free-text feedback and service records into
> structured signals), while **weighing people**, i.e. aggregating their preferences under an
> explicit welfare criterion with bounded, budget-feasible updates, must stay in **explicit,
> auditable rules**. This separation yields policies that are better aligned with what people
> actually want, fairer to those who do not speak up, more robust to noisy feedback, and fully
> traceable.

## 2. The story arc (use it as the spine of the Introduction)
1. **Problem.** Social-care policy (our case: loneliness among older adults in community care)
   must be *adaptive* (needs change week to week) and *accountable* (staff, families and
   regulators must be able to see why a decision was made). Rudin (2019) argues that high-stakes
   decisions should use interpretable models.
2. **The shift in question.** Prior simulation work, including our own earlier workshop paper,
   asks *"does the simulated policy reduce loneliness?"* The real question is *"is the policy
   what people want?"* Older adults are heterogeneous: some welcome visits, some find them
   intrusive, some enjoy group activities, some avoid them. Pushing more intervention is not
   always better; a single-objective simulation hides this.
3. **Why this is hard.** (a) Preferences are expressed in **free text** ("my daughter comes
   often, I'm fine") that is indirect and noisy. (b) **Not everyone speaks.** The loneliest and
   frailest respond least, so responsive policies drift toward the vocal (a well-known
   non-response bias in surveys). (c) **Aggregation is a normative choice**
   (utilitarian vs. Rawlsian vs. Nash). It must be visible and changeable, not buried inside a model.
4. **Tempting but flawed answer.** Let an LLM read all feedback and set the policy end to end.
   It is flexible, but the fairness trade-off becomes implicit, decisions are not reproducible,
   and noise propagates without bounds.
5. **Our answer.** A three-layer architecture that *separates understanding from weighing*:
   heterogeneous persona agents grounded in public survey data (simulation layer); an LLM that
   parses feedback and screens silent residents (diagnosis layer); explicit welfare aggregation
   with inverse-propensity correction and bounded, budget-feasible updates (control layer).
   Evaluation uses **hidden ground-truth utilities**. The LLM never judges outcomes.
6. **Evidence.** A resource-matched evaluation (Pareto frontier against static policies), a
   voice-bias study, robustness to noisy and strategic feedback, audit metrics, multiple LLM
   families, and a persona-validity check against population survey data.

## 3. Contributions (final wording depends on results; see §9)
1. **Problem formulation.** Preference-aware policy adaptation in agent-based social care, with
   heterogeneous hidden utilities, free-text feedback and non-random silence. Evaluation is
   against hidden ground truth, which avoids LLM-judges-LLM circularity.
2. **Architecture.** LLM-as-interpreter / rules-as-arbiter: an explicit welfare aggregation
   (utilitarian, Rawlsian or Nash) with IPW correction for non-response, LLM silent-risk screening,
   and bounded, budget-feasible updates. Every policy change is logged with its inputs and the rule that fired.
3. **Guarantees.** (P1) exact replay and attribution; (P2) bounded policy sensitivity to diagnosis
   error; (P3) unbiasedness of the IPW segment estimates under segment-level missing-at-random
   (MAR), and the bias of naive pooling.
4. **Empirical study.** A data-grounded population (BRFSS 2022 SD/HE module + ATUS time use,
   fused by predictive mean matching). Resource-matched comparisons against static Pareto
   policies, state-only closed-loop control (our prior method), structured-survey and
   keyword baselines, informed black-box LLM control (with and without step caps), and a myopic
   oracle. Voice-bias, robustness, audit, multi-model and sensitivity analyses.

## 4. Relation to our earlier workshop paper (MABS 2026), handled for double-blind review
- Cite it **in the third person** as prior work (e.g., "[X] proposed separating LLM diagnosis
  from deterministic control in a homogeneous facility model…").
- State the differences explicitly in Related Work, and again in one sentence in the
  Introduction:
  1. heterogeneous hidden preferences instead of a single loneliness objective;
  2. natural-language feedback and silence instead of state-only diagnosis;
  3. explicit welfare aggregation with non-response correction;
  4. budgets and resource-matched evaluation;
  5. a data-grounded population;
  6. theory;
  7. a far larger evaluation (30 seeds vs. 4, N=200 vs. 30).
- The prior method appears as a baseline (C3). **Do not reuse text or figures from the MABS paper.**
- Before submission, check the venue's policy on prior workshop publication and add a
  disclosure if it requires one.

## 5. Positioning and related work (organize in 4 paragraphs; verify every reference before citing)
1. **LLM-augmented agent-based simulation.** Generative agents (Park et al., 2023, UIST); a survey of
   LLM-empowered ABM (Gao et al., 2024, *Humanities & Social Sciences Communications*); validation
   concerns (Larooij & Törnberg, *Artificial Intelligence Review*). **Our angle**: we use LLMs for
   feedback expression and interpretation, but ground truth is a hidden utility, so validity
   does not rest on LLM "behaviour realism".
2. **LLMs as simulated humans and their limits.** Silicon samples (Argyle et al., 2023, *Political
   Analysis*); simulating multiple humans (Aher, Arriaga & Kalai, 2023, ICML); whose opinions LMs
   reflect (Santurkar et al., 2023, ICML); LLMs flattening identity groups (Wang, Morgenstern &
   Dickerson, 2025, *Nature Machine Intelligence*). **Our angle**: personas are conditioned on
   survey-derived attributes, and we test their distributional validity against BRFSS (E8).
3. **AI for policy design and preference aggregation.** RL-based economic policy in ABM (Zheng et
   al., 2022, *Science Advances*, "The AI Economist"); human-centred mechanism design (Koster et al.,
   2022, *Nature Human Behaviour*); LLMs finding agreement or common ground among diverse people
   (Bakker et al., 2022, NeurIPS; Tessler et al., 2024, *Science*); generative social choice
   (Fish et al.); social choice for AI alignment (Conitzer et al., 2024, ICML position paper);
   Nash welfare fairness (Caragiannis et al., 2019, *ACM TEC*); the handbook of computational
   social choice (Brandt et al., 2016). **Our angle**: an explicit social-choice rule sits in the
   control loop, and the LLM is only the interpreter.
4. **Interpretable and safe control.** Interpretable models for high-stakes decisions (Rudin,
   2019); explainable RL (Milani et al., 2024, *ACM CSUR*); shielding and safe RL (Alshiekh et al.,
   2018, AAAI; García & Fernández, 2015, JMLR); LLM closed-loop policy assistants (pandemic,
   traffic; cite those used in the MABS paper). **Our angle**: bounded updates and rule-level
   traceability as design requirements, with a sensitivity guarantee.

Also cite domain and method sources:
- Loneliness and mortality: Holt-Lunstad et al., 2015, *Perspectives on Psychological Science*;
  the WHO 2025 Commission on Social Connection.
- Intervention effect sizes: Masi et al., 2011, *PSPR*.
- UCLA-3 scale: Hughes et al., 2004, *Research on Aging*.
- Non-response bias: Groves, 2006, *Public Opinion Quarterly*.
- IPW: Horvitz & Thompson, 1952; Hájek. Missing data: Little & Rubin.
- Simplex projection: Duchi et al., 2008, ICML.
- ODD protocol: Grimm et al., 2020, JASSS.
- Data: BRFSS (CDC) and ATUS (BLS).

**Do not invent references.** If a detail cannot be verified, leave `[CITE?]`.

## 6. Paper structure and page budget (8 pages)

| § | Title | Pages | Content |
|---|---|---|---|
| – | Abstract | – | 150–200 words: problem → gap → architecture → evaluation → key numbers (placeholders) |
| 1 | Introduction | 1.0 | story arc §2; Fig. 1 teaser (architecture); contributions; one sentence on how this differs from prior work |
| 2 | Related Work | 0.7 | four paragraphs from §5 |
| 3 | Problem Formulation | 0.8 | agents, hidden utility, observations, voice, budgets, the welfare objective family; what the controller can and cannot see |
| 4 | Architecture | 1.3 | simulation, diagnosis and control layers; aggregation rules; IPW + silent screening; bounded update + projection; audit log. Algorithm 1 (weekly cycle) |
| 5 | Properties | 0.6 | P1–P3 with short proofs; full proofs in the appendix or supplementary material |
| 6 | Experimental Setup | 0.9 | population grounding (BRFSS+ATUS, PMM, GMM personas, preference mapping); calibration by virtual RCT; conditions table; metrics; seeds and statistics |
| 7 | Results | 2.0 | RQ1 Pareto (Fig. 2) + Table 1; RQ2 voice bias (Fig. 3); RQ3 welfare trade-off + audit (Fig. 4, Table 2); RQ4 robustness; RQ5 parsing and LLM necessity; model generality (short) |
| 8 | Discussion & Limitations | 0.5 | what the results mean for deployment; limitations (§10); ethics |
| 9 | Conclusion | 0.2 | |
| – | References | extra | |
| – | Supplementary | extra | ODD description, all parameters, prompts, sensitivity, persona validity, legacy replication |

## 7. Notation (keep consistent with SPEC.md)
- Agents i ∈ {1..N}; days t; weeks w. Loneliness ℓ_i(t) ∈ [0,1]; stress s_i; frailty f_i.
- Hidden preferences: μ_i ∈ [−1,1] (group-activity preference), λ_i ≥ 0 (intrusion aversion),
  τ_i (comfortable visits per week).
- Weekly hidden utility: u_i^w = −mean_t ℓ_i(t)² − λ_i·max(0, V_i^w − τ_i) + c_A·μ_i·A_i^w.
- Voice probability: π^voice_i = v0(1 − s_ℓ ℓ_i)(1 − s_f f_i)(1 − s_c cog_i).
- Observable segments g (lives alone × mobility limitation × age ≥ 80); segment shares π_g;
  number of events n_e; budgets B_v, B_e.
- Parsed signals: d_i ∈ {−1,0,1} (visits), e_i (events), sat_i. Response rate r̂_g.
  Segment estimates D_g, ŝ_g.
- Welfare criteria: utilitarian, Rawlsian (soft-min), Nash (log). Step bound δ.
- Metrics: W (mean welfare), W_min (worst hidden persona group), W_silent (bottom-quartile voice),
  burden, visits.

## 8. Formal properties (state them precisely; keep proofs short)
- **P1 (Replay and attribution).** Given the logged diagnosis outputs, the control layer is a
  deterministic function of (observations, parsed signals, previous policy). Hence the policy
  trajectory is exactly reproducible, and each change Δπ_t decomposes into rule-level
  contributions of identifiable inputs. *Proof:* composition of deterministic maps. Rule ids
  and inputs are logged. Contrast: an end-to-end LLM controller is reproducible only up to
  sampling and cannot be decomposed without re-querying.
- **P2 (Bounded sensitivity).** Let the parsed segment signals differ from the true ones by at
  most ε in sup-norm. Then the per-step change in shares satisfies ‖π_{t+1} − π'_{t+1}‖_∞ ≤ min(δ, L·ε)
  (L is the Lipschitz constant of the target map π*(·), computable for each welfare rule on the
  domain D ∈ [−1,1], ŝ ∈ [0.05,1]). Projection onto the simplex is non-expansive in ℓ₂ (Duchi et
  al.), so over H steps the deviation is ≤ H·min(δ, L·ε). The event rule changes only if the
  population demand lies within ε of the threshold. *Proof sketch:* Lipschitz composition with
  clipping and projection.
- **P3 (Voice bias and IPW).** If agent i responds with probability r_g depending only on its
  segment (MAR within segment), the Hájek segment mean over responders is consistent for the
  segment mean, and weighting segments by n_g gives a consistent population estimate. Naive
  pooling weights segment g by n_g·r_g, so its bias equals Σ_g (n_g r_g/Σ n_h r_h − n_g/N)·m_g,
  which is non-zero whenever response rates and segment needs co-vary. Within-segment MNAR
  (lonelier people speak less) is not corrected by IPW. This motivates LLM silent-risk screening
  from service records, whose effect we measure empirically (E3). *State this limitation honestly.*

## 9. Result-contingent narratives (choose the branch that matches the numbers)
**RQ1 (Pareto, resource-matched).**
- If C7 lies **above** the static frontier (higher W at equal accepted visits): the headline
  becomes "feedback-driven explicit adaptation dominates any static policy at equal resources".
- If C7 lies **on** the frontier: reframe to "adaptation matches the best static policy *without
  knowing it in advance*". The best static point differs by population and budget (show this
  from E2/E9), and C7 gains on W_min and W_silent. Lead with RQ2 and RQ3.
- If C7 lies **below** the frontier: do not hide it. Lead with fairness, robustness and
  auditability, and discuss the cost of adaptation.
- **Always** report the Max policy and burden. Never claim loneliness gains without the
  corresponding visit counts.

**RQ5 (is the LLM needed?).**
- If LLM parsing (C7) > keyword (C5) on parse F1 and W: "free text needs an interpreter".
- If C5 ≈ C7: say that the architecture is parser-agnostic. The LLM's added value is then
  in silent screening and indirect language; show the E8 accuracy on indirect-style personas.
- C4 (structured survey) is an *idealized* reference. If it wins, say that structured surveys
  are best when people will actually fill them in; free text is what people give in practice.

**Black-box (C8/C9).**
- If C8 < C7: attribute the gap to the mechanisms in the data (policy variance across re-samples,
  over- or under-reacting to vocal segments), not to "LLMs are bad at control".
- If C9 closes most of the gap: this *supports* the thesis that the structure (bounds) matters;
  C9 remains non-auditable at the aggregation level.
- If C8 ≥ C7 on W: emphasize W_min, reproducibility and traceability, and state the trade-off plainly.

**Welfare criteria (E5).** Present them as a *choice for stakeholders*, made visible by the
architecture: utilitarian maximizes W, Rawlsian raises W_min at some cost to W. Do not declare a
single winner.

## 10. Limitations and ethics (must appear)
- The population is **simulated**. It is grounded in BRFSS and ATUS, but those data describe
  states and time use, not preferences. The mapping from data to preferences is an explicit
  assumption, tested by sensitivity analysis including a random mapping (E9).
- Intervention effects are calibrated to meta-analytic effect sizes by a virtual RCT. They are
  not estimated from intervention data.
- Persona LLMs are not real residents. We use them only to *express* hidden states. E8 checks
  their distributional fidelity. A human study is future work.
- The results are pre-NSHAP: network structure and loneliness persistence use literature defaults.
- IPW corrects between-segment non-response only.
- Ethics: no personal data are used (only public-use survey microdata and synthetic agents).
  In real deployment, feedback is sensitive data, and choosing a welfare criterion requires
  stakeholder participation. Automated screening must support staff, not replace their judgement.

## 11. Figures and tables (captions must be self-contained)
- **Fig. 1** Architecture: simulation (heterogeneous personas; feedback / silence) → diagnosis (LLM
  parse + silent screen) → control (IPW aggregation → welfare rule → bounded update → budget
  projection) → policy → back to simulation. Mark which quantities are hidden and which are observable.
- **Fig. 2** Pareto: accepted visits (x) vs. mean welfare W (y). Static frontier (C2 hull), with
  C0, C1, C3, C7r, C8, C9 and C10 marked. One panel per budget level, or combined.
- **Fig. 3** Voice bias: W_silent and W_min vs. silence strength s_ℓ for C4, C5, C6, C7r and C8.
- **Fig. 4** Welfare criteria (W vs. W_min) and robustness (relative drop vs. η and exaggeration).
- **Table 1** Main results (E1): W, W_min, W_silent, final ℓ, burden, visits, events, LLM calls; mean ± 95% CI.
- **Table 2** Audit (E6): replay identity, policy variance under re-sampling at T=0.1/0.7, trace
  completeness, influence concentration.
- **Table 3** (supplementary) Models (E7), sensitivity (E9), legacy replication, persona validity (E8).

## 12. Writing rules
- **Tone**: precise and modest. Every claim links to a figure, table or proposition. No hype words
  ("revolutionary", "human-level"). Avoid "full auditability"; say "control-layer auditability"
  and define it.
- **Numbers**: always report them with CIs and n (seeds). Paired Wilcoxon tests with Holm
  correction. Use "significant" only in the statistical sense.
- **Always pair outcome with resource**: loneliness or welfare gains are always reported
  together with visits used.
- **Don't claim** that the simulation predicts real-world effects, that the personas are
  real people, or that one welfare criterion is correct.
- **Definitions first**: define auditability (P1 properties), alignment, W_min and W_silent
  before using them.
- Double-blind: no author names, no links to identifying repositories (use an anonymous
  repository for code if needed), and a third-person self-citation.
- Format: ACM `sigconf` (AAMAS template), 8 pages + references. Use `\autoref`/`\cref`
  consistently. Tables use booktabs.

## 13. Abstract template (fill in placeholders)
> Adaptive social-care policies should respond to what people want, yet preferences are diverse,
> expressed in free text, and often unspoken by those most in need. We study preference-aware
> policy adaptation in an agent-based model of community care for older adults whose
> heterogeneous, hidden preferences are grounded in public survey data. We propose an
> architecture that separates *understanding* from *weighing*: an LLM interprets residents'
> feedback and screens silent residents, while explicit welfare rules aggregate the resulting
> signals with non-response correction and bounded, budget-feasible updates, keeping every policy
> change traceable. We prove replay, bounded-sensitivity and non-response properties, and evaluate
> against static Pareto policies, a state-only closed loop, survey and keyword baselines, informed
> black-box LLM control and an oracle. At equal resources, our approach [RQ1 result], improves
> the welfare of silent residents by [x]% over response-driven aggregation [RQ2], and yields
> reproducible decisions whose fairness trade-off is explicit [RQ3], across [k] LLM families.

## 14. Title candidates
1. *Listening to Diverse Voices: Auditable Preference-Aware Policy Adaptation with LLM Interpreters*
2. *LLMs Understand, Rules Decide: Fair and Auditable Adaptation to Heterogeneous Preferences in Agent-Based Social Care*
3. *Who Gets Heard? Preference Aggregation, Silence, and Auditable LLM-Assisted Policy Adaptation*

## 15. Inputs the writer will receive, and a checklist before submission
Inputs: `SPEC.md`; `results/*/summary.csv`; figures; `RESULTS.md`; `DECISIONS.md`; `data/MANIFEST.json`;
the MABS paper PDF (for context only).

Checklist:
- [ ] Every number in the text matches `summary.csv`; no placeholders left.
- [ ] Resource-matched comparison present; Max policy reported.
- [ ] All three propositions stated, with proofs (short proofs in the paper, full proofs in the supplementary material).
- [ ] Limitations cover data-to-preference mapping, persona validity, pre-NSHAP status and MNAR.
- [ ] Prior workshop paper cited in the third person; differences stated; no text reuse.
- [ ] All references verified; no `[CITE?]` left.
- [ ] Anonymized; within 8 pages; supplementary material prepared (ODD, parameters, prompts, extra results).
