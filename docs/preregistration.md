# KEEPER revision: protocol of the confirmatory experiments (E1–E9a)

This protocol is committed to the repository, and tagged `protocol-2026-11-11`, before the first Phase 4 job.
Every result file records the protocol's SHA-256 (the CPPR driver's `--prereg docs/preregistration.md`, and
the same field in every other runner). The evidence of the registration date is the commit and tag; no
third-party deposit has been made. A change after registration is logged in
`docs/preregistration_deviations.md` with its date, reason and effect; nothing is changed silently.

Terms. **CPPR**: credibility-preserving pseudonymous retention; its pools are pseudonymous, not anonymous.
**BFS tiers**: the legacy seeded endorsement tiers (paper Eq. 14). **BTP tiers**: budgeted trust propagation
(upgrade M-A); **capped** BTP limits the trust one endorsement transfers per step to `edge_cap_units · θ0`;
**uncapped** BTP is the approved rule without the cap (the ablation). **SCEM**: source-clocked escrow
memory; **M-B**: SCEM witnesses weighted by the causal BTP tier. **Target**: the account an attack tries to
move. **Gain**: the change of a target's reputation proxy (the panel's mean predicted trust,
`keeper.laundering.core.Panel.reputation`) in the attack's direction (+ for whitewash and direct laundering,
− for bad-mouthing). **Control-adjusted gain**: a target's gain under an attack arm minus its gain under the
control arm of the same run (or of the E3 run with the same seed and tier settings), which removes the
effect of background deletions and retraining.

## 1. Hypotheses, endpoints and decision rules

**Statistical hypotheses.** A hypothesis is **supported** when (i) the 95% headline interval (Section 4) of
every component lies entirely on the stated side, and (ii) its intersection–union p-value (the largest
component p-value) survives Holm's correction over the eight statistical hypotheses (H1, H2w, H2b, H3b, H6a,
H6b, H7, H8) at family-wise 5%. A component whose two-sided interval lies entirely on the wrong side is
reported as **contradicted**. A component with results for fewer than 80% of its seeds is not evaluable,
and its hypothesis is reported as not supported. Otherwise the result is reported as not supported, in the
paper section that makes the claim. A removal threshold (the "≥ 0.8" parts of H1, H2w and H2b; the "≥ 0.5"
part of H8) is read three ways: **met** when the interval's lower bound reaches the threshold;
**inconclusive** when the lower bound is below the threshold and the upper bound reaches it (the paper says
"removal not established"); **partial** when the upper bound is below the threshold (the paper says the
mechanism "reduces" the effect, with the measured share).

**Exact properties.** H3a and H5 are properties that hold by construction. They are checked in every run. A
violation is a code defect: the runs stop, the defect is fixed, the runs are repeated and the event is
logged.

**Deterministic hypothesis.** H4 is computed exactly over every victim record of a deterministic engine (a
census, with no sampling). It is supported when its rule holds on each network, and it is not part of the
Holm family. Unlike an exact property, it can fail through SCEM's indirect evidence paths (Section 2.3).

| ID | Claim | Exp. | Endpoint (per unit) | Rule |
| --- | --- | --- | --- | --- |
| H1 | CPPR removes direct laundering | E3 | edge-level target: (a) gain under Plain Erasure minus gain under CPPR-BTP; (b) the laundering cut 1 − ḡ_adj(CPPR) / ḡ(Plain Erasure), where ḡ_adj(CPPR) is the mean control-adjusted CPPR gain and the Plain Erasure gain is not control-adjusted (the protocol has no Plain Erasure control) | (a) interval > 0 and (b) interval lower bound ≥ 0.8, on each network |
| H2w | CPPR confines whitewash fakes without attack edges | E3 | whitewash target: (Plain VPE_ww − Plain VPE_ctl) − (CPPR_ww − CPPR_ctl); suppression 1 − mean(CPPR_ww − CPPR_ctl) / mean(VPE_ww − VPE_ctl) | difference interval > 0 and suppression lower bound ≥ 0.8, on each network |
| H2b | Same for bad-mouth fakes | E3 | bad-mouth victim: the same two quantities with losses (minus gains) and the control arm scored on the victims | as H2w |
| H3a | BTP admits a bounded number of fakes per attack edge | E3, E4 | fakes in Tier ≥ 1 per target group, capped BTP | ≤ floor(K · g · edge_cap_units) in every run and group (0 when g = 0) |
| H3b | Under one attack edge, the attacker's best whitewash gain is smaller against capped BTP than against BFS | E4 | per seed s and method: the best-response cell is the g = 1 cell with the largest mean control-adjusted CPPR gain over the other seeds (leave one seed out); seed s contributes its own mean gain in that cell; the endpoint is BFS's minus capped BTP's | interval > 0 on each network |
| H4 | M-B makes witness laundering cost attack edges | E1 | per victim record, the revoked share of the warner's escrowed warning, attack minus clean (the total effect, through every evidence path) | with M-B, at g = 0, lag 1, k ∈ {5, 50}, ring off and on: the mean over victim records ≤ 0.01 on each network (the 95th percentile and the maximum are reported) |
| H5 | Commitment bounds the window of vulnerability | E1 | per eligible victim record (Section 2.3), attack minus clean revoked share; k = 5, g = 0, M-B off | for every committing policy P: ≤ 1e-12 for every record eligible under P, and no_commit's mean over the records eligible under P ≥ 0.01; a policy with no eligible record is reported as untested |
| H6a | SCEM's revocation (not its delay) reduces a silent sender's effect | E2 | target: attack gain under SCEM without hazard minus attack gain under SCEM (learned φ); lag 1 | interval > 0 on each network, for whitewash and for bad-mouth |
| H6b | SCEM reduces a silent sender's effect relative to the S3R backbone | E2 | target: attack gain under S3R LocalGDTE minus attack gain under SCEM (learned φ); lag 1 | interval > 0 on each network, for whitewash and for bad-mouth |
| H7 | An insider cannot link CPPR's pooled verdicts (criterion G3) | E6 | per seed: the maximum over the attackers of Section 2.8 of the G3 linkage AUC on the direct arm's CPPR-BTP verdicts | upper interval bound < 0.60 on each network |
| H8 | The laundering gain is not an artifact of the missing self term | E9a | target: (a) U1 gain with `self_term` on; (b) the ratio ḡ_on / ḡ_off of the mean U1 gains with the self term on and off | (a) interval > 0 and (b) interval lower bound ≥ 0.5, on each network |

**Reading of nulls, fixed in advance.** If H3b is not supported, the paper states: "BTP bounds Tier ≥ 1
admission per attack edge (H3a); under the count gate, the attacker's best gain per attack edge is not
reduced." H1(b), H2w(b), H2b(b) and H8(b) follow the three-way reading above. If H7 is not supported, the
paper states that criterion G3 is unmet. M-B is claimed to remove the direct witness channel for Sybils
that have received no rating, to bound it at w_0 · κ ≤ 0.01 per Sybil otherwise, and to bound it per
attack edge (Property 1, a direct-channel bound proved for streams in which only the attack reaches the
warner and unit-tested in Task 3.3). On real streams the total effect also flows through SCEM's indirect
paths (V2 surprise, durable tallies, lag survival); H4 measures that total, and the residuals with a Sybil
ring and with attack edges are reported (Section 2.3). If H4 fails, the paper reports M-B as reducing, not
removing, witness laundering, with the measured residual.

**Secondary outcomes** (intervals reported, no decision rule): utility (test MCC against the ungated
backbone and against Plain Erasure, E3); CPPR-BFS and uncapped CPPR-BTP on the E3 endpoints; account-level
direct laundering; CPPR's overshoot (cut > 1.2); every E4 cell and admission by endorser rank; the
uncalibrated-operator sensitivity (Section 2.6); the Regime-2 baselines (E5); top-1 linkage against chance
(E6); KEEPER combined (E7, descriptive: no superiority conclusion is drawn from it); the calibrated cold
start (E8, descriptive); the self-term-off gain and edgeless shares (E9a); the release-policy trade-off, the
witness-attack residuals and the granularity study (E1); the SCEM ablations, the lag-3 streams and both
models' learned φ (E2).

## 2. Fixed settings

### 2.1 Common

- Networks: Bitcoin-Alpha, Bitcoin-OTC and Wiki-RfA.
- Seeds: E3 uses seeds 5–14. E2 uses seeds 5–14 for its H6 models (S3R, SCEM learned φ, SCEM without hazard)
  at lag 1 with their clean twins, and seeds 5–9 for everything else. E4, E5, E7 and E9a (with E8) use
  seeds 5–9. E1 and E6 are deterministic given their inputs. Seeds 0–4 produced the stored results and are
  not used for any confirmatory claim. The targets are drawn per seed, so some targets also appeared in
  seeds 0–4 (E3 edge targets: 44 of 80 distinct on Alpha, 38 of 126 on OTC, 36 of 98 on Wiki-RfA); a
  sensitivity analysis restricted to targets unused in seeds 0–4 is reported for H1–H3b.
- Draws: the CPPR workload and target permutation use `default_rng(3000 + seed)`, the honest background
  `default_rng(int(1e6·ρ) + 17·seed)`; probe4's target draw depends only on the dataset (rng 7). Every
  training calls `seed_all(seed)` through `core.make_model`; new runners build every model through
  `core.make_model` (which re-pins cuDNN's benchmark flag) and log determinism warnings.
- Hardware: NVIDIA RTX A6000, deterministic mode (`CUBLAS_WORKSPACE_CONFIG=:4096:8`), one visible GPU per
  process. Every run executes from a git worktree fixed at one commit, recorded with the result.
- Backbone: LocalGDTE, 150 epochs, code defaults. In the CPPR driver every model (base, Plain Erasure, Plain
  VPE, CPPR, baselines) uses the same tier-fed count gate (Eq. 15) with tier weights (0.01, 0.40, 0.70,
  1.00); H1 therefore isolates retention. The ungated backbone appears only in the utility comparison.

### 2.2 BTP (upgrade M-A) and attack-edge endorsers

- Laziness 0.5; seeds S0 = accounts with more positive than negative in-edges in snapshots 0–4;
  K = ceil(log2 n_ref); θ0 = eps / n_ref, n_ref = active accounts of the honest calibration graph.
- Eligibility (ruling R56): an account is eligible for Tier ≥ 1 when it is trusted (trust ≥ θ0) and its
  positive in-edges from trusted raters outnumber its negative in-edges from trusted raters (the
  credible-negatives guard). Untrusted accounts therefore can neither confer trust nor remove eligibility.
  The rule that counts every rater's negative in-edges (Task 3.1's) is the ablation.
- The calibration (S0, K, θ0) is computed once, on the honest pre-attack training graph of the run, and used
  for every tier computation of that run (ruling R32); fakes never become seeds or move K or θ0.
- Capped BTP (the KEEPER default): edge_cap_units = 2.0; eps = 0.02 (Alpha), 0.01 (OTC), 0.5 (Wiki-RfA);
  K = 12, 13, 14; n_ref = 3629, 5684, 10152; guarantee floor(2Kg) = 24, 26, 28 at g = 1. Uncapped BTP:
  eps = 0.05, 0.05, 0.5. Both eps were chosen on honest data only (`results/analysis/tiers_report.json`) as
  the largest value in {1, 0.5, 0.2, 0.1, 0.05}, extended by {0.02, 0.01, 0.005} when none qualifies, that
  keeps at least 90% of the BFS Tier ≥ 1 accounts. The cap (ruling R37) and its size (R38) were set after
  the no-training admission probe and before any trained attack run. The credible-negatives guard (R56) left
  eps, K, n_ref, the guarantee and the best star unchanged (Task 3.1b, `results/analysis/tiers_report.json`). Every analysis asserts that each run's
  recorded eps, K, n_ref and edge_cap_units equal these values.
- Random endorsers (rulings R39, R43): one pool for every tier method — honest accounts with base-graph tier
  ≥ 2 under BFS and under the reference capped BTP calibration, excluding the arm's targets, already in BFS's
  trusted set when the single pass reads the group's attack edges, and with capped inflow B(u) ≥ θ0 (a
  necessary condition for admitting any fake under capped BTP). g distinct accounts per group are drawn from
  `default_rng((20261027, seed, arm index))`, so BFS, capped and uncapped runs of the same seed face the same
  attack edges. The share removed by each criterion is recorded per run.
- Max-B(u) endorsers (ruling R44): group i receives the accounts ranked (i − 1)g + 1 … ig by B(u) among
  non-target accounts (ties to the smaller id): the capped B(u) for BFS and capped-BTP runs, the run's own
  uncapped B(u) for uncapped runs. Results are also reported by rank.
- Attack edges go to the group's fakes round-robin (ring) or to the hub (star), at the group's snapshot,
  before the group's own ratings.

### 2.3 E1: SCEM engine (CPU, deterministic)

- Stream: each network's training stream (snapshots 0–20). Engine `run_scem_streaming_weights_np` with its
  defaults (L_f = 1).
- E1a, release policies without attack: arrival; exponential (N = 1, φ = 0.25); Erlang N = 2 (φ = 0.5), N = 3
  (φ = 0.75), N = 5 (φ = 1.25), all with mean delay 4 snapshots; quarantine (D = 4, then commit); no_commit
  (never released or closed; the durable tallies behind κ and V2 read the surviving weight retroactively,
  ruling R57, so that no_commit differs from Erlang only in commitment);
  each with M-B off (the stored witness rule) and on (capped BTP causal tiers on the clean stream, frozen
  calibration; uncapped as a sensitivity). Metrics: silent-sender revocation = mean of 1 − kept weight over
  ratings of flagged senders (flagged as in `stage_b.engine_metrics`: at least two negative ratings received
  after the rating and more negative than positive), under the default rule and the future-only rule (the
  sender had received no negative rating at or before the rating's snapshot); honest mass revoked = mean of
  1 − kept weight over ratings of senders never flagged (per sender, over the whole stream); All Retained;
  latency = released share of surviving mass at ages 1 and 4.
- E1b, adversarial witnesses (Erlang N = 2): victims are records W → T with T in the `pick_targets` cohort
  (distrust majority, at most 3 distrust in-edges). The records are split into batches so that a batch holds
  at most one record per warner and k × (records in the batch) ≤ 2% of n_ref (at k = 50, one record per
  batch), so that the injected Sybils never flood the network's tiers; each batch is one attacked stream.
  For each record, k fresh Sybils rate W negatively at t_k + lag; with `sybil_ring` and k ≥ 2 the Sybils
  also form a positive directed ring (s_1 → … → s_k → s_1); g attack edges go to Sybils 1..g from endorsers
  drawn uniformly, from `default_rng((20261027, network index, batch, cell index))`, among honest accounts
  whose causal capped tier at the attack-edge snapshot is ≥ 2 (never W or T). Timing (ruling R34): attack
  edges and ring at max(t_k + lag − 1, t_seed + 1), bad-mouthing at t_k + lag ≥ that snapshot + 1; a
  κ-farming variant places the ring at t_seed + 1, the earliest allowed snapshot; records whose
  bad-mouthing would fall at or before t_seed + 1 or after snapshot K_snap − 1 = 20 are skipped and
  counted. M-B tiers are computed causally on each stream with the frozen clean-stream calibration.
  Grid: M-B off/on × k ∈ {1, 5, 10, 50} × g ∈ {0, 1, 2} × lag ∈ {0, 1, 3, 7, 14}, ring on; ring off at
  k = 5, g ∈ {0, 1}, lag 1; the κ-farming variant at k ∈ {5, 50}, g = 0, lag 1. Revoked share of a record =
  1 − ω_k − π_k at the end of the stream (from `return_records`). Two differences per record: attack minus
  clean (the total effect; H4), and attack minus the comparator stream, which is the attacked stream
  without the bad-mouthing edges, evaluated with the attacked stream's witness levels (the direct witness
  channel). Reported per cell: census mean, 95th percentile and maximum of both differences over the victim
  records; the share of victim records (records sent by the warners W) whose direct-channel difference lies
  within Task 3.3's `property1_bound` (descriptive: the bound is guaranteed only for records not sent by a
  witness, under the conditions stated in keeper/scem/property1.py); and the honest accounts whose tier
  differs from the clean stream (0 by construction under the credible-negatives guard).
- E1c, window property (H5): policies exponential, Erlang N = 2, 3, 5, quarantine and no_commit; M-B off;
  k = 5, g = 0, ring off. A record is eligible for policy P when lag + L_f ≥ A_P and
  t_k + lag + L_f ≤ K_snap − 1, with A_P = 18.42, 13.28, 11.21, 9.28 snapshots for exponential and Erlang
  N = 2, 3, 5, and 4 for quarantine; the lags are those of E1b. E1c is repeated on the 88-snapshot stream of
  E1d with lags {0, 4, 12, 28, 56} (the 21-snapshot lags scaled by 4).
- E1d, granularity (M-C): the full stream of each network re-cut by the loader's own snapshot rule into 22,
  44 and 88 snapshots; engine constants kept in snapshot units; t_seed and the causal calibration's seed
  window scale with the cut (t_seed = 4, 9, 19); Erlang N = 2 and arrival, M-B off and on; E1a metrics and
  the horizon in days; on 88 snapshots also E1c's policies for H5.

### 2.4 E2: SCEM end to end (GPU)

- Models are ungated. SCEM models are `keeper.backbone.local_model_scem.LocalGDTE`, built through
  `core.make_model` with the model class set as in `keeper.scem.stage_b.cmd_model`; M-B is on in every SCEM
  model, with tiers computed causally on each stream under the frozen clean-stream capped calibration.
  Models: S3R LocalGDTE; SCEM with learned φ; SCEM with φ fixed at 0.5 (G5); SCEM without hazard
  (λ_max = 0); SCEM with N = 1. The learned φ of every SCEM model is reported, so that H6a's contrast can be
  read against any difference in delay.
- `default_rng(5000 + seed)` draws, in this order: 15 whitewash targets from the `pick_targets` cohort; 15
  bad-mouth targets from the trusted cohort (at least 3 trust in-edges and no distrust in-edge); each
  target's snapshot t_s, uniform on the integers 5–14; then, for lag 1 and afterwards for lag 3, each
  silent sender's two witnesses, without replacement among accounts whose causal capped tier at t_s + lag
  is ≥ 2, excluding the target.
- Per target, 3 fresh silent senders rate it once at t_s (+1 whitewash, −1 bad-mouth) and do nothing else;
  each silent sender is rated negatively by its two witnesses at t_s + lag. One attack stream per lag
  (lag ∈ {1, 3}) holds both target kinds; the clean twin is the unmodified stream.
- Endpoint: attack gain = (proxy under attack − proxy on the clean stream) in the attack's direction, from
  the final embedding, with the panel built on the clean training graph; also the percentile shift and test
  MCC.

### 2.5 E3: CPPR confirmatory (GPU)

- Per network and seed (5–14), runs of `keeper.cppr.cp_vpe_gpu` with
  `--arms direct,whitewash,badmouth,control --hold-test-snapshot --dump-targets DIR`: capped BTP (also
  `--ungated-baseline --dump-verdicts DIR`), BFS (also `--dump-verdicts DIR`), and uncapped BTP on seeds 5–9.
- Code defaults: 15 edge-level and 5 account-level targets, ρ = 0.02, k_fake = 3, g = 0, ring region.
  Plain Erasure keeps the original snapshots; every model is scored on the original test snapshot. The
  control arm retrains Plain VPE and CPPR on the original graph with the background deletions only and is
  scored on the whitewash targets and the bad-mouth victims.

### 2.6 E4: attack edges (GPU)

- Per network and seed (5–9), `--arms whitewash --hold-test-snapshot --methods cppr --dump-targets DIR`.
  Cells: g = 1 with random endorsers ring s ∈ {3, 50} and star s = 50, and with max-B(u) endorsers star
  s ∈ {3, 50} and ring s = 3; g = 2 with random endorsers ring s = 3 and max-B(u) endorsers star s = 3.
  Each cell runs under BFS and capped BTP; uncapped BTP runs only the two g = 1 max-B(u) star cells
  (s ∈ {3, 50}): 18 runs per network and seed, 270 in all. s is `--k-fake`. The capped-BTP cell g = 1,
  random, ring, s = 3 also trains Plain VPE (`--methods plain_vpe,cppr`), because E5 pairs with it. Every
  E4 run uses the base-model cache (`--base-cache DIR`): a seed's base model is trained once per tier
  setting and reloaded after a bit-identity check of its reputations and test metrics, which changes no
  output.
- Gains are control-adjusted with the E3 control arm of the same seed and tier settings; the analysis
  asserts identical base reputations r0.
- No-training analyses (CPU, exact): admission per group in the whitewash topology (every fake rates its
  target; common random pool; ranked max-B(u) endorsers) against g ∈ {1, 2, 5}, s ∈ {3, 10, 50, 350}, ring
  and star, random and max-B(u); the tiers report's figures for regions that rate no target are labelled
  "pure region". The uncalibrated-operator sensitivity: admission in every E4 cell when the calibration is
  computed on the attacked training graph. The demotion census: for k ∈ {1, 5, 10} fresh Sybils per victim,
  each bad-mouthing one honest Tier ≥ 2 account (100 victims, one per stream), the share of victims whose
  tier drops under BFS, under BTP with the all-raters guard, and under BTP with the credible-negatives guard
  (0 by construction). Victims are drawn from each method's own Tier ≥ 2 accounts (the main reading) and,
  as a secondary reading, from a pool common to all methods.

### 2.7 E5: Regime-2 baselines (GPU)

- Methods (driver `--methods`): retain with author (the erased ratings stay with their authors inside the
  model; for the direct arm this is the base model, for whitewash a model trained on the injected graph
  without erasure); per-rating pseudonym (each erased rating is re-emitted at its snapshot from its own
  fresh node whose gate weight is its rater's tier weight); withdrawal counts (Plain Erasure's graph, with
  log1p of the positive and of the negative ratings withdrawn from each account appended as two input
  columns before the model is built; the tier-prototype overwrite applies to the original columns only).
- Arms: direct and whitewash at g = 0, and whitewash at g = 1 (random endorsers, ring, s = 3), capped BTP,
  seeds 5–9. Plain Erasure, Plain VPE and CPPR come from the E3 (g = 0) and E4 (g = 1, random ring s = 3)
  runs with the same seed and flags; the analysis asserts identical r0, snap_f and endorsers.
- Linkage candidates for the per-rating pseudonym are the members of the CPPR pool that would have held the
  rating (same window and tier).

### 2.8 E6: linkage (CPU)

- G3 linkage AUC: the mean over (pooled verdict, true rater) items of the within-pool AUC of the true rater
  against the other pool members (ties count ½), on the direct arm's verdicts; single-member pools are
  excluded from the AUC, their share is reported, and a sensitivity scores them as AUC 1.
- Attackers: the hand-crafted combined attacker (`keeper.cppr.linkage.linkage_attack`); each single signal
  (structure, timing, model probability, degree, activity, in-window activity), cross-fitted; and a learned
  attacker, `HistGradientBoostingClassifier(random_state=0, early_stopping=False)` on the dump columns
  structure, timing, model_prob, degree, activity, activity_window, candidate_tier, pool_size, pool_tier,
  trained leave one seed out.
- Applied to CPPR-BTP and CPPR-BFS (E3) and the per-rating pseudonym (E5). Test for H7: one-sided t on the
  per-seed AUC − 0.60 with the Nadeau–Bengio corrected variance (factor 1 + J · n_test / n_train, J = number
  of folds); bootstrap unit: the pooled verdict within seed.

### 2.9 E7: KEEPER combined (GPU, descriptive)

- Backbone: rating-level LocalGDTE-SCEM with M-B and the CPPR gate, capped BTP, built through
  `core.make_model` with the model class set as in `keeper.scem.stage_b.cmd_model`; pool nodes take their
  pool's tier, including for M-B. Seeds 5–9.
- Arms: direct laundering (CPPR-BTP and Plain Erasure); whitewash at g = 0 and g = 1 (random endorsers, ring,
  s = 3; CPPR-BTP and Plain VPE); witness laundering on a stream without deletions, where 5 Sybils per
  warning bad-mouth the honest warners of the direct targets at lag 1 (ring, g = 0 and g = 1), with M-B on
  and off.
- Reported: target gains, fake gains, test MCC, and the gain of the witness arm with M-B on and off.

### 2.10 E8: calibrated cold start (descriptive)

- Inputs: the E9a self-term-off runs (seeds 5–9). They dump every active account's base proxy, the
  test-edge probabilities and, for each target, every active account's proxy re-scored on that target's
  post-U1 graph (as probe4 does for its percentile after deletion).
- r̃(v) = (n_v · r(v) + m · r_prior) / (n_v + m), m ∈ {1, 2, 4}; n_v = live incoming training ratings (after
  U1 for the target); r_prior = 10th percentile of the active accounts' base proxies. Edge probabilities
  p̃ = (n_v · p + m · r_prior) / (n_v + m) for edges into v.
- Reported: each target's percentile of r̃ among all active accounts on its own post-U1 graph; the mean
  percentile change (r̃ against r, base model) of accounts with at most one incoming training rating; test
  MCC on every row of the test snapshot (`cold_start.held_out_snapshot`, which includes rows with an endpoint
  unseen in training) whose destination has at most two incoming training ratings, including zero, with the
  threshold rule of `core.test_metrics`.

### 2.11 E9a: self-term ablation (GPU)

- LocalGDTE (ungated) with `self_term` on and off, seeds 5–9. Targets: probe4's 50 drawn targets per network,
  and, as a separate stratum, every account with a distrust majority and at least 4 distrust in-edges.
- U1 (unlink) removes all of a target's distrust in-edges from the inference graph without retraining.
  Reported: gain, percentile after U1 and edgeless share, with the self term on and off; H8 uses the 50
  drawn targets, unweighted (probe4's post-stratified mean is reported alongside).

## 3. Units and data handling

- The unit is the target, except: the victim record in E1 (records cluster within warners; census
  statistics), and the pooled verdict in E6.
- Targets recur across seeds (E3 edge targets: 45 of 80 distinct targets appear in two or more seeds on
  Alpha, 24 of 126 on OTC, 40 of 98 on Wiki-RfA; E9a uses one draw for every seed). Seed and target are
  therefore treated as crossed, with missing cells, in every target-level interval and test.
- Exclusions: none beyond the code's eligibility rules. A run that crashes is rerun once with the same seed;
  a run failing twice is reported missing and never replaced by another seed. A paired difference uses the
  same seed and target in both arms; a missing pair is dropped and counted.

## 4. Intervals and tests

- Point estimate: the mean of per-seed means over the available pairs, each seed weighted equally.
- Headline 95% interval: the hull of (a) the crossed bootstrap percentile interval (10,000 replicates; seeds
  and target ids resampled independently, `crossed_bootstrap` with missing cells) and (b) the t-interval
  over per-seed means (df = seeds − 1). Ratios (cut, suppression) use the ratio of seed-averaged means with
  the corresponding ratio intervals. Proportions of 0/1 outcomes at exactly 0 or 1 use Clopper–Pearson over
  distinct units. Every interval is clipped to its support. A nested bootstrap is reported as a sensitivity.
- Component p-value for a difference tested against 0: the larger of the one-sided seed-level paired t
  p-value and the crossed-bootstrap p-value (1 + #{replicates ≤ 0}) / 10,001. For a ratio tested against a
  threshold c (0.8 or 0.5): the larger of the one-sided linearized seed-t p-value of (ratio − c), using the
  residuals of `ratio_seed_t_interval`, and (1 + #{replicates < c}) / 10,001. H3b: each crossed replicate
  resamples seeds and target ids, recomputes every cell's mean gain on the resampled targets, then each
  method's leave-one-seed-out best-response cell, the per-seed contributions and their difference; the
  seed-t uses the per-seed contributions. H8(a) is a one-sample test of the mean gain against 0. H7 uses
  the Nadeau–Bengio corrected t of Section 2.8 for both its p-value and its headline interval. Since the
  hull contains the seed-t interval, rule (i) of Section 1 already implies a one-sided seed-t p < 0.025.
- Expected precision: from the stored seeds 0–4 (BFS, before control adjustment), the suppression of
  whitewash gains on Alpha was 1.01 with standard error 0.34, so the projected 95% lower bound at 10 seeds
  is about 0.45 even under full suppression; Wiki-RfA's cut (0.96) and suppression (0.89) project lower
  bounds near 0.79. The removal parts of H1 and H2 may therefore be inconclusive on some networks, and the
  three-way reading of Section 1 applies.
- Power: with α = 0.05/8 (Holm's first step), a one-sided t needs d_z = 2.48 with 5 seeds and 1.28 with 10
  seeds for 80% power (d_z = mean of per-seed differences / their standard deviation; 1.36 and 0.85 at
  α = 0.05). The stored
  seeds give d_z 3.7–4.5 for H1 and 1.3–7.1 for H2; H3b, H6 at lag 3 and H8 rest on 5 seeds and may be
  underpowered, which is stated with their results.

## 5. Amendments, clarifications and reporting

- A choice left open here for a runner not yet written is fixed in code before that experiment's first run
  and logged in the deviations file, with its commit, as a clarification.
- A mechanism revised after a failed hypothesis is evaluated only on new seeds (15–24), under a dated
  amendment committed before those runs; the original result stays in the paper.
- Every outcome, including an unsupported or contradicted hypothesis and a missing run, is reported in the
  paper section that makes the claim. Numbers are generated from the result files; none is typed by hand.

## 6. Command templates (CPPR driver)

Common: `env -u PYTHONPATH -u VIRTUAL_ENV ~/.conda/envs/ps/bin/python -m keeper.cppr.cp_vpe_gpu --ds DS
--seed S --epochs 150 --hold-test-snapshot --prereg docs/preregistration.md --out ABS.json`.

- Capped BTP: `--tiers btp --btp-eps EPS_C` (EPS_C = 0.02, 0.01, 0.5; the cap defaults to 2.0).
- Uncapped BTP: `--tiers btp --btp-eps EPS_U --btp-edge-cap none` (EPS_U = 0.05, 0.05, 0.5).
- BFS: `--tiers bfs` (no `--btp-eps`).
- Every run with attack edges, under every tier method: `--pool-eps EPS_C` (the reference capped
  calibration of the endorser protocol, Section 2.2).
- E3: `--arms direct,whitewash,badmouth,control --dump-targets DIR [--dump-verdicts DIR] [--ungated-baseline]`.
- E4: `--arms whitewash --methods cppr --attack-edges G --k-fake S --sybil-topology {ring,star}
  --endorser-select {random,max_trust} --dump-targets DIR --base-cache DIR` (`--methods plain_vpe,cppr` in
  the one cell of Section 2.6).
- Task lists are generated by `keeper.cluster.cppr_tasks` from `keeper/cluster/protocol_settings.py`; the
  results root is the main repository, never a run worktree.
