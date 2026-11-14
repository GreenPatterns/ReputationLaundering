# Deviations from, and clarifications of, the protocol in docs/preregistration.md

Each entry gives its date, what changed, why, and its effect. Entries are added before the runs they concern.

## 2026-11-11: GPU model per experiment family

**Protocol text (Section 2.1).** Every run uses an NVIDIA RTX A6000.

**Change.** The CPPR family (E3, E4, E5) runs on RTX A6000 as registered. The SCEM and backbone family (E2, E7, E9a with E8) runs on NVIDIA L40S. Within a family every run uses one GPU model, and each result records the model it ran on.

**Reason.** On 2026-11-11 the cluster had two free A6000 GPUs, which E3 and E4 occupy for more than a day. L40S GPUs were free.

**Effect.** No comparison in the protocol pairs a run of one family with a run of the other. E5 pairs with E3 and E4, and all three stay on A6000. E9a's U1 gains follow the definitions of the stored attack results, which were produced on L40S. Deterministic mode and one visible GPU per process are unchanged.

## 2026-11-11: registration tag

The protocol was committed on 2026-11-11 (commit 150d85c, tag `protocol-2026-11-11`), not on 2026-10-27 as drafted, because the first confirmatory job started on 2026-11-11. No confirmatory run existed before that commit.

## 2026-11-11: E1 clarifications, fixed in code before the first E1 run (commit of this entry)

Section 5 of the protocol lets a choice that the text leaves open be fixed in code before the experiment's first run and logged here. The E1 runner (`keeper/scem/e1_runner.py`) fixes these:

1. **Batch cap (Section 2.3, E1b).** A batch holds at most `n_ref // (50 k)` victim records, which is the rule "k × records ≤ 2% of n_ref" as written. At k = 50 this allows 1 record per batch on Bitcoin-Alpha, 2 on Bitcoin-OTC and 4 on Wiki-RfA; the protocol's parenthesis "one record per batch" holds on Bitcoin-Alpha only.
2. **Batch construction.** A cell's batches are built from the records it does not skip, first-fit in record-id order.
3. **Cell index of the endorser draw.** It is the cell's position in the attack grid and does not depend on M-B, so M-B off and on face the same attacked stream.
4. **Added cell.** H4 names k = 50 with the ring off, which the E1b grid lacked. That cell (g = 0, lag 1) is added; the grid has 65 cells.
5. **Honest mass revoked.** The mean over ratings of senders that are never flagged anywhere in the stream; the mean of per-sender means is reported next to it.
6. **Latency at age a.** The mean of ω / (ω + π) at snapshot t_k + a − 1 over records with positive surviving mass.
7. **E1d streams.** The full stream, held-out snapshot included, is re-cut into 22, 44 and 88 snapshots; t_seed (4, 9, 19) scales the M-B rule, the calibration and the attack timing. With M-B off, the engine's stored witness rule keeps its fixed bootstrap floor at snapshots 0 to 4 in every cut, because scaling it would change the engine.
8. **H5 eligibility.** Only attacked records count; a record skipped under the timing rule is never eligible. The check "≤ 1e-12" is one-sided as written, and the largest absolute difference is reported too.
9. **H5 scope.** H5 is evaluated per network on the 21-snapshot and on the 88-snapshot stream, pooling (lag, record) pairs.

## 2026-11-11: correction of a statement in Section 2.3

**Protocol text.** E1b reports "the honest accounts whose tier differs from the clean stream (0 by construction under the credible-negatives guard)".

**Correction.** "0 by construction" holds without attack edges (g = 0) only. With g ≥ 1, the fakes that an attack edge admits join the eligible accounts and shift the tercile cut points, so some honest accounts move between Tiers 1 to 3; the runner's smoke test on Bitcoin-Alpha found 1 to 87 such accounts per batch. The count is reported for every cell, and the paper states the g = 0 guarantee only. No hypothesis depends on the statement.

## 2026-11-11: E5 clarifications, fixed in code before the first E5 run (commit of this entry)

The E5 baselines (`keeper/cppr/baselines.py`, Section 2.7) fix these open choices:

1. **Which erased ratings are retained.** The ratings that CPPR would keep: every rating of an edge request and every rating that a deleted account issued about a live account. Ratings about a deleted account are dropped, as under CPPR. The per-rating pseudonym and the withdrawal counts use the same set.
2. **Runs without CPPR.** An E5 run trains the baselines only; Plain Erasure, Plain VPE and CPPR come from the paired E3 and E4 runs. The stored releases are still built in every run, so every random draw is that of the paired run; the analysis asserts identical r0, snapshot draws and endorsers.
3. **Withdrawal counts on an injected arm.** The arm's graph without the erased rows, with the snapshots kept, as Plain VPE and CPPR do.
4. **Per-rating pseudonym.** Each re-emitted rating keeps its own snapshot and timestamp; its linkage candidates are the members of the CPPR pool that would have held it.
5. **Gains.** Baseline gains are not control-adjusted, because no baseline has a control run; they are compared with CPPR and Plain VPE, and with Plain Erasure on the direct arm.
6. **Known confound of the withdrawal-count baseline.** Its model has two more input columns than the base model, hence a different first projection and different initial weights, while the reference reputation r0 comes from the base model. Its gain therefore mixes the effect of erasure with the effect of the changed architecture. The result is reported with this caveat; a reference run of the wider model without erasure would be an amendment on new seeds.

## 2026-11-11: E2 clarifications, fixed in code before the first E2 run (commit of this entry)

The E2 driver (`keeper/scem/e2_driver.py`, `keeper/scem/e2_streams.py`, Section 2.4) fixes these open choices:

1. **SCEM without hazard.** The module takes a flag `hazard` (default on, which leaves the module byte-identical; the source pin in `tests/test_scem_equivalence.py` covers it). With the flag off the hazard is zero in every channel, which is the forward pass of λ_max = 0 but with finite gradients (λ_max = 0 itself gives undefined gradients in 22 parameters). The supersession rule stays on.
2. **Witnesses.** Each silent sender gets two distinct witnesses; one witness may serve several silent senders. The pool is the accounts whose causal capped tier in row t_s + lag of the clean stream is at least 2, without the target. All 30 snapshots t_s are drawn in one call, whitewash targets first.
3. **Node counts.** The clean twin has the network's node count and an attack stream has 90 more (the 90 silent senders), as in the CPPR driver; the standardisation of the input features therefore differs slightly between twin and attack stream.
4. **Percentile.** A target's percentile is the share of the clean graph's active accounts whose reputation is strictly lower, under the same model.
5. **Recorded M-B levels.** Each result stores a digest of the witness levels, the per-snapshot tier counts and the witnesses' levels, not the full array; the analysis checks eps, the cap and the guard, and that each network uses one calibration.
6. **Secondary outcomes.** Mean gain, percentile shift and the contrast with SCEM (learned φ) at lags 1 and 3, with intervals only. The φ of the fixed-φ model is recorded as 0.5.
7. **Integrity.** Every run needs a clean git state, and the commit must be the same within each (network, seed, model) so that a clean twin and its attack run come from one code state.

**Note for reading H6a.** With a learned φ the commitment age (53.1 snapshots at N = 2, 73.7 at N = 1) exceeds the 21 training snapshots, so only the fixed-φ model commits records during training. H6a compares SCEM with and without its hazard at a learned φ, so it tests revocation inside an open escrow window, not commitment.

## 2026-11-11 (update): E2 and E7 run on A6000 after all

Four more A6000 GPUs became free at about 18:00 local time, before any E2 or E7 run had started. E2 and E7 therefore run on RTX A6000 as the protocol registered. The only experiment on L40S is E9a (with E8, which reads E9a's runs). The earlier entry "GPU model per experiment family" stays as the record of what was planned at that time.

## 2026-11-11: H5 (window property) flagged by the batched design; test corrected to one record per stream

**Event.** The E1 analysis flagged H5, an exact property, in several blocks: on the 21-snapshot streams the quarantine policy on Bitcoin-OTC (largest difference 1.7e-4) and Wiki-RfA (9.4e-5); on the 88-snapshot streams every policy on Bitcoin-OTC (up to 3.4e-4) and Wiki-RfA (up to 0.064). The Erlang policies on the 21-snapshot streams, KEEPER's setting, showed exactly 0 on all three networks.

**Diagnosis.** Section 2.3 attacks several victim records in one stream (up to 14, 22 and 42 on the three networks at k = 5). A record is eligible for H5 when its own attack takes effect after its commitment age, but the attacks on the other records of its batch fall inside its escrow window and reach it through the engine's indirect paths (witness reliability and self-surprise). Re-running each flagged record alone, with the engine unchanged, gives a difference of exactly 0.0 in all 15 cases tried, including the largest (batched −0.073, alone 0.0). The engine is therefore not defective; the batched design does not isolate the late attack that H5 is about.

**Correction.** H5 is evaluated on streams that attack one record each (studies `e1c_single` and `e1d_h5_single`), for every (lag, record) pair that is eligible under at least one committing policy, with the rule unchanged (every eligible pair within 1e-12; no-commit's mean at least 0.01). The batched numbers are kept and reported as a descriptive result: the combined effect of many simultaneous witness attacks on records in escrow. This entry is written before the single-record runs.

**Rule as implemented for the rerun.** The check is two-sided, |attack − clean| ≤ 1e-12 for every eligible pair, which is stricter than the one-sided reading of clarification 8 above. A pair is run only if it is eligible under at least one committing policy. The single-record studies run from a later commit than the other E1 studies; the engine files are identical in both commits, and each result records its commit.

## 2026-11-11: E7 clarifications, fixed in code before the first E7 run (commit of this entry)

The combined run (Section 2.9; `keeper/cppr/rating_backbone.py`, `keeper/cppr/witness_arm.py`) fixes these open choices:

1. **Witness levels per arm.** The credibility-weighted witness levels of an arm come from the arm's graph before release, the graph from which the gate's tiers come: causal capped tiers under the calibration frozen on the honest training graph.
2. **Nodes that a release adds.** A pool node takes its pool's tier at every snapshot, for the gate and for the witness weights; a single-pool node takes its pool's mean weight after the seed snapshots, as the gate weights it.
3. **Requirements of the rating-level backbone.** It runs with capped tiers at KEEPER's cap and guard and with the held test snapshot, and it refuses the E5 baselines, the ungated baseline and the base-model cache.
4. **Model.** The rating-level model is E2's SCEM model with a learned rate, two stages and the hazard on.
5. **Result files.** One file per network, seed and number of attack edges for the retention arms; the run without attack edges also retrains the single-pool method on the direct arm, because the method list applies to every arm.
6. **Witness arm.** The credibility gate stays on (capped tiers on the arm's own stream); only the witness weighting is switched. Clean twins are one per network, seed and weighting, shared by the runs with zero and one attack edge.
7. **Targets and gains.** The targets are the 15 edge-level direct targets and the victims are all their warnings. A positive gain means laundering (the target's reputation rises). Gains are reported over all targets and over the targets with at least one attacked warning, because the timing rule leaves some targets unattacked. The reported difference is weighting off minus weighting on.
8. **Endorsers.** Drawn from the clean stream's causal capped tiers.

**Compute.** E7 has 120 runs, about 28 GPU-hours on RTX A6000, more than the 18 planned; it runs in full, as registered.

## 2026-11-11: two items found while writing the supplement

**Reading of "contradicted" for removal thresholds (decided 2026-10-27, before any confirmatory run; logged now).** Section 1 calls a component contradicted when its interval lies entirely on the wrong side. For the removal thresholds (the "at least 0.8" parts of H1, H2w and H2b, and the "at least 0.5" part of H8), the analysis code marks a component as contradicted only when its whole interval lies below 0, that is, when the method is worse than nothing; an interval that lies below the threshold but above 0 is read as "partial" under the three-way reading of the same section. The decision of each hypothesis is unaffected, because a partial component already makes the hypothesis not supported.

**Analyses of Section 2.6 that were not run.** The no-training admission study in the whitewash topology with the common endorser pool, and the sensitivity to a calibration computed on the attacked graph over every E4 cell, have no implementation and were not run. Admission is reported from the trained runs (the per-group counts behind H3a) and from the star-size probe on pure fake groups (`results/analysis/tiers_report.json`). No hypothesis depends on the omitted analyses.
