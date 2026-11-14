"""Property 1 of upgrade M-B, per record: how much adversarial witness evidence can add to a revoked share.

design-M-B.md states Property 1 for a sender's hazard: N1 witnesses in Tier >= 1 and N0 in Tier 0 add at most
L_h c_max (N1 + w_0 N0) to it through the hazard map. property1_bound turns this into a bound on each record's
revoked share in the streaming engine, keeper.scem.scem.run_scem_streaming_weights_np, against the same stream
without the adversarial ratings. The bound covers the direct witness channel only (property1_bound's docstring
gives the exact scope). E1 reports it as a descriptive diagnostic (ruling R58), not as a run-by-run check.
keeper.scem.witness_attack re-exports property1_bound.
"""
import inspect
import math

import numpy as np

from keeper.scem import scem as SC
from keeper.scem.policies import release_policy

ROUTING = {1: (0.80, 0.20), -1: (0.20, 0.80)}   # the engine's channel routing psi of a positive / negative rating


def _engine_config(engine_kwargs: dict) -> dict:
    """The engine's parameters for a run: run_scem_streaming_weights_np's defaults, overridden by the run's."""
    params = inspect.signature(SC.run_scem_streaming_weights_np).parameters
    config = {name: p.default for name, p in params.items() if p.default is not inspect.Parameter.empty}
    unknown = sorted(set(engine_kwargs) - set(config))
    if unknown:
        raise TypeError(f"not parameters of the engine: {unknown}")
    config.update(engine_kwargs)
    return config


def _revoked_under(rel, t_k: int, hazard: np.ndarray, K_snap: int) -> float:
    """Revoked share of a record emitted at snapshot t_k whose only hazard is hazard[t] in snapshot t, by the
    engine's segment rules under the release policy rel (segment release, closure at A, provisional mass)."""
    exponent = released = 0.0
    for t in range(t_k, K_snap):
        x_a, x_b = float(t - t_k), min(float(t + 1 - t_k), rel.A)
        if x_b <= x_a:
            break
        released += float(rel.released(hazard[t], x_a, x_b, exponent))
        exponent += hazard[t] * (x_b - x_a)
        if t + 1 - t_k >= rel.A:
            return 1.0 - released - math.exp(-exponent) * float(rel.escrowed(rel.A))
    return 1.0 - released - math.exp(-exponent) * float(rel.escrowed(float(K_snap - t_k)))


def property1_bound(edges, sign, snap, witness_edge_ids, witness_kappa, *, K_snap: int,
                    **engine_kwargs) -> np.ndarray:
    """[E] bound on how much the witness evidence of the ratings witness_edge_ids (the Sybils' bad-mouthing)
    raises each record's revoked share in the engine, against the same stream without those ratings.

    witness_kappa holds the reliability each rating registered with: records["witness_kappa"] of the run with
    return_records. engine_kwargs are that run's options (policy, N, phi, H, L_h, gamma_sigma, cap_c, ...).

    Derivation, from run_scem_streaming_weights_np:
    1. Witness j registers kappa_hat_j at t_j. Its evidence matures at t_j + L_f, multiplied by its sender's
       lag survival (at most 1), and is live in the H snapshots t_j + L_f, ..., t_j + L_f + H - 1. Each
       (subject, contributor) pair is capped at cap_c. A Sybil is a contributor of its own, so it raises
       xi^{V1,-} of its subject W by at most min(kappa_hat_j, cap_c) while live, and changes nothing else:
       X_W(t) = sum over the live j of min(kappa_hat_j, cap_c).
    2. h = L_h max(xi^- - xi^+ + 0.35 xi^V2, 0) rises by at most L_h X_W(t). nu = softplus(h) - log 2 is
       1-Lipschitz, and lambda_sigma = lambda_max (1 - exp(-gamma_sigma nu / lambda_max)) has slope at most
       gamma_sigma. So the hazard of W's records in snapshot t rises by at most d_sigma(t) = gamma_sigma L_h X_W(t).
    3. A record's surviving mass is E[S(tau ^ x)], where tau is its release time under the policy (the engine
       integrates S over each segment, commits S(A) Q(A) at closure and keeps S Q provisional), and S is its
       survival. An added exponent D, with slope at most d_sigma, scales S by exp(-D), so the revoked share
       rises by E[S (1 - exp(-D(tau ^ x)))] <= E[1 - exp(-D_bar(tau ^ x))]. The right side is the revoked share
       of the same record with the hazard d_sigma alone, which _revoked_under computes with the policy's closed
       forms. Mass released before the evidence matures is immune.
    The bound is sum_sigma psi_sigma R_sigma, over the record's routing psi. Because 1 - exp(-D_bar(tau ^ x))
    <= 1 - exp(-D_bar(x)), it is at most 1 - exp(-gamma_max L_h sum_t sum_j min(kappa_hat_j, cap_c)), summed over
    the record's open snapshots. Records whose sender receives none of the evidence get 0.

    Scope. The bound covers the direct witness channel: the evidence the witness ratings add to the hazard of
    their subject W. It is guaranteed for every record not sent by a witness when, after the bad-mouthing:
    * W sends no rating. The witness ratings enter W's durable tallies, and kappa's confidence factor counts
      negative mass too, so they can raise W's own later witness weight; W's higher hazard also shrinks the lag
      survival of W's later evidence.
    * No account other than the witnesses rates W, because V2 reads W's consensus, which the witness ratings
      change.
    * No account that received a rating from W or from a witness sends or receives a rating. Its tallies, and with
      them its kappa and its consensus, depend on how much of that rating survives.
    * No witness sends or receives a rating, because a witness's own hazard and tallies change (next paragraph).
    inject_adversarial_witnesses meets the last clause, and the third for the witnesses' receivers: its Sybils rate
    only W and one another, and act last; W itself is covered by the first two clauses.
    The witnesses' own records are not covered. V2 self-surprise registers for a witness when W's durable consensus
    at the bad-mouthing has mass m = dur_pos + dur_neg of at least 2 and leans positive, (dur_pos + 1) / (m + 2) >
    1 - exp(-0.7) ~ 0.503, because the witness's negative rating then contradicts it. The witness's hazard rises,
    and its other records, such as a Sybil ring, are revoked more.
    Otherwise indirect paths can move other records, W's included, beyond the bound. The comparator must reuse the
    attacked stream's witness levels, because the BTP distrust guard counts the witness ratings against W.
    tests/test_scem_witness_attack.py pins six records outside the scope, each with bound 0:
    * the rating X sends after W's later warning about X (+2.13e-3); W's warning itself stays within its bound;
    * the rating Z sends after the later endorsement of Z by T, a receiver of W's warning (+1.50e-4);
    * a Sybil's ring record under V2 self-surprise (+4.01e-2);
    * the rating sent by an account that rated a Sybil after the bad-mouthing (+4.18e-4);
    * the rating Z sends, while Y, a receiver of the witnesses' earlier ratings, later endorses Z (+6.30e-4);
    * a rating sent by X, which rates W after the bad-mouthing (+2.92e-2; the second clause).
    A seeded randomized test over 300 streams that meet every clause finds no record not sent by a witness above
    its bound.
    """
    config = _engine_config(engine_kwargs)
    rel = release_policy(config["policy"], config["N"], config["phi"], config["q_quantile"], config["quarantine_D"])
    L_f, H, cap = int(config["L_f"]), int(config["H"]), float(config["cap_c"])
    slopes = float(config["L_h"]) * np.asarray(config["gamma_sigma"], dtype=np.float64)
    edges, sign, snap = np.asarray(edges).reshape(-1, 2), np.asarray(sign), np.asarray(snap)
    weights = np.asarray(witness_kappa, dtype=np.float64)
    evidence = {}                                        # subject -> [K_snap] added evidence X(t)
    for j in np.asarray(witness_edge_ids, dtype=np.int64).reshape(-1).tolist():
        start = int(snap[j]) + L_f
        evidence.setdefault(int(edges[j, 1]), np.zeros(K_snap))[start:start + H] += min(weights[j], cap)
    bound = np.zeros(len(edges), dtype=np.float64)
    for k in np.flatnonzero(np.isin(edges[:, 0], list(evidence))).tolist():
        added = evidence[int(edges[k, 0])]
        if added.any():
            psi = ROUTING[1 if sign[k] > 0 else -1]
            bound[k] = sum(p * _revoked_under(rel, int(snap[k]), s * added, K_snap) for p, s in zip(psi, slopes))
    return bound
