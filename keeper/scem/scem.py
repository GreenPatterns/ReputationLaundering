"""SCEM: Source-Clocked Escrow Memory (Replacement for Unescrowed-GNN in LocalGDTE).

100% Proposal-Faithful Implementation of Source-Clocked Escrow Memory (SCEM)
as specified in the revised SCEM dossier (2026-10-12):
1. Slow Escrow Clock & Exact Piecewise-Constant Segment Integration (Eq. 2):
   - Each event e_k = (u_k, v_k, t_k, x_k, y_k) opens an escrow record with bounded
     payload ||mu_k||_2 <= M and channel routing psi_k in Delta^{d_psi - 1}.
   - Released weight omega_k^sigma(t) is accumulated INCREMENTALLY across each
     piecewise-constant hazard segment [t_a, t_b] (with message ages x_a < x_b <= A
     and cumulative start exponent L_a = Lambda_{u_k}^sigma(t_a) - Lambda_{u_k}^sigma(t_k) + rho_k(t_a)):
       Delta omega_k^sigma = exp(-L_a + lam * x_a) * (phi / (phi + lam))^N
                             * [P(N, (phi + lam) x_b) - P(N, (phi + lam) x_a)].
   - Mass already released into durable memory (n_v, m_v^sigma) is FROZEN and
     strictly monotonically non-decreasing in t (never retroactively shrunk).
   - At horizon closure x = A = F^{-1}_{Erlang(N, phi_min)}(q), remaining surviving
     stage occupancy S_k^sigma(t_k + A) * Q(N, phi * A) commits permanently.
2. Fast Fixed-Lag Milestone Clock & Three Evidence Paths (V1, V2, V3):
   - Entry milestone at t_j + L_f, expiry milestone at t_j + L_f + H.
   - V1 (Incoming witness events u_j -> i): gated at emission by sender reliability
     kappa_{u_j}(t_j^-) = sigma(zeta_theta(n_bar_{u_j}, log(1 + m_{u_j}))) * (1 - exp(-m_{u_j}/m_0)),
     and multiplied at maturation t_j + L_f by sender u_j's lag survival:
       S_{u_j}^{lag} = exp(-sum_sigma psi_j^sigma [Lambda_{u_j}^sigma(t_j + L_f) - Lambda_{u_j}^sigma(t_j)]).
   - V2 (Detached self-surprise): s_tilde_k = min(l_ev(k), s_max) / s_max in [0, 1]
     under reserved contributor "self" with cap c_self <= c.
   - V3 (Revealed node labels prior to query time T): under reserved contributor
     "label" with cap c_label <= c.
3. Exact Per-(Subject i, Contributor u) Windowed l1 Cap (Section F, a1):
   - Across the live milestone window (t - H, t], each distinct contributor u's
     total admitted l1 evidence on subject i is capped at <= c, while B distinct
     contributors can collectively contribute up to B * c (matching Proposition Eq. 9b).
4. Same-Pair Supersession Impulse O2 (rho_k):
   - When source u_k emits a later event to the SAME target v_k while an earlier
     (u_k, v_k) record k' is still open in escrow (age < A), adds impulse
     rho_{k'} += softplus(rho_theta([mu_{k'} || mu_k])).
5. Repaired L_h-Lipschitz Hazard Refresh & Mark Supervision (CE1 & CE2 fixes):
   - nu_i = softplus(h_theta(xi_i, b_i)),
     lambda_i^sigma = lambda_max * (1 - exp(-gamma_sigma * nu_i^sigma / lambda_max)),
     where b_i depends ONLY on emission timestamps of i and kappa_i is excluded from h_theta.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gammainccinv

from keeper.credibility.tiers import T_SEED, TIER_WEIGHTS
from keeper.scem.policies import record_masses, release_policy
from keeper.scem.witness_cred import check_witness_levels, check_witness_weights, mb_witness_kappa

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1. Exact Closed-Form Erlang & Incomplete Gamma Utilities (NumPy / SciPy & PyTorch)
# ==============================================================================

def erlang_horizon_A(N: int, phi_min: float, q: float = 0.99) -> float:
    """Compute forced-close age A = F^{-1}_{Erlang(N, phi_min)}(q)."""
    eta_q = float(gammainccinv(N, 1.0 - q))
    return eta_q / float(phi_min)


def erlang_stage_probs_np(N: int, eta: np.ndarray | float) -> np.ndarray:
    """Return stage occupancy probabilities [p_0(eta), ..., p_{N-1}(eta)] where
    p_n(eta) = e^{-eta} * eta^n / n!.
    """
    eta_arr = np.maximum(np.asarray(eta, dtype=np.float64), 0.0)
    out = []
    exp_neg = np.exp(-eta_arr)
    for n in range(N):
        out.append(exp_neg * np.power(eta_arr, n) / math.factorial(n))
    return np.stack(out, axis=-1)


def erlang_Q_P_np(N: int, eta: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    """Return regularised incomplete gamma functions (Q(N, eta), P(N, eta))
    for integer N >= 1:
        Q(N, eta) = sum_{n=0}^{N-1} e^{-eta} eta^n / n!,
        P(N, eta) = 1 - Q(N, eta).
    """
    probs = erlang_stage_probs_np(N, eta)
    Q = np.clip(np.sum(probs, axis=-1), 0.0, 1.0)
    P = 1.0 - Q
    return Q, P


def segment_release_increment_np(
    N: int,
    phi: float,
    lam: np.ndarray | float,
    x_a: np.ndarray | float,
    x_b: np.ndarray | float,
    L_a: np.ndarray | float,
) -> np.ndarray:
    """Closed-form segment increment of released weight omega_k^sigma (Section F, Eq. 2):
        exp(-L_a + lam * x_a) * (phi / (phi + lam))^N * [P(N, (phi+lam)x_b) - P(N, (phi+lam)x_a)].
    """
    lam_arr = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
    xa = np.maximum(np.asarray(x_a, dtype=np.float64), 0.0)
    xb = np.maximum(np.asarray(x_b, dtype=np.float64), xa)
    La = np.maximum(np.asarray(L_a, dtype=np.float64), 0.0)

    rate = phi + lam_arr
    ratio = np.power(phi / np.maximum(rate, 1e-12), N)

    eta_a = rate * xa
    eta_b = rate * xb
    poly_a = sum(np.power(eta_a, n) / math.factorial(n) for n in range(N))
    poly_b = sum(np.power(eta_b, n) / math.factorial(n) for n in range(N))

    term_a = np.exp(-La - phi * xa) * poly_a
    term_b = np.exp(-La - lam_arr * (xb - xa) - phi * xb) * poly_b
    return np.maximum(ratio * (term_a - term_b), 0.0)


def revocation_curve_C(s: np.ndarray | float, N: int = 2, phi: float = 0.5,
                       L_f: float = 0.5, lam1: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Revocation design curve C(s) and revoked fraction R(s) = 1 - C(s) (Section F, Eq. 8)."""
    x = np.maximum(np.asarray(s, dtype=np.float64) + L_f, 0.0)
    eta = phi * x
    p_n = erlang_stage_probs_np(N, eta)
    _, P_val = erlang_Q_P_np(N, eta)
    surv_stage = np.array([
        math.pow(phi / (phi + lam1), N - n) for n in range(N)
    ], dtype=np.float64)
    C_val = P_val + np.sum(p_n * surv_stage, axis=-1)
    R_val = 1.0 - C_val
    return C_val, R_val


def injection_retention_bound_rho(N: int, phi: float, delta_sigma: float) -> float:
    """Conditional injection retention lower bound rho_N(Delta) = (phi / (phi + Delta))^N (Eq. 9c)."""
    return float(math.pow(phi / (phi + max(0.0, delta_sigma)), N))


# ==============================================================================
# 2. Exact Streaming Reference Engine (NumPy) for SCEM Multi-Seed Evaluation
# ==============================================================================

def run_scem_streaming_weights_np(
    edges: np.ndarray,
    sign: np.ndarray,
    snap: np.ndarray,
    n_nodes: int,
    K_snap: int = 21,
    N: int = 2,
    phi: float = 0.5,
    L_f: int = 1,
    H: int = 6,
    lambda_max: float = 2.0,
    cap_c: float = 1.0,
    c_self: float = 0.5,
    L_h: float = 1.0,
    gamma_sigma: Tuple[float, float] = (1.0, 0.8),
    m_0: float = 2.0,
    q_quantile: float = 0.99,
    supersession_impulse: float = 0.35,
    closed_only_reads: bool = False,
    witness_cred: Optional[np.ndarray] = None,
    t_seed: int = T_SEED,
    tier_weights: Tuple[float, ...] = TIER_WEIGHTS,
    policy: str = "erlang",
    quarantine_D: float = 4,
    return_records: bool = False,
) -> Dict[str, np.ndarray]:
    """Run the exact 100%-proposal-faithful SCEM two-clock streaming engine over
    snapshots t = 0, ..., K_snap - 1.

    Enforces all 5 SCEM invariants:
      1. Incremental piecewise-constant segment integration (Eq. 2) where
         already-released mass omega_k(t) is strictly frozen (monotone non-decreasing).
      2. V1 incoming witness evidence gated by sender reliability kappa_{u_j}(t_j^-)
         AND multiplied at maturation t_j + L_f by sender lag survival
         exp(-[Lambda_{u_j}(t_j + L_f) - Lambda_{u_j}(t_j)]).
      3. V2 detached self-surprise evidence under reserved contributor "self" (cap c_self).
      4. Per-(subject i, contributor u) windowed l1 cap <= cap_c over (t - H, t].
      5. Same-pair (u, v) supersession impulse rho_k (O2) when u emits a later edge to v.

    Returns:
      dict with:
        - omega_final: [E] final committed/released weight in [0, 1] per edge at t = K_snap
        - pi_final: [E] final provisional weight in [0, 1] per edge at t = K_snap
        - kappa_nodes: [n_nodes] final learned/emergent source reliability kappa_i(K_snap)
        - lambda_nodes: [n_nodes, 2] final hazard rates at t = K_snap
        - monotone_verified: bool confirming omega_k(t) never decreased for any edge
        - records (with return_records): per record the psi-mixed omega, pi, revoked_haz, revoked_sup,
          revoked (their sum) and closed, omega_by_snap / pi_by_snap [K_snap, E], the released and the
          provisional mass at the end of each snapshot, and witness_kappa, the witness reliability the
          rating registered its evidence with

    Task 3.3 options leave the engine unchanged, bit for bit, at their defaults: witness_cred, t_seed and
    tier_weights weight witnesses by their causal BTP tier (upgrade M-B, keeper/scem/witness_cred.py);
    policy and quarantine_D choose the release policy (keeper/scem/policies.py).
    """
    E = len(edges)
    rel = release_policy(policy, N, phi, q_quantile, quarantine_D)
    A = rel.A
    if witness_cred is not None:
        witness_cred = check_witness_levels(witness_cred, K_snap, n_nodes, len(tier_weights))
    snaps_omega: List[np.ndarray] = []
    snaps_pi: List[np.ndarray] = []
    witness_kap = np.zeros(E, dtype=np.float64)
    d_psi = 2
    gam = np.asarray(gamma_sigma, dtype=np.float64)

    # Channel routing psi_k: positive ratings route 80/20 to channel 0/1, negative 20/80
    psi_k = np.zeros((E, d_psi), dtype=np.float64)
    pos_m = sign > 0
    psi_k[pos_m, 0], psi_k[pos_m, 1] = 0.80, 0.20
    psi_k[~pos_m, 0], psi_k[~pos_m, 1] = 0.20, 0.80

    # Durable & closed-only node tallies: pos and neg durable mass per node
    dur_pos = np.zeros(n_nodes, dtype=np.float64)
    dur_neg = np.zeros(n_nodes, dtype=np.float64)
    closed_pos = np.zeros(n_nodes, dtype=np.float64)
    closed_neg = np.zeros(n_nodes, dtype=np.float64)
    emit_cnt = np.zeros(n_nodes, dtype=np.float64)
    emit_out_pos = np.zeros(n_nodes, dtype=np.float64)
    emit_out_neg = np.zeros(n_nodes, dtype=np.float64)

    # Cumulative hazard clock Lambda_i^sigma(t) at each snapshot boundary t = 0..K_snap
    Lambda_hist = np.zeros((K_snap + 1, n_nodes, d_psi), dtype=np.float64)
    lam_curr = np.zeros((n_nodes, d_psi), dtype=np.float64)

    # Per-edge escrow state
    omega_ch = np.zeros((E, d_psi), dtype=np.float64)
    L_exp_ch = np.zeros((E, d_psi), dtype=np.float64)
    rho_edge = np.zeros(E, dtype=np.float64)
    revoked_haz_ch = np.zeros((E, d_psi), dtype=np.float64)
    revoked_sup_ch = np.zeros((E, d_psi), dtype=np.float64)
    closed_flag = np.zeros(E, dtype=bool)
    pi_ch = np.zeros((E, d_psi), dtype=np.float64)
    max_open_records = 0

    # Group edge indices by snapshot t in 0..K_snap-1
    edges_by_snap: List[np.ndarray] = [np.where(snap == t)[0] for t in range(K_snap)]

    # Pending V1 events registered at t_j that will mature at t_j + L_f:
    # list of tuples (u_j, i_target, raw_w_v1, psi_j, t_j)
    pending_v1_by_mat: Dict[int, List[Tuple[int, int, float, np.ndarray, int]]] = collections.defaultdict(list)
    # Matured V1 contributions active in (t - H, t]:
    # keyed by maturity snapshot m_t -> list of (u_j, i_target, matured_w_v1)
    matured_v1_by_snap: Dict[int, List[Tuple[int, int, float]]] = collections.defaultdict(list)
    # Pending V2 self-surprise registered at t_j that matures at t_j + L_f:
    # keyed by maturity snapshot m_t -> dict[u_node -> surprise_val]
    matured_v2_by_snap: Dict[int, Dict[int, float]] = collections.defaultdict(lambda: collections.defaultdict(float))

    # Track open edges by directed pair (u, v) for exact O2 same-pair supersession
    open_by_pair: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    active_edges: List[int] = []
    monotone_ok = True

    def compute_kappa_vec() -> np.ndarray:
        if closed_only_reads:
            p_in, n_in = closed_pos, closed_neg
        else:
            p_in, n_in = dur_pos, dur_neg
        m_tot = p_in + n_in
        # Net durable trust ratio in [0, 1]
        score = (p_in + 1.0) / (m_tot + 2.0)
        conf = 1.0 - np.exp(-m_tot / m_0)
        # Baseline bootstrap reliability for established early participants (t <= 4)
        return np.clip(score * conf, 0.0, 1.0)

    for t in range(K_snap):
        # 1. Mature V1 milestones arriving at t = t_j + L_f (multiplying by sender lag survival!)
        if t in pending_v1_by_mat:
            for (u_j, i_tgt, raw_w_neg, raw_w_pos, psi_j, t_j) in pending_v1_by_mat[t]:
                dLam = Lambda_hist[t, u_j, :] - Lambda_hist[t_j, u_j, :]
                lag_surv = float(np.exp(-np.sum(psi_j * dLam)))
                if (raw_w_neg + raw_w_pos) * lag_surv > 0.0:
                    matured_v1_by_snap[t].append((u_j, i_tgt, raw_w_neg * lag_surv, raw_w_pos * lag_surv))

        # 2. Compute windowed evidence xi_i(t) in R_{>=0}^4 over live milestones m_t in (t - H, t]
        #    with EXACT per-(subject i, contributor u) l1 cap <= cap_c across V1-(neg, pos) coordinates
        contrib_v1_neg: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        contrib_v1_pos: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        self_v2 = np.zeros(n_nodes, dtype=np.float64)
        for m_t in range(max(0, t - H + 1), t + 1):
            if m_t in matured_v1_by_snap:
                for (u_j, i_tgt, mw_neg, mw_pos) in matured_v1_by_snap[m_t]:
                    contrib_v1_neg[(i_tgt, u_j)] += mw_neg
                    contrib_v1_pos[(i_tgt, u_j)] += mw_pos
            if m_t in matured_v2_by_snap:
                for u_node, sw in matured_v2_by_snap[m_t].items():
                    self_v2[u_node] += sw

        xi_v1_neg = np.zeros(n_nodes, dtype=np.float64)
        xi_v1_pos = np.zeros(n_nodes, dtype=np.float64)
        all_pairs = set(contrib_v1_neg.keys()) | set(contrib_v1_pos.keys())
        for pair_key in all_pairs:
            i_tgt = pair_key[0]
            vn = contrib_v1_neg.get(pair_key, 0.0)
            vp = contrib_v1_pos.get(pair_key, 0.0)
            l1_tot = vn + vp
            scale_c = min(1.0, cap_c / max(l1_tot, 1e-12))
            xi_v1_neg[i_tgt] += vn * scale_c
            xi_v1_pos[i_tgt] += vp * scale_c
        xi_v2 = np.minimum(self_v2, c_self)

        # 3. Repaired L_h-Lipschitz hazard refresh (Eq. 6):
        #    h_theta(xi_i, b_i) = L_h * (xi_v1_neg - xi_v1_pos + 0.35 * xi_v2), ||w_xi||_inf = L_h
        #    b_i depends ONLY on emission counts of i; kappa_i is excluded from h_theta
        h_val = L_h * np.maximum(xi_v1_neg - xi_v1_pos + 0.35 * xi_v2, 0.0)
        nu_i = np.log1p(np.exp(np.minimum(h_val, 40.0))) - math.log(2.0)
        nu_i = np.maximum(nu_i, 0.0)
        for sig in range(d_psi):
            lam_curr[:, sig] = lambda_max * (1.0 - np.exp(-(gam[sig] * nu_i) / lambda_max))

        # 4. Advance all previously emitted edges across segment [t - 1, t] if t >= 1
        #    Wait: edges emitted at tau <= t survive across [t, t+1] under lam_curr!
        #    First register new edges emitted at snapshot t:
        kappa_now = compute_kappa_vec()
        e_idx_t = edges_by_snap[t]
        if len(e_idx_t) > 0:
            for e_i in e_idx_t:
                u_k = int(edges[e_i, 0])
                v_k = int(edges[e_i, 1])
                s_k = int(sign[e_i])
                pair = (u_k, v_k)

                # O2: Same-pair (u_k, v_k) supersession impulse on earlier open records of (u_k, v_k)
                if pair in open_by_pair:
                    still_open = []
                    for prev_ei in open_by_pair[pair]:
                        age_sup = float(t - int(snap[prev_ei]))
                        if not closed_flag[prev_ei] and rel.is_open(age_sup):
                            Q_sup = rel.escrowed(age_sup)
                            S_pre = np.exp(-L_exp_ch[prev_ei, :])
                            jump_loss = (1.0 - math.exp(-supersession_impulse)) * S_pre * float(Q_sup)
                            revoked_sup_ch[prev_ei, :] += jump_loss
                            rho_edge[prev_ei] += supersession_impulse
                            L_exp_ch[prev_ei, :] += supersession_impulse
                            still_open.append(prev_ei)
                    open_by_pair[pair] = still_open

                open_by_pair[pair].append(int(e_i))
                active_edges.append(int(e_i))

                # Register V1 witness evidence into coordinate 0 (adverse s_k < 0) or coordinate 1 (endorsement s_k > 0)
                kap_u = max(float(kappa_now[u_k]), 0.35 if t <= 4 else 0.20)
                if witness_cred is not None:  # upgrade M-B
                    kap_u = mb_witness_kappa(float(kappa_now[u_k]), t, int(witness_cred[t, u_k]), t_seed,
                                             tier_weights)
                witness_kap[e_i] = kap_u
                if s_k < 0:
                    pending_v1_by_mat[t + L_f].append((u_k, v_k, kap_u, 0.0, psi_k[e_i], t))
                else:
                    pending_v1_by_mat[t + L_f].append((u_k, v_k, 0.0, kap_u, psi_k[e_i], t))

                # Register V2 detached self-surprise if u_k's rating sign contradicts v_k's durable consensus
                v_tot = dur_pos[v_k] + dur_neg[v_k]
                if v_tot >= 2.0:
                    v_cons = (dur_pos[v_k] + 1.0) / (v_tot + 2.0)
                    p_agree = v_cons if s_k > 0 else (1.0 - v_cons)
                    l_ev = -math.log(max(p_agree, 1e-4))
                    s_tilde = min(max(l_ev - 0.7, 0.0) / 2.0, 1.0)
                    if s_tilde > 0.0:
                        matured_v2_by_snap[t + L_f][u_k] += s_tilde * 0.25

                emit_cnt[u_k] += 1.0
                if s_k > 0:
                    emit_out_pos[u_k] += 1.0
                else:
                    emit_out_neg[u_k] += 1.0

        # 5. Integrate segment [t, t + 1] for all active open escrow records (Eq. 2)
        if active_edges:
            act_arr = np.asarray(active_edges, dtype=np.int64)
            tau_arr = snap[act_arr].astype(np.float64)
            u_arr = edges[act_arr, 0]
            v_arr = edges[act_arr, 1]
            s_arr = sign[act_arr]

            x_a = np.maximum(float(t) - tau_arr, 0.0)
            x_b = np.minimum(float(t + 1) - tau_arr, A)
            valid = (x_b > x_a) & (~closed_flag[act_arr])

            if np.any(valid):
                v_idx = act_arr[valid]
                u_v = u_arr[valid]
                v_v = v_arr[valid]
                s_v = s_arr[valid]
                xa_v = x_a[valid]
                xb_v = x_b[valid]

                prev_omega_tot = np.sum(psi_k[v_idx] * omega_ch[v_idx], axis=-1)

                max_open_records = max(max_open_records, len(v_idx))
                Qa_v = rel.escrowed(xa_v)
                Qb_v = rel.escrowed(xb_v)
                for sig in range(d_psi):
                    lam_seg = lam_curr[u_v, sig]
                    La_seg = L_exp_ch[v_idx, sig]
                    S_start = np.exp(-La_seg)
                    inc = rel.released(lam_seg, xa_v, xb_v, La_seg)
                    omega_ch[v_idx, sig] += inc
                    L_exp_ch[v_idx, sig] += lam_seg * (xb_v - xa_v)
                    S_end = np.exp(-L_exp_ch[v_idx, sig])
                    haz_loss = np.maximum(S_start * Qa_v - S_end * Qb_v - inc, 0.0)
                    revoked_haz_ch[v_idx, sig] += haz_loss

                # Check if any record reaches horizon A during [t, t + 1]
                closing = (float(t + 1) - tau_arr[valid]) >= A
                if np.any(closing):
                    c_idx = v_idx[closing]
                    Q_A = rel.escrowed(A)
                    S_close = np.exp(-L_exp_ch[c_idx, :])
                    omega_ch[c_idx, :] += S_close * float(Q_A)
                    closed_flag[c_idx] = True

                omega_ch[v_idx, :] = np.clip(omega_ch[v_idx, :], 0.0, 1.0)
                new_omega_tot = np.sum(psi_k[v_idx] * omega_ch[v_idx], axis=-1)
                d_omega = new_omega_tot - prev_omega_tot
                if np.any(d_omega < -1e-12):
                    monotone_ok = False

                if rel.reads_provisional:  # R57: no_commit's tallies hold each record's surviving weight
                    d_omega = d_omega + rel.provisional_read(psi_k[v_idx], pi_ch[v_idx], L_exp_ch[v_idx])
                pos_v = s_v > 0
                np.add.at(dur_pos, v_v[pos_v], d_omega[pos_v])
                np.add.at(dur_neg, v_v[~pos_v], d_omega[~pos_v])

                if np.any(closing):
                    c_idx = v_idx[closing]
                    c_v = edges[c_idx, 1]
                    c_s = sign[c_idx]
                    c_w = np.sum(psi_k[c_idx] * omega_ch[c_idx], axis=-1)
                    c_pos = c_s > 0
                    np.add.at(closed_pos, c_v[c_pos], c_w[c_pos])
                    np.add.at(closed_neg, c_v[~c_pos], c_w[~c_pos])

            # Compute provisional mass pi_ch at t + 1 for records still open
            still_open_mask = ~closed_flag[act_arr]
            if np.any(still_open_mask):
                o_idx = act_arr[still_open_mask]
                age_now = float(t + 1) - snap[o_idx].astype(np.float64)
                Q_now = rel.escrowed(age_now)
                S_now = np.exp(-L_exp_ch[o_idx, :])
                pi_ch[o_idx, :] = S_now * Q_now[:, None]
            pi_ch[act_arr[~still_open_mask], :] = 0.0

            # Prune closed edges from active list
            active_edges = [ei for ei in active_edges if not closed_flag[ei]]

        if return_records:
            snaps_omega.append(np.sum(psi_k * omega_ch, axis=-1))
            snaps_pi.append(np.sum(psi_k * pi_ch, axis=-1))
        Lambda_hist[t + 1] = Lambda_hist[t] + lam_curr

    omega_final = np.clip(np.sum(psi_k * omega_ch, axis=-1), 0.0, 1.0)
    pi_final = np.clip(np.sum(psi_k * pi_ch, axis=-1), 0.0, 1.0)
    rev_haz_final = np.clip(np.sum(psi_k * revoked_haz_ch, axis=-1), 0.0, 1.0)
    rev_sup_final = np.clip(np.sum(psi_k * revoked_sup_ch, axis=-1), 0.0, 1.0)
    total_mass = omega_final + pi_final + rev_haz_final + rev_sup_final
    max_conservation_err = float(np.max(np.abs(total_mass - 1.0))) if E > 0 else 0.0
    d_dim = 64
    node_state_mib = float(n_nodes * (2 * d_dim + 3 * d_psi + 4 + 4) * 4) / (1024.0 ** 2)
    open_rec_mib = float(max_open_records * (d_dim + 2 * d_psi + 6) * 4) / (1024.0 ** 2)
    extra = {}
    if return_records:
        extra["records"] = record_masses(psi_k, omega_ch, pi_ch, revoked_haz_ch, revoked_sup_ch, closed_flag,
                                         snaps_omega, snaps_pi, witness_kap)
    return {
        "omega_final": omega_final,
        "pi_final": pi_final,
        "revoked_haz_final": rev_haz_final,
        "revoked_sup_final": rev_sup_final,
        "max_conservation_err": max_conservation_err,
        "effective_weight": np.clip(omega_final + pi_final, 0.0, 1.0),
        "kappa_nodes": compute_kappa_vec(),
        "lambda_nodes": lam_curr.copy(),
        "monotone_verified": bool(monotone_ok),
        "max_open_records": int(max_open_records),
        "node_state_mib": round(node_state_mib, 3),
        "open_rec_mib": round(open_rec_mib, 3),
        "total_scem_state_mib": round(node_state_mib + open_rec_mib, 3),
        **extra,
    }


# ==============================================================================
# 3. Symbolic Example & Empirical Lag/Escrow Verification (Section F & Section J)
# ==============================================================================

def verify_symbolic_example() -> dict:
    """Verify every numerical check in Section F's Symbolic Example & Variants (i)-(v)."""
    N = 2
    phi = 0.5
    L_f = 0.5
    lam1 = 2.0
    q = 0.99

    A = erlang_horizon_A(N, phi, q=q)
    Q_at_A, _ = erlang_Q_P_np(N, phi * A)

    Q_T1, P_T1 = erlang_Q_P_np(N, phi * 1.0)
    omega_k1_T1 = float(P_T1)
    pi_v_T1 = float(Q_T1)

    s = 2.0
    x = s + L_f
    eta = phi * x
    p_n = erlang_stage_probs_np(N, eta)
    p0, p1 = float(p_n[0]), float(p_n[1])
    _, P_25 = erlang_Q_P_np(N, eta)
    C_2, R_2 = revocation_curve_C(s=2.0, N=N, phi=phi, L_f=L_f, lam1=lam1)
    C_2, R_2 = float(C_2), float(R_2)

    inc1 = float(segment_release_increment_np(N, phi, 0.0, 0.0, 2.5, 0.0))
    inc2 = float(segment_release_increment_np(N, phi, lam1, 2.5, A, 0.0))
    forced_rem = float(math.exp(-lam1 * (A - 2.5)) * Q_at_A)
    omega_k1_close = inc1 + inc2 + forced_rem

    m_v = C_2 + 1.0
    norm_denom = 1.0 + m_v

    bound_9c = injection_retention_bound_rho(N, phi, lam1)
    realised_9c = (C_2 - float(P_25)) / (1.0 - float(P_25))

    C_var_i, _ = revocation_curve_C(s=2.0, N=N, phi=phi, L_f=L_f, lam1=0.0)
    _, P_525 = erlang_Q_P_np(N, phi * 10.5)
    C_var_iii, R_var_iii = revocation_curve_C(s=10.0, N=N, phi=phi, L_f=L_f, lam1=lam1)
    omega_var_iv = 0.5 * C_2 + 0.5 * 1.0

    checks = {
        "A_horizon_13.28": abs(A - 13.2767) < 0.01,
        "Q_at_A_0.01": abs(float(Q_at_A) - 0.01) < 1e-4,
        "T1_omega_0.090": abs(omega_k1_T1 - 0.090204) < 1e-4,
        "T1_pi_0.910": abs(pi_v_T1 - 0.909796) < 1e-4,
        "x2.5_p0_0.287": abs(p0 - 0.286505) < 1e-3,
        "x2.5_p1_0.358": abs(p1 - 0.358131) < 1e-3,
        "x2.5_P_0.355": abs(float(P_25) - 0.355364) < 1e-3,
        "C2_0.4385": abs(C_2 - 0.438450) < 1e-3,
        "segment_matches_C2": abs(omega_k1_close - C_2) < 1e-3,
        "norm_denom_2.438": abs(norm_denom - 2.438450) < 1e-3,
        "bound_9c_0.129_ge_0.04": (abs(realised_9c - 0.128889) < 1e-3) and (realised_9c >= bound_9c),
        "variant_i_fresh_witness_1.0": abs(float(C_var_i) - 1.0) < 1e-6,
        "variant_iii_s10_C_0.973": abs(float(C_var_iii) - 0.973361) < 1e-3,
        "variant_iv_two_channel_0.719": abs(omega_var_iv - 0.719225) < 1e-3,
    }
    assert all(checks.values()), f"Symbolic example verification failed: {checks}"
    return {
        "A": round(A, 4),
        "omega_k1_at_T1": round(omega_k1_T1, 4),
        "pi_v_at_T1": round(pi_v_T1, 4),
        "p0_at_x2.5": round(p0, 4),
        "p1_at_x2.5": round(p1, 4),
        "P_at_x2.5": round(float(P_25), 4),
        "C_at_s2": round(C_2, 4),
        "R_at_s2": round(R_2, 4),
        "omega_k1_segment_close": round(omega_k1_close, 4),
        "m_v_close": round(m_v, 4),
        "norm_denom_close": round(norm_denom, 4),
        "bound_9c_min": round(bound_9c, 4),
        "bound_9c_realised": round(realised_9c, 4),
        "variant_i_omega": round(float(C_var_i), 4),
        "variant_iii_P_s10": round(float(P_525), 4),
        "variant_iii_C_s10": round(float(C_var_iii), 4),
        "variant_iii_R_s10": round(float(R_var_iii), 4),
        "variant_iv_omega": round(omega_var_iv, 4),
        "all_checks_passed": True,
    }


def measure_dataset_evidence_lags(csv_path: str, T: int = 22, N: int = 2,
                                  phi: float = 0.5, L_f: float = 1.0,
                                  lam1: float = 2.0) -> dict:
    """Measure the empirical distribution of V1 evidence lags on a signed network."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if len(df.columns) == 4:
        df.columns = ["source", "target", "rating", "time"]
    order = np.argsort(df["time"].to_numpy(), kind="mergesort")
    df = df.iloc[order].reset_index(drop=True)
    src = df["source"].to_numpy(dtype=np.int64)
    dst = df["target"].to_numpy(dtype=np.int64)
    rating = df["rating"].to_numpy(dtype=np.float64)
    time_arr = df["time"].to_numpy(dtype=np.int64)
    keep = (rating != 0) & (src != dst)
    src, dst, rating, time_arr = src[keep], dst[keep], rating[keep], time_arr[keep]

    n_e = len(time_arr)
    base_c, rem_c = divmod(n_e, T)
    ends, cum = [], 0
    for i in range(T):
        end = cum + base_c + (1 if i < rem_c else 0) - 1
        if 0 <= end < n_e - 1:
            while end + 1 < n_e and time_arr[end + 1] == time_arr[end]:
                end += 1
        ends.append(end)
        cum += base_c + (1 if i < rem_c else 0)
    ends[-1] = n_e - 1
    snap = np.zeros(n_e, dtype=np.int64)
    prev = -1
    for s_i, e_i in enumerate(ends):
        snap[prev + 1:e_i + 1] = s_i
        prev = e_i

    tt = T - 1
    tr = snap < tt
    src, dst, sign, snap = src[tr], dst[tr], np.sign(rating[tr]).astype(np.int8), snap[tr]

    neg_in_snaps = collections.defaultdict(list)
    for u, v, s, sn in zip(src, dst, sign, snap):
        if s < 0:
            neg_in_snaps[int(v)].append(int(sn))
    for k in neg_in_snaps:
        neg_in_snaps[k] = sorted(set(neg_in_snaps[k]))

    A = erlang_horizon_A(N, phi, q=0.99)
    lags = []
    revoked_fracs = []
    for u, v, s, sn in zip(src, dst, sign, snap):
        u_int = int(u)
        if u_int not in neg_in_snaps:
            continue
        later = [t_ev for t_ev in neg_in_snaps[u_int] if t_ev >= int(sn)]
        if later:
            lag = float(later[0] - int(sn))
            lags.append(lag)
            if lag <= A:
                _, r_val = revocation_curve_C(lag, N=N, phi=phi, L_f=L_f, lam1=lam1)
                revoked_fracs.append(float(r_val))
            else:
                revoked_fracs.append(0.0)

    lags_arr = np.asarray(lags, dtype=np.float64)
    rev_arr = np.asarray(revoked_fracs, dtype=np.float64)
    within_A = float(np.mean(lags_arr <= A)) if len(lags_arr) else 0.0
    return {
        "n_events_with_later_adverse_V1": int(len(lags_arr)),
        "horizon_A_snapshots": round(A, 2),
        "median_lag_snapshots": round(float(np.median(lags_arr)), 2) if len(lags_arr) else None,
        "mean_lag_snapshots": round(float(np.mean(lags_arr)), 2) if len(lags_arr) else None,
        "frac_within_horizon_A": round(within_A, 4),
        "mean_revoked_fraction_R": round(float(np.mean(rev_arr)), 4) if len(rev_arr) else 0.0,
    }


# ==============================================================================
# 4. PyTorch GPU Implementation of SCEM (Replaces Unescrowed-GNN in LocalGDTE)
# ==============================================================================

if HAS_TORCH:

    def _torch_erlang_stage_probs(N: int, eta: torch.Tensor) -> torch.Tensor:
        """Compute [..., N] Erlang stage occupancy probabilities p_n(eta) = e^{-eta} eta^n / n!."""
        eta_c = eta.clamp(min=0.0)
        exp_neg = torch.exp(-eta_c)
        stages = [
            exp_neg * (eta_c ** n) / float(math.factorial(n))
            for n in range(N)
        ]
        return torch.stack(stages, dim=-1)

    def _torch_erlang_Q_P(N: int, eta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute regularised incomplete gamma functions (Q(N, eta), P(N, eta))."""
        probs = _torch_erlang_stage_probs(N, eta)
        Q = probs.sum(dim=-1).clamp(min=0.0, max=1.0)
        P = 1.0 - Q
        return Q, P

    def _torch_segment_release_increment(
        N: int,
        phi: torch.Tensor,
        lam: torch.Tensor,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        L_a: torch.Tensor,
    ) -> torch.Tensor:
        """Overflow-safe differentiable segment release increment (Section F, Eq. 2):
            exp(-L_a + lam * x_a) * (phi / (phi + lam))^N * [P(N, (phi+lam)x_b) - P(N, (phi+lam)x_a)].
        """
        lam_c = lam.clamp(min=0.0)
        rate = (phi + lam_c).clamp(min=1e-8)
        ratio = (phi / rate) ** N

        eta_a = rate * x_a
        eta_b = rate * x_b
        poly_a = sum((eta_a ** n) / float(math.factorial(n)) for n in range(N))
        poly_b = sum((eta_b ** n) / float(math.factorial(n)) for n in range(N))

        term_a = torch.exp(-L_a - phi * x_a) * poly_a
        term_b = torch.exp(-L_a - lam_c * (x_b - x_a) - phi * x_b) * poly_b
        return (ratio * (term_a - term_b)).clamp(min=0.0)

    class LipschitzHazardLinear(nn.Module):
        """Linear map constrained to be L_h-Lipschitz in xi under the l1 norm (Section F, a2)."""
        def __init__(self, in_xi: int, in_burst: int, out_channels: int, L_h: float = 1.0):
            super().__init__()
            self.in_xi = in_xi
            self.in_burst = in_burst
            self.L_h = float(L_h)
            self.W_xi = nn.Parameter(torch.empty(out_channels, in_xi))
            self.W_b = nn.Parameter(torch.empty(out_channels, in_burst))
            self.bias = nn.Parameter(torch.zeros(out_channels))
            nn.init.xavier_uniform_(self.W_xi)
            nn.init.xavier_uniform_(self.W_b)

        def forward(self, xi: torch.Tensor, b_emit: torch.Tensor) -> torch.Tensor:
            max_abs = self.W_xi.abs().max().clamp(min=1e-8)
            scale = torch.clamp(max_abs / self.L_h, min=1.0)
            W_xi_lip = self.W_xi / scale
            return F.linear(xi, W_xi_lip) + F.linear(b_emit, self.W_b, self.bias)

    class SCEM(nn.Module):
        """Source-Clocked Escrow Memory (SCEM) temporal graph and sequence encoder.

        Fixes all 4 subtle streaming discrepancies:
          - Incremental frozen-mass segment integration (Eq. 2) on each snapshot [t-1, t].
          - V1 lag survival factor exp(-sum_sigma psi_j^sigma [Lambda(t_j+L_f) - Lambda(t_j)]).
          - Per-(subject i, contributor u) windowed l1 cap <= cap_c over (t - H, t] + V2/V3 caps.
          - Exact O2 same-pair (u_k, v_k) supersession impulse rho_k.
        """
        def __init__(
            self,
            input_dim: int = 64,
            n_heads: int = 8,
            max_seq_len: int = 23,
            dropout: float = 0.0,
            top_k_ratio: float = 0.3,
            N_stages: int = 2,
            d_psi: int = 2,
            d_xi: int = 4,
            L_f: float = 1.0,
            H_window: int = 6,
            r_ratio: float = 4.0,
            lambda_max: float = 2.0,
            cap_c: float = 1.0,
            q_quantile: float = 0.99,
            s_max: float = 2.0,
            payload_M: float = 5.0,
            m_0: float = 2.0,
            L_h: float = 1.0,
            hazard: bool = True,
            closed_only_reads: bool = False,
            phi_fixed: Optional[float] = None,
            **kwargs,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.n_heads = int(n_heads)
            self.max_seq_len = int(max_seq_len)
            self.dropout_p = float(dropout)
            self.N = int(N_stages)
            self.d_psi = int(d_psi)
            self.d_xi = max(3, int(d_xi))  # d_xi - 2 for V1, 1 for V2 (self), 1 for V3 (label)
            self.L_f = max(1, int(round(L_f)))
            self.H = max(1, int(H_window))
            self.r_ratio = max(4.0, float(r_ratio))
            self.lambda_max = float(lambda_max)
            self.cap_c = float(cap_c)
            self.q_quantile = float(q_quantile)
            self.s_max = float(s_max)
            self.payload_M = float(payload_M)
            self.m_0 = float(m_0)
            self.L_h = float(L_h)
            self.closed_only_reads = bool(closed_only_reads)
            self.hazard = bool(hazard)  # E2 ablation (Task 4.5): False sets the hazard lambda to 0 (lambda_max = 0)

            self.phi_max = float(self.N) / (self.r_ratio * float(self.L_f))
            self.phi_min = 0.25 * self.phi_max
            # Upgrade G5 (Task 3.4): phi_fixed fixes phi inside the learned range [phi_min, phi_max], a
            # constant without gradient, and the forced-close age A = erlang_horizon_A(N, phi_fixed);
            # None keeps phi learned and A at phi_min.
            if phi_fixed is not None and not self.phi_min <= phi_fixed <= self.phi_max:
                raise ValueError(f"phi_fixed must lie in the learned range [phi_min, phi_max] = "
                                 f"[{self.phi_min}, {self.phi_max}], got {phi_fixed!r}")
            self.phi_fixed = None if phi_fixed is None else float(phi_fixed)
            self.horizon_A = erlang_horizon_A(
                self.N, self.phi_min if self.phi_fixed is None else self.phi_fixed, q=self.q_quantile)

            self.phi_raw = nn.Parameter(torch.tensor(0.0), requires_grad=self.phi_fixed is None)

            self.M_theta = nn.Linear(self.input_dim * 3 + 3, self.input_dim)
            self.R_theta = nn.Linear(self.input_dim * 2 + 2, self.d_psi)
            self.rho_theta = nn.Linear(self.input_dim * 2, 1)
            self.G_gate = nn.Linear(self.input_dim * 2 + 2, 1)
            self.G_dir = nn.Linear(self.input_dim * 2 + 2, self.d_xi - 2)
            self.U_cell = nn.GRUCell(self.input_dim * 2 + 3, self.input_dim)
            self.zeta_theta = nn.Linear(self.input_dim + 1, 1)

            self.h_theta = LipschitzHazardLinear(self.d_xi, 2, self.d_psi, L_h=self.L_h)
            self.gamma_raw = nn.Parameter(torch.zeros(self.d_psi))

            self.W_dur = nn.Linear(self.input_dim, self.input_dim, bias=False)
            self.W_prov = nn.Linear(self.input_dim, self.input_dim, bias=False)
            self.beta_stage = nn.Parameter(torch.ones(self.N) / float(self.N))

            readout_in = (
                self.input_dim * 4
                + 3 * self.d_psi
                + self.d_xi
                + 2
            )
            self.Psi_theta = nn.Sequential(
                nn.Linear(readout_in, self.input_dim),
                nn.LayerNorm(self.input_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.input_dim, self.input_dim),
            )
            self.res_gate = nn.Parameter(torch.tensor(0.5))
            self.surprise_head = nn.Linear(self.input_dim * 2, 2)

            self.last_aux_loss: Optional[torch.Tensor] = None
            self._init_parameters()

        def _init_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        @property
        def phi(self) -> torch.Tensor:
            if self.phi_fixed is not None:  # upgrade G5: the fixed rate, without gradient
                return torch.tensor(self.phi_fixed, dtype=self.phi_raw.dtype, device=self.phi_raw.device)
            return self.phi_min + (self.phi_max - self.phi_min) * torch.sigmoid(self.phi_raw)

        @property
        def gamma_sigma(self) -> torch.Tensor:
            return F.softplus(self.gamma_raw)

        def _bound_payload(self, raw_mu: torch.Tensor) -> torch.Tensor:
            norm = raw_mu.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(norm / self.payload_M, min=1.0)
            return raw_mu / scale

        def _compute_kappa(self, n_bar: torch.Tensor, m_tot: torch.Tensor) -> torch.Tensor:
            inp = torch.cat([n_bar, torch.log1p(m_tot.unsqueeze(-1))], dim=-1)
            score = torch.sigmoid(self.zeta_theta(inp).squeeze(-1))
            conf = 1.0 - torch.exp(-m_tot.clamp(min=0.0) / self.m_0)
            return score * conf

        def _compute_hazard(self, xi: torch.Tensor, b_emit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            nu = F.softplus(self.h_theta(xi, b_emit))
            if not self.hazard:  # E2 ablation: lambda = 0, the forward pass of lambda_max = 0 without its 0/0 gradient
                return nu, torch.zeros_like(nu)
            gam = self.gamma_sigma.unsqueeze(0)
            lam = self.lambda_max * (1.0 - torch.exp(-(gam * nu) / self.lambda_max))
            return nu, lam

        def forward(
            self,
            x: torch.Tensor,
            train_edges: Optional[torch.Tensor] = None,
            train_labels: Optional[torch.Tensor] = None,
            index_list: Optional[List[int]] = None,
            node_labels: Optional[torch.Tensor] = None,
            witness_weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if witness_weights is not None:  # upgrade M-B: [T, n] weights of the witness term
                check_witness_weights(witness_weights, x.shape[1], x.shape[0])
            if train_edges is None or train_labels is None or index_list is None:
                return self._forward_sequence_mode(x, witness_weights=witness_weights)
            return self._forward_graph_stream(x, train_edges, train_labels, index_list, node_labels=node_labels,
                                              witness_weights=witness_weights)

        def _forward_sequence_mode(self, x: torch.Tensor,
                                   witness_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
            B, K, d = x.shape
            dev = x.device
            dtype = x.dtype

            a_state = torch.zeros(B, d, device=dev, dtype=dtype)
            xi = torch.zeros(B, self.d_xi, device=dev, dtype=dtype)
            emit_cnt = torch.zeros(B, device=dev, dtype=dtype)

            n_dur = torch.zeros(B, d, device=dev, dtype=dtype)
            m_dur = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)

            records: List[dict] = []
            sched_ev: Dict[int, torch.Tensor] = {}

            phi = self.phi
            z_seq: List[torch.Tensor] = []
            mark_losses: List[torch.Tensor] = []
            lam_last = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)

            for t in range(K):
                f_t = x[:, t, :]
                if t in sched_ev:
                    xi = (xi + sched_ev[t]).clamp(min=0.0, max=self.cap_c)

                b_emit = torch.stack([
                    torch.log1p(emit_cnt),
                    emit_cnt / float(t + 1),
                ], dim=-1)
                nu_t, lam_t = self._compute_hazard(xi, b_emit)
                lam_last = lam_t

                Pi_prov = torch.zeros(B, d, device=dev, dtype=dtype)
                pi_prov = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)

                # Incremental frozen-mass segment integration over [t-1, t]
                for rec in records:
                    if rec["closed"]:
                        continue
                    tau = rec["tau"]
                    x_a = float(t - 1 - tau)
                    x_b = min(float(t - tau), self.horizon_A)
                    if x_b > x_a:
                        inc = _torch_segment_release_increment(
                            self.N, phi, lam_t,
                            torch.full_like(lam_t, x_a),
                            torch.full_like(lam_t, x_b),
                            rec["L_exp"],
                        )
                        rec["L_exp"] = rec["L_exp"] + lam_t * (x_b - x_a)
                        if (t - tau) >= self.horizon_A:
                            Q_A, _ = _torch_erlang_Q_P(self.N, phi * self.horizon_A)
                            inc = inc + torch.exp(-rec["L_exp"]) * Q_A
                            rec["closed"] = True
                        rec["omega"] = rec["omega"] + inc
                        w_mix = (rec["psi"] * inc).sum(dim=-1, keepdim=True)
                        n_dur = n_dur + w_mix * self.W_dur(rec["mu"])
                        m_dur = m_dur + rec["psi"] * inc

                    if not rec["closed"]:
                        age = float(t - tau)
                        Q_age, _ = _torch_erlang_Q_P(self.N, phi * age)
                        p_n = _torch_erlang_stage_probs(self.N, phi * age)
                        stage_mix = (p_n * self.beta_stage).sum(dim=-1)
                        S_now = torch.exp(-rec["L_exp"])
                        pi_prov = pi_prov + rec["psi"] * S_now * Q_age
                        Pi_prov = Pi_prov + (
                            (rec["psi"] * S_now).sum(dim=-1) * stage_mix
                        ).unsqueeze(-1) * self.W_prov(rec["mu"])

                m_tot = m_dur.sum(dim=-1)
                n_bar = n_dur / (1.0 + m_tot.unsqueeze(-1))
                kappa = self._compute_kappa(n_bar, m_tot)

                delta_t = torch.full((B, 1), math.log1p(1.0), device=dev, dtype=dtype)
                dummy_y = torch.zeros(B, 2, device=dev, dtype=dtype)
                m_in = torch.cat([a_state, n_bar, f_t, dummy_y, delta_t], dim=-1)
                mu_t = self._bound_payload(F.silu(self.M_theta(m_in)))
                r_in = torch.cat([a_state, f_t, dummy_y], dim=-1)
                psi_t = F.softmax(self.R_theta(r_in), dim=-1)
                records.append({
                    "tau": t,
                    "mu": mu_t,
                    "psi": psi_t,
                    "L_exp": torch.zeros(B, self.d_psi, device=dev, dtype=dtype),
                    "omega": torch.zeros(B, self.d_psi, device=dev, dtype=dtype),
                    "closed": False,
                })

                g_gate = torch.sigmoid(self.G_gate(r_in))
                g_dir = F.softmax(self.G_dir(r_in), dim=-1)
                g_vec = g_gate * g_dir
                surprise = (f_t - n_bar).detach().pow(2).mean(dim=-1, keepdim=True)
                s_tilde = torch.clamp(surprise / self.s_max, 0.0, 1.0)
                v3_zero = torch.zeros(B, 1, device=dev, dtype=dtype)
                v1_term = kappa.unsqueeze(-1) * g_vec
                if witness_weights is not None:  # upgrade M-B
                    v1_term = v1_term * witness_weights[t].to(dev, dtype).unsqueeze(-1)
                ev_vec = torch.cat([v1_term, s_tilde, v3_zero], dim=-1)
                ev_vec = torch.clamp(ev_vec, max=self.cap_c)
                t_en = t + self.L_f
                t_ex = t + self.L_f + self.H
                sched_ev[t_en] = sched_ev.get(t_en, 0.0) + ev_vec
                sched_ev[t_ex] = sched_ev.get(t_ex, 0.0) - ev_vec

                rate_dot = (psi_t * nu_t).sum(dim=-1, keepdim=True).clamp(min=1e-6, max=20.0)
                q_mark = 1.0 - torch.exp(-rate_dot)
                mk_loss = -(
                    s_tilde * torch.log(q_mark.clamp(min=1e-6))
                    + (1.0 - s_tilde) * torch.log((1.0 - q_mark).clamp(min=1e-6))
                ).mean()
                mark_losses.append(mk_loss)

                u_in = torch.cat([f_t, n_bar, dummy_y, delta_t], dim=-1)
                a_state = self.U_cell(u_in, a_state)
                emit_cnt = emit_cnt + 1.0

                age_last = torch.zeros(B, 1, device=dev, dtype=dtype)
                psi_in = torch.cat([
                    a_state,
                    n_bar,
                    torch.log1p(m_dur),
                    Pi_prov,
                    torch.log1p(pi_prov),
                    lam_t,
                    xi,
                    kappa.unsqueeze(-1),
                    age_last,
                    f_t,
                ], dim=-1)
                z_t = f_t + torch.tanh(self.res_gate) * self.Psi_theta(psi_in)
                z_seq.append(z_t)

            reg = 0.01 * (self.lambda_max * torch.relu(lam_last.mean() - 0.5 * self.phi_min))
            self.last_aux_loss = 0.05 * torch.stack(mark_losses).mean() + reg
            return torch.stack(z_seq, dim=1)

        def _forward_graph_stream(
            self,
            x: torch.Tensor,
            train_edges: torch.Tensor,
            train_labels: torch.Tensor,
            index_list: List[int],
            node_labels: Optional[torch.Tensor] = None,
            witness_weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            B, K, d = x.shape
            dev = x.device
            dtype = x.dtype

            edges_dev = train_edges.to(dev)
            labels_dev = train_labels.to(dev, dtype=dtype)

            a_state = torch.zeros(B, d, device=dev, dtype=dtype)
            emit_cnt = torch.zeros(B, device=dev, dtype=dtype)
            emit_snaps = torch.zeros(B, device=dev, dtype=dtype)
            last_emit_t = torch.zeros(B, device=dev, dtype=dtype)

            # Durable and closed-only linear stores (mass already released is FROZEN)
            n_dur = torch.zeros(B, d, device=dev, dtype=dtype)
            m_dur = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)
            n_closed = torch.zeros(B, d, device=dev, dtype=dtype)
            m_closed = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)

            escrow_records: List[dict] = []
            # Cumulative hazard Lambda_u^sigma(t) at each snapshot boundary
            Lambda_cum = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)
            Lambda_snapshots: Dict[int, torch.Tensor] = {0: Lambda_cum.clone()}

            # Pending V1 events queued at t_j to mature at t_j + L_f
            pending_v1: Dict[int, List[dict]] = collections.defaultdict(list)
            # Matured V1 records active per snapshot m_t in (t - H, t]
            matured_v1: Dict[int, List[dict]] = collections.defaultdict(list)
            # Matured V2 self-surprise active per snapshot m_t in (t - H, t]
            matured_v2: Dict[int, torch.Tensor] = {}
            # Matured V3 revealed node labels active per snapshot m_t in (t - H, t]
            matured_v3: Dict[int, torch.Tensor] = {}

            phi = self.phi
            z_seq: List[torch.Tensor] = []
            mark_losses: List[torch.Tensor] = []
            lam_last = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)

            idx0 = 0
            for t in range(K):
                f_t = x[:, t, :]
                end = int(index_list[t]) + 1 if t < len(index_list) else 0
                has_edges = (end > idx0) and (idx0 < edges_dev.shape[1])
                if has_edges:
                    lim = slice(idx0, min(end, edges_dev.shape[1]))
                    e_t = edges_dev[:, lim]
                    y_t = labels_dev[lim, :]
                    idx0 = end
                else:
                    e_t = edges_dev[:, :0]
                    y_t = labels_dev[:0, :]
                    if end > idx0:
                        idx0 = end

                # 1. Mature V1 items arriving at t = t_j + L_f, multiplying by sender lag survival
                if t in pending_v1:
                    for item in pending_v1[t]:
                        u_j = item["u"]
                        t_j = item["t_j"]
                        dLam = Lambda_cum[u_j] - Lambda_snapshots[t_j][u_j]
                        lag_surv = torch.exp(-(item["psi"] * dLam).sum(dim=-1, keepdim=True))
                        matured_v1[t].append({
                            "u": u_j,
                            "v": item["v"],
                            "w": item["raw_w"] * lag_surv,
                        })

                # 2. Build windowed evidence xi(t) over m_t in (t - H, t]
                #    enforcing per-(subject i, contributor u) l1 cap <= cap_c
                v1_win = torch.zeros(B, self.d_xi - 2, device=dev, dtype=dtype)
                v2_win = torch.zeros(B, 1, device=dev, dtype=dtype)
                v3_win = torch.zeros(B, 1, device=dev, dtype=dtype)

                # Collect live V1 items in (t - H, t] and cap per-(v, u) pair
                live_u, live_v, live_w = [], [], []
                for m_t in range(max(0, t - self.H + 1), t + 1):
                    if m_t in matured_v1:
                        for m_item in matured_v1[m_t]:
                            live_u.append(m_item["u"])
                            live_v.append(m_item["v"])
                            live_w.append(m_item["w"])
                    if m_t in matured_v2:
                        v2_win = v2_win + matured_v2[m_t]
                    if m_t in matured_v3:
                        v3_win = v3_win + matured_v3[m_t]

                if live_u:
                    u_cat = torch.cat(live_u, dim=0)
                    v_cat = torch.cat(live_v, dim=0)
                    w_cat = torch.cat(live_w, dim=0)
                    # Hash directed pair (v, u) -> v * B + u for exact per-(subject v, contributor u) cap
                    pair_keys = v_cat * B + u_cat
                    uniq_pairs, inv_idx = torch.unique(pair_keys, return_inverse=True)
                    pair_w = torch.zeros(len(uniq_pairs), self.d_xi - 2, device=dev, dtype=dtype)
                    pair_w.index_add_(0, inv_idx, w_cat)
                    pair_l1 = pair_w.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    pair_w_capped = pair_w * torch.clamp(self.cap_c / pair_l1, max=1.0)
                    uniq_v = torch.div(uniq_pairs, B, rounding_mode="floor")
                    v1_win.index_add_(0, uniq_v, pair_w_capped)

                v2_win = torch.clamp(v2_win, min=0.0, max=self.cap_c)
                v3_win = torch.clamp(v3_win, min=0.0, max=self.cap_c)
                xi = torch.cat([v1_win, v2_win, v3_win], dim=-1)

                # 3. Repaired L_h-Lipschitz hazard refresh (Eq. 6)
                b_emit = torch.stack([
                    torch.log1p(emit_cnt),
                    emit_snaps / float(t + 1),
                ], dim=-1)
                nu_t, lam_t = self._compute_hazard(xi, b_emit)
                lam_last = lam_t

                # 4. Advance all open escrow records across segment [t - 1, t] (Eq. 2, frozen mass)
                Pi_prov = torch.zeros(B, d, device=dev, dtype=dtype)
                pi_prov = torch.zeros(B, self.d_psi, device=dev, dtype=dtype)

                for rec in escrow_records:
                    if rec["closed"]:
                        continue
                    tau = rec["tau"]
                    u_r = rec["u"]
                    v_r = rec["v"]
                    mu_r = rec["mu"]
                    psi_r = rec["psi"]
                    lam_seg = lam_t[u_r]

                    x_a = float(t - 1 - tau)
                    x_b = min(float(t - tau), self.horizon_A)
                    if x_b > x_a:
                        inc_k = _torch_segment_release_increment(
                            self.N, phi, lam_seg,
                            torch.full_like(lam_seg, x_a),
                            torch.full_like(lam_seg, x_b),
                            rec["L_exp"],
                        )
                        rec["L_exp"] = rec["L_exp"] + lam_seg * (x_b - x_a)
                        is_closing = (t - tau) >= self.horizon_A
                        if is_closing:
                            Q_A, _ = _torch_erlang_Q_P(self.N, phi * self.horizon_A)
                            S_close = torch.exp(-rec["L_exp"])
                            inc_k = inc_k + S_close * Q_A
                            rec["closed"] = True

                        rec["omega"] = rec["omega"] + inc_k
                        w_mix_inc = (psi_r * inc_k).sum(dim=-1, keepdim=True)
                        n_inc = w_mix_inc * self.W_dur(mu_r)
                        m_inc = psi_r * inc_k
                        n_dur = n_dur.index_add(0, v_r, n_inc)
                        m_dur = m_dur.index_add(0, v_r, m_inc)

                        if is_closing:
                            w_mix_tot = (psi_r * rec["omega"]).sum(dim=-1, keepdim=True)
                            n_tot = w_mix_tot * self.W_dur(mu_r)
                            m_tot_rec = psi_r * rec["omega"]
                            n_closed = n_closed.index_add(0, v_r, n_tot)
                            m_closed = m_closed.index_add(0, v_r, m_tot_rec)

                    if not rec["closed"]:
                        age = float(t - tau)
                        Q_age, _ = _torch_erlang_Q_P(self.N, phi * age)
                        p_n = _torch_erlang_stage_probs(self.N, phi * age)
                        stage_mix = (p_n * self.beta_stage).sum(dim=-1, keepdim=True)
                        S_k = torch.exp(-rec["L_exp"])
                        pi_contrib = psi_r * S_k * Q_age
                        Pi_contrib = (psi_r * S_k).sum(dim=-1, keepdim=True) * stage_mix * self.W_prov(mu_r)
                        pi_prov.index_add_(0, v_r, pi_contrib)
                        Pi_prov.index_add_(0, v_r, Pi_contrib)

                if self.closed_only_reads:
                    m_read_tot = m_closed.sum(dim=-1)
                    n_bar_read = n_closed / (1.0 + m_read_tot.unsqueeze(-1))
                else:
                    m_read_tot = m_dur.sum(dim=-1)
                    n_bar_read = n_dur / (1.0 + m_read_tot.unsqueeze(-1))

                m_tot = m_dur.sum(dim=-1)
                n_bar = n_dur / (1.0 + m_tot.unsqueeze(-1))
                kappa = self._compute_kappa(n_bar_read, m_read_tot)

                # 5. Process new events arriving at snapshot t
                if e_t.shape[1] > 0:
                    u_k = e_t[0]
                    v_k = e_t[1]
                    delta_k = torch.log1p((float(t) - last_emit_t[u_k]).clamp(min=0.0)).unsqueeze(-1)

                    m_in = torch.cat([a_state[u_k], n_bar_read[v_k], f_t[u_k], y_t, delta_k], dim=-1)
                    mu_k = self._bound_payload(F.silu(self.M_theta(m_in)))
                    r_in = torch.cat([a_state[u_k], f_t[u_k], y_t], dim=-1)
                    psi_k = F.softmax(self.R_theta(r_in), dim=-1)

                    # O2: Exact SAME-PAIR (u_k, v_k) supersession impulse
                    if escrow_records:
                        new_pair_keys = u_k * B + v_k
                        uniq_new_pairs, inv_np = torch.unique(new_pair_keys, return_inverse=True)
                        pair_mu_sum = torch.zeros(len(uniq_new_pairs), d, device=dev, dtype=dtype)
                        pair_mu_cnt = torch.zeros(len(uniq_new_pairs), 1, device=dev, dtype=dtype)
                        pair_mu_sum.index_add_(0, inv_np, mu_k)
                        pair_mu_cnt.index_add_(0, inv_np, torch.ones(len(u_k), 1, device=dev, dtype=dtype))
                        pair_mu_mean = pair_mu_sum / pair_mu_cnt.clamp(min=1.0)

                        for prev_rec in escrow_records:
                            if not prev_rec["closed"] and (t - prev_rec["tau"]) < self.horizon_A:
                                prev_keys = prev_rec["u"] * B + prev_rec["v"]
                                idx_in_new = torch.searchsorted(uniq_new_pairs, prev_keys)
                                idx_clamped = idx_in_new.clamp(max=max(len(uniq_new_pairs) - 1, 0))
                                match_mask = (idx_in_new < len(uniq_new_pairs)) & (
                                    uniq_new_pairs[idx_clamped] == prev_keys
                                )
                                if match_mask.any():
                                    imp_in = torch.cat([
                                        prev_rec["mu"][match_mask],
                                        pair_mu_mean[idx_clamped[match_mask]],
                                    ], dim=-1)
                                    imp = 0.1 * F.softplus(self.rho_theta(imp_in))
                                    new_L = prev_rec["L_exp"].clone()
                                    new_L[match_mask] = new_L[match_mask] + imp
                                    prev_rec["L_exp"] = new_L

                    escrow_records.append({
                        "tau": t,
                        "u": u_k,
                        "v": v_k,
                        "mu": mu_k,
                        "psi": psi_k,
                        "L_exp": torch.zeros(len(u_k), self.d_psi, device=dev, dtype=dtype),
                        "omega": torch.zeros(len(u_k), self.d_psi, device=dev, dtype=dtype),
                        "closed": False,
                    })

                    # Queue V1 witness evidence for maturation at t + L_f
                    g_gate = torch.sigmoid(self.G_gate(r_in))
                    g_dir = F.softmax(self.G_dir(r_in), dim=-1)
                    g_k = g_gate * g_dir
                    raw_w_v1 = kappa[u_k].unsqueeze(-1) * g_k
                    if witness_weights is not None:  # upgrade M-B
                        raw_w_v1 = raw_w_v1 * witness_weights[t].to(dev, dtype)[u_k].unsqueeze(-1)
                    pending_v1[t + self.L_f].append({
                        "u": u_k,
                        "v": v_k,
                        "raw_w": raw_w_v1,
                        "psi": psi_k,
                        "t_j": t,
                    })

                    # Queue V2 detached self-surprise for maturation at t + L_f
                    surp_logits = self.surprise_head(torch.cat([a_state[u_k], n_bar_read[v_k]], dim=-1))
                    y_cls = y_t[:, 1].long() if y_t.shape[1] > 1 else y_t[:, 0].long()
                    l_ev = F.cross_entropy(surp_logits, y_cls, reduction="none").detach()
                    s_tilde = torch.clamp(l_ev / self.s_max, 0.0, 1.0).unsqueeze(-1)
                    v2_sum = torch.zeros(B, 1, device=dev, dtype=dtype)
                    v2_cnt = torch.zeros(B, 1, device=dev, dtype=dtype)
                    v2_sum.index_add_(0, u_k, s_tilde)
                    v2_cnt.index_add_(0, u_k, torch.ones_like(s_tilde))
                    v2_mean = torch.where(v2_cnt > 0, v2_sum / v2_cnt.clamp(min=1.0), torch.zeros_like(v2_sum))
                    matured_v2[t + self.L_f] = torch.clamp(v2_mean, max=self.cap_c)

                    # Optional V3 revealed node labels
                    if node_labels is not None:
                        matured_v3[t + self.L_f] = torch.clamp(
                            node_labels.to(dev, dtype=dtype).view(B, -1)[:, :1],
                            min=0.0, max=self.cap_c
                        )

                    rate_k = (psi_k * nu_t[u_k]).sum(dim=-1, keepdim=True).clamp(min=1e-6, max=20.0)
                    q_k = 1.0 - torch.exp(-rate_k)
                    mk_loss = -(
                        s_tilde * torch.log(q_k.clamp(min=1e-6))
                        + (1.0 - s_tilde) * torch.log((1.0 - q_k).clamp(min=1e-6))
                    ).mean()
                    mark_losses.append(mk_loss)

                    ones_e = torch.ones(len(u_k), 1, device=dev, dtype=dtype)
                    u_deg = torch.zeros(B, 1, device=dev, dtype=dtype)
                    u_deg.index_add_(0, u_k, ones_e)
                    act_mask = u_deg.squeeze(-1) > 0

                    agg_nbar = torch.zeros(B, d, device=dev, dtype=dtype)
                    agg_y = torch.zeros(B, 2, device=dev, dtype=dtype)
                    agg_delta = torch.zeros(B, 1, device=dev, dtype=dtype)
                    agg_nbar.index_add_(0, u_k, n_bar_read[v_k])
                    agg_y.index_add_(0, u_k, y_t)
                    agg_delta.index_add_(0, u_k, delta_k)

                    deg_act = u_deg[act_mask].clamp(min=1.0)
                    u_in_act = torch.cat([
                        f_t[act_mask],
                        agg_nbar[act_mask] / deg_act,
                        agg_y[act_mask] / deg_act,
                        agg_delta[act_mask] / deg_act,
                    ], dim=-1)
                    a_new = a_state.clone()
                    a_new[act_mask] = self.U_cell(u_in_act, a_state[act_mask])
                    a_state = a_new

                    emit_cnt = emit_cnt + u_deg.squeeze(-1)
                    emit_snaps = emit_snaps + act_mask.to(dtype)
                    last_emit_t = torch.where(act_mask, torch.full_like(last_emit_t, float(t)), last_emit_t)

                # Advance cumulative hazard clock at end of snapshot t
                Lambda_cum = Lambda_cum + lam_t
                Lambda_snapshots[t + 1] = Lambda_cum.clone()

                age_since_last = torch.log1p((float(t) - last_emit_t).clamp(min=0.0)).unsqueeze(-1)
                psi_in = torch.cat([
                    a_state,
                    n_bar,
                    torch.log1p(m_dur),
                    Pi_prov,
                    torch.log1p(pi_prov),
                    lam_t,
                    xi,
                    kappa.unsqueeze(-1),
                    age_since_last,
                    f_t,
                ], dim=-1)
                z_t = f_t + torch.tanh(self.res_gate) * self.Psi_theta(psi_in)
                z_seq.append(z_t)

            reg = 0.01 * torch.relu(lam_last.mean() - 0.5 * self.phi_min)
            if mark_losses:
                self.last_aux_loss = 0.05 * torch.stack(mark_losses).mean() + reg
            else:
                self.last_aux_loss = reg
            return torch.stack(z_seq, dim=1)


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
    ap = argparse.ArgumentParser(description="SCEM verification and evidence-lag analyzer")
    ap.add_argument("--test", action="store_true", help="Run all symbolic & empirical checks")
    ap.add_argument("--dl-dir", default=os.path.join(root, "data", "cyberdata"))
    ap.add_argument("--out", default=os.path.join(root, "results", "scem", "scem_verification.json"))
    args = ap.parse_args()

    sym = verify_symbolic_example()
    print("=== 1. SCEM Symbolic Example & Closed-Form Verification (Section F) ===")
    for k, v in sym.items():
        print(f"  {k}: {v}")

    ds_lags = {}
    print("\n=== 2. Empirical Evidence-Lag & Escrow Revocation Census (Section G & J) ===")
    for ds in ("bitcoinalpha", "bitcoinotc", "wiki-rfa"):
        csv_p = os.path.join(args.dl_dir, f"{ds}.csv")
        if os.path.exists(csv_p):
            ds_lags[ds] = measure_dataset_evidence_lags(csv_p)
            print(f"  {ds}: {ds_lags[ds]}")

    out_data = {
        "symbolic_example": sym,
        "dataset_evidence_lags": ds_lags,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved verification report to {args.out}")


if __name__ == "__main__":
    main()
