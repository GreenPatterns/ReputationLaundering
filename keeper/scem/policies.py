"""Release policies of the SCEM streaming engine (Task 3.3; the E1 release-policy study).

keeper.scem.scem.run_scem_streaming_weights_np keeps three quantities per record and channel: the survival
S = exp(-L) of its hazard exponent L (hazard and supersession impulses), the released mass omega, and the
provisional mass pi = S Q(x). Q(x) is the share of the record that would still be in escrow at age x had no
hazard acted. A release policy supplies four things:

* escrowed(x), which is Q(x);
* released(lam, x_a, x_b, L_a), the mass released over the ages [x_a, x_b] under the constant hazard lam,
  starting from the exponent L_a;
* A, the closure age, at which the surviving escrowed mass S(A) Q(A) is committed;
* is_open(age), whether a record of that age can still be superseded by a later rating of the same pair.
A policy also says whether the durable tallies behind kappa and V2 read the provisional mass
(reads_provisional); only no_commit does (ruling R57).

The engine books a segment's hazard loss as S(x_a) Q(x_a) - S(x_b) Q(x_b) - released, and a supersession
impulse's loss as (1 - exp(-impulse)) S Q. So omega + pi + revoked = 1 per record under every policy, and
released mass never decreases. The policies are those of the E1 grid:

* erlang (the default): Erlang(N, phi) release and closure at A = F^-1_Erlang(N, phi)(q). It calls the
  engine's own closed forms with the engine's arguments, so the default engine is unchanged bit for bit.
  N = 1 is the exponential policy.
* arrival: the whole record is released at age 0+, before any hazard acts, and it is never open to
  supersession, so omega = 1 and nothing is revocable. The engine closes it at the end of its first
  snapshot (A = 1), with nothing left to commit.
* quarantine: nothing is released before age D. The record closes at A = D and commits its surviving
  mass S(D); until then pi = S.
* no_commit: nothing is ever released or closed (A = inf), so omega = 0 and pi = S over the whole stream.
  This is the retroactive-reweighting baseline. Every later rating of the same pair supersedes the record.
  The durable tallies behind kappa and V2 hold each record's surviving weight S instead of released mass, and
  follow it retroactively (ruling R57), so no_commit differs from erlang in commitment only.
"""
import math
from typing import List

import numpy as np

POLICIES = ("erlang", "arrival", "quarantine", "no_commit")


class ErlangRelease:
    """Erlang(N, phi) release; closure at A = F^-1_Erlang(N, phi)(q)."""
    name = "erlang"
    reads_provisional = False

    def __init__(self, N: int, phi: float, q: float):
        from keeper.scem import scem   # imported here, not at the top: keeper.scem.scem imports this module
        self._scem, self.N, self.phi = scem, N, phi
        self.A = scem.erlang_horizon_A(N, phi, q=q)

    def escrowed(self, x):
        return self._scem.erlang_Q_P_np(self.N, self.phi * x)[0]

    def released(self, lam, x_a, x_b, L_a):
        return self._scem.segment_release_increment_np(self.N, self.phi, lam, x_a, x_b, L_a)

    def is_open(self, age: float) -> bool:
        return age < self.A


class ArrivalRelease:
    """Released whole at age 0+ and never open to supersession; closed at the end of its first snapshot."""
    name = "arrival"
    A = 1.0
    reads_provisional = False

    @staticmethod
    def escrowed(x):
        return np.where(np.asarray(x, dtype=np.float64) > 0.0, 0.0, 1.0)

    @staticmethod
    def released(lam, x_a, x_b, L_a):
        """The surviving mass exp(-L_a) in the segment that starts at age 0, before its hazard acts; 0 later."""
        return np.where(np.asarray(x_a, dtype=np.float64) > 0.0, 0.0, np.exp(-np.asarray(L_a, dtype=np.float64)))

    @staticmethod
    def is_open(age: float) -> bool:
        return False


class QuarantineRelease:
    """Nothing released before age D; closure at A = D commits the surviving mass S(D)."""
    name = "quarantine"
    reads_provisional = False

    def __init__(self, D: float):
        self.A = D

    @staticmethod
    def escrowed(x):
        return np.ones_like(np.asarray(x, dtype=np.float64))

    @staticmethod
    def released(lam, x_a, x_b, L_a):
        return np.zeros(np.broadcast(lam, x_a, x_b, L_a).shape)

    def is_open(self, age: float) -> bool:
        return age < self.A


class NoCommitRelease(QuarantineRelease):
    """Never released or closed: a quarantine that never ends (A = inf), so pi = S over the whole stream. The
    tallies read the surviving weight S retroactively (ruling R57)."""
    name = "no_commit"
    reads_provisional = True

    def __init__(self):
        super().__init__(math.inf)

    @staticmethod
    def provisional_read(psi, pi_prev, exponent):
        """The change of each record's psi-mixed surviving weight since the tallies last read it: S =
        exp(-exponent) after the segment, minus pi at the previous snapshot (0 for a record that just arrived)."""
        return np.sum(psi * (np.exp(-exponent) - pi_prev), axis=-1)


def release_policy(name: str = "erlang", N: int = 2, phi: float = 0.5, q: float = 0.99, quarantine_D: float = 4):
    """The engine's release policy `name`. N, phi and q belong to the Erlang policy, quarantine_D to the
    quarantine; the other policies ignore them."""
    if name == "erlang":
        return ErlangRelease(N, phi, q)
    if name == "arrival":
        return ArrivalRelease()
    if name == "quarantine":
        D = float(quarantine_D)
        if not (math.isfinite(D) and D > 0):
            raise ValueError(f"quarantine_D must be positive and finite, got {quarantine_D}")
        return QuarantineRelease(D)
    if name == "no_commit":
        return NoCommitRelease()
    raise ValueError(f"unknown release policy {name!r}; expected one of {POLICIES}")


def record_masses(psi_k, omega_ch, pi_ch, revoked_haz_ch, revoked_sup_ch, closed, snaps_omega: List[np.ndarray],
                  snaps_pi: List[np.ndarray], witness_kappa) -> dict:
    """The engine's per-record output (return_records): the channel masses mixed by each record's routing
    psi_k, unclipped, so that omega + pi + revoked = 1 holds up to rounding. omega_by_snap and pi_by_snap
    [K_snap, E] hold the released and the provisional mass at the end of each snapshot (0 before a record
    arrives). The released share of the surviving mass at age a of a record k from snapshot t_k is
    omega_by_snap[t_k + a - 1, k] / (omega_by_snap + pi_by_snap)[t_k + a - 1, k]. witness_kappa is the witness
    reliability with which each rating registered its evidence about its receiver (the stored rule, or M-B);
    keeper.scem.property1.property1_bound reads it."""
    def mix(channels):
        return np.sum(psi_k * channels, axis=-1)

    E = len(psi_k)
    rev_haz, rev_sup = mix(revoked_haz_ch), mix(revoked_sup_ch)
    return {"omega": mix(omega_ch), "pi": mix(pi_ch), "revoked_haz": rev_haz, "revoked_sup": rev_sup,
            "revoked": rev_haz + rev_sup, "closed": np.asarray(closed, dtype=bool).copy(),
            "omega_by_snap": np.asarray(snaps_omega, dtype=np.float64).reshape(len(snaps_omega), E),
            "pi_by_snap": np.asarray(snaps_pi, dtype=np.float64).reshape(len(snaps_pi), E),
            "witness_kappa": np.asarray(witness_kappa, dtype=np.float64).copy()}
