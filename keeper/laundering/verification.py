"""H4: request-verification defences against laundering (proposal §6).

Laundering requests are AUTHENTIC erasures of genuine data, so the prediction is
that every authenticity check passes them (H4: >= 99% pass every check). This
module implements the checks a platform could run, applies them to laundering
requests and to benign reference requests (random authentic erasures of the
same size), and reports pass rates. Calibrated checks use thresholds taken from
the BENIGN distribution only, so laundering requests are judged by the same
rule as everyone else.

Checks (all applied per erasure request set E):
  membership     every requested edge exists in the platform's data with the
                 claimed endpoints and sign.
  role           the requester may ask: the subject of the edge (C1), its rater
                 (C2 edge), or the account owner (C2 account).
  fingerprint    the head-gradient fingerprint of each requested edge,
                 ||f_e|| * ||softmax(f_e W) - y_e|| (closed form), lies inside the
                 central 99% of member edges' fingerprints. A forged or
                 non-member edge would be an outlier.
  estimation     UnlearnGuard-style (WWW'25): the server estimates the expected
                 unlearning update with a one-step influence (Newton) estimate and
                 compares it with the update the unlearning operator produced;
                 cosine >= the benign 1st percentile passes.
  rate_limit     at most ``rate_limit`` erasure requests touching one target in
                 the window (per-target monitoring). C3 never touches the target.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from keeper.laundering import attack as A  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.laundering import unlearn as U  # noqa: E402


@torch.no_grad()
def head_fingerprints(m, Tr, tt) -> np.ndarray:
    """Closed-form norm of each training edge's loss gradient w.r.t. the head W."""
    z = C.embed(m, Tr, tt)
    zd = z.device
    f = torch.cat((z[torch.as_tensor(Tr.edges[:, 0], device=zd)],
                   z[torch.as_tensor(Tr.edges[:, 1], device=zd)]), 1)
    p = torch.softmax(f @ m.regression_weights.to(zd), 1)
    y = torch.nn.functional.one_hot(torch.as_tensor(S.class_ids(Tr.labels), device=zd), 2)
    return (f.norm(dim=1) * (p - y.float()).norm(dim=1)).cpu().numpy()


def flat_delta(m, m0) -> torch.Tensor:
    return torch.cat([(p.detach() - q.detach()).ravel()
                      for p, q in zip(m.parameters(), m0.parameters())])


def check_request(Tr, D_all_keys, rows, requester_role, fp, fp_band):
    """Membership, role and fingerprint checks for one request set (Tr rows)."""
    e = Tr.edges[rows]
    sign = np.where(S.neg_mask(Tr.labels[rows]), -1, 1)
    member = all((int(a), int(b), int(s)) in D_all_keys for (a, b), s in zip(e, sign))
    role = requester_role in ("subject", "rater", "account_owner")
    lo, hi = fp_band
    fp_ok = bool(((fp[rows] >= lo) & (fp[rows] <= hi)).all())
    return dict(membership=member, role=role, fingerprint=fp_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n", type=int, default=30, help="laundering and benign requests each")
    ap.add_argument("--rate-limit", type=int, default=3)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    out_name = a.out or f"verification_{a.ds}_s{a.seed}.json"
    C.result_path(out_name, a.force)          # refuse an existing result before the work, not after it
    T0 = time.time()
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(a.ds)
    mem = C.apply_memory_profile(D)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = C.Panel(Tr, n_probe=256, seed=1234)
    from keeper.laundering.probe4 import load_calibration
    hp = load_calibration(a.calibration or os.path.join(C.RES, f"calibration_{a.ds}_s0.json"))
    cohort, neg, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    m0 = C.train(C.make_model(D, Tr, a.seed, tt), Tr, tt, epochs=a.epochs)
    keys = {(int(x), int(y), int(s)) for (x, y), s in
            zip(D.edges, np.where(S.neg_mask(D.labels), -1, 1))}
    fp = head_fingerprints(m0, Tr, tt)
    fp_band = (float(np.quantile(fp, 0.005)), float(np.quantile(fp, 0.995)))

    rng = np.random.default_rng(2_000 + a.seed)
    tg = np.sort(rng.choice(cohort, min(a.n, len(cohort)), replace=False))
    reqs = []
    for v in tg.tolist():
        rows = np.asarray(neg[v], dtype=np.int64)
        reqs.append(dict(kind="launder_C1", target=v, rows=rows, role="subject"))
        reqs.append(dict(kind="launder_C2_edge", target=v, rows=rows, role="rater"))
        sc = A.Scorer(m0, Tr, tt, panel, v)
        p3 = A.plan_indirect(sc, Tr, v, len(rows))
        r3 = np.where(p3.mask(Tr))[0]
        if len(r3):
            reqs.append(dict(kind="launder_C3", target=v, rows=r3, role="rater"))
    for i in range(min(a.n, len(tg))):
        k = len(neg[int(tg[i])])
        reqs.append(dict(kind="benign_random", target=None,
                         rows=np.sort(rng.choice(len(Tr.edges), k, replace=False)),
                         role="rater"))

    for r in reqs:
        r.update(check_request(Tr, keys, r["rows"], r["role"], fp, fp_band))
        mask = np.zeros(len(Tr.edges), bool)
        mask[r["rows"]] = True
        d_op = flat_delta(U.neggrad_plus(m0, G, Tr, mask, tt, **hp["neggrad"]).model, m0)
        est = (U.influence_unlearn(m0, G, Tr, mask, tt, **hp["influence"])
               if hp["influence_feasible"] else
               U.head_newton(m0, G, Tr, mask, tt, **hp["head_newton"]))
        d_est = flat_delta(est.model, m0)
        denom = float(d_op.norm() * d_est.norm())
        r["estimation_cos"] = float((d_op @ d_est) / denom) if denom > 0 else None
        if r["target"] is not None:
            touches = int(((Tr.edges[r["rows"], 1] == r["target"])
                           | (Tr.edges[r["rows"], 0] == r["target"])).sum())
            r["requests_touching_target"] = touches
            r["rate_limit"] = touches <= a.rate_limit
        else:
            r["rate_limit"] = True
    ben = [r["estimation_cos"] for r in reqs
           if r["kind"] == "benign_random" and r["estimation_cos"] is not None]
    tau = float(np.quantile(ben, 0.01)) if ben else None
    for r in reqs:
        r["estimation"] = (r["estimation_cos"] is not None and tau is not None
                           and r["estimation_cos"] >= tau)
        r["passes_all"] = all(r[c] for c in ("membership", "role", "fingerprint",
                                              "estimation", "rate_limit"))
    summary = {}
    for kind in sorted({r["kind"] for r in reqs}):
        rs = [r for r in reqs if r["kind"] == kind]
        summary[kind] = dict(n=len(rs), pass_all=float(np.mean([r["passes_all"] for r in rs])),
                             **{c: float(np.mean([r[c] for r in rs]))
                                for c in ("membership", "role", "fingerprint", "estimation",
                                          "rate_limit")})
    laund = [r["passes_all"] for r in reqs if r["kind"].startswith("launder")]
    out = dict(ds=a.ds, seed=a.seed, epochs=a.epochs, memory_profile=mem,
               fingerprint_band=fp_band, estimation_threshold_benign_p1=tau,
               rate_limit=a.rate_limit, calibration=hp, summary=summary,
               h4=dict(pass_rate=float(np.mean(laund)), n=len(laund),
                       verdict="SUPPORTED" if np.mean(laund) >= 0.99 else "NOT SUPPORTED"),
               requests=[{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in r.items()} for r in reqs],
               runtime_s=round(time.time() - T0, 1))
    path = C.dump(out, out_name, D, force=a.force)
    print(f"[{a.ds} s{a.seed}] H4 pass-all {out['h4']['pass_rate']:.3f} over {len(laund)} "
          f"laundering requests | per kind: "
          + ", ".join(f"{k} {v['pass_all']:.2f}" for k, v in summary.items()))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
