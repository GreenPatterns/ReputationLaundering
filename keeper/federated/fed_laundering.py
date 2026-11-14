"""Federated setting (proposal §5c, H6): does client-side unlearning survive
robust aggregation?

The framework's federated stack is used as is: fedlab's PartitioningCyberData
(clients are contiguous time windows), the framework's client trainer for the
FedAvg warm-up and for every honest client's NORMAL update, and the harness's
aggregators. A deletion request is served by the client that holds the erased
edges: it uploads a NegGrad+ unlearning update computed by
:func:`laundering.unlearn.neggrad_plus` on its local data. That is the corrected
operator: retained graph, class-weighted criterion, clipping, and calibrated
hyperparameters.

Per target and per aggregator A (FedAvg, median, trimmed mean, geometric
median, Krum):
  theta_A^unl   A(updates with the owning client's unlearning update)
  theta_A^norm  A(all normal updates)
  forgetting    CE on F in the SERVED state (owning client's window minus F)
                under theta_A^unl minus under theta_A^norm
  retained      forgetting_A / forgetting_FedAvg            (H6: < 0.5 for
                geometric median and Krum)
  projection    <theta_A^unl - theta_A^norm, d> / <same for FedAvg, d>, where
                d = unlearning update - the owning client's normal update
  laundering    target reputation on the client window minus F under
                theta_A^unl, relative to theta* on the full window
"Exempt" applies plain FedAvg to the deletion round whatever A is (§5c: exempting
deletion rounds gives laundering the full effect).

Targets: strict distrust-majority (<= 3 distrust in-edges) whose distrust in-
edges ALL lie in one client's local training prefix, so the request has a
single owner.

Run: CUDA_VISIBLE_DEVICES=0 <ps python> -m laundering.fed_laundering --ds bitcoinalpha --seed 0
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.laundering import unlearn as U  # noqa: E402

AGGS = ("fedavg", "median", "trimmed", "geomedian", "krum")


def setup_fl(ds, seed, warmup):
    """FedAvg warm-up with the framework's trainer, with the D3 fix: features and
    class weights from training edges only (the client snapshots, minus the global
    held-out snapshot)."""
    from keeper.federated.run import load_config, setup_models_and_utils, get_device
    from keeper.third_party.fedlab.partitioned_HetroEdges import PartitioningCyberData
    from keeper.third_party.fedlab.serialization import SerializationTool
    from keeper.federated.fed_harness import build_args, make_trainer, agg_fedavg
    cfg = load_config(ds)
    device = get_device()
    LocalGDTE, read_graph, setup_features, _, _, pp = setup_models_and_utils()
    args = build_args(cfg, ds, pp, seed=seed, rounds=warmup, attack=False)
    C.seed_all(seed)
    gs = PartitioningCyberData(args=args, wrkr=args.total_client)
    args.num_nodes = int(cfg["num_nodes"])        # full roster: node ids index X
    args.max_snapshots_inaClient = gs.maax
    # training edges = every client snapshot except the very last (global held-out)
    clients = [gs.get_dataloader(c) for c in range(args.total_client)]
    snaps = [s for c in clients for s in c["client_edges"]]
    labs = [s for c in clients for s in c["client_labels"]]
    tr_e = np.concatenate(snaps[:-1])
    tr_l = np.concatenate(labs[:-1])
    X = torch.from_numpy(setup_features({"edges": tr_e, "ncount": args.num_nodes,
                                         "labels": tr_l}, seed=seed,
                                        dim=args.layers[-1])).to(device)
    y = S.class_ids(tr_l)
    cw = torch.FloatTensor(1.0 / np.bincount(y, minlength=2).clip(min=1) * len(y)).to(device)
    model = LocalGDTE(device, args, X, num_labels=2, index_list=[0], class_weight=cw).to(device)
    C.set_deterministic(True)
    trainer = make_trainer(model, cfg, args, device, [])
    trainer.setup_dataset(gs)
    trainer.set_experimental_setting(args.task)
    gp = SerializationTool.serialize_model(model).clone().cpu()
    for rnd in range(warmup):
        trainer._current_round = rnd
        trainer.local_process([gp], list(range(args.total_client)))
        packs = sorted(trainer.uplink_package, key=lambda p: int(p[3]))
        gp = agg_fedavg([p[0].to(gp.device) for p in packs]).cpu()
        SerializationTool.deserialize_model(model, gp.to(device))
    # one more pass at theta* to capture every client's NORMAL update
    trainer._current_round = warmup
    trainer.local_process([gp], list(range(args.total_client)))
    packs = sorted(trainer.uplink_package, key=lambda p: int(p[3]))
    normal = {int(p[3]): p[0].to(gp.device).clone() for p in packs}
    return dict(args=args, model=model, gs=gs, theta=gp, normal=normal, device=device)


def client_graph(gs, cid, n):
    """A client's local data as a Graph (snapshots local) and its training prefix."""
    cd = gs.get_dataloader(cid)
    se, sl = cd["client_edges"], cd["client_labels"]
    e = np.concatenate(se).astype(np.int64)
    lab = np.concatenate(sl).astype(np.float32)
    snap = np.concatenate([np.full(len(s), i, np.int64) for i, s in enumerate(se)])
    G = C.Graph(e, lab, np.zeros(len(e), np.int64), snap, n)
    tt = len(se) - 1                      # train on prefix [0, k-1), as the client does
    return G, C.train_split(G, tt), tt


def aggregate(name, vecs):
    from keeper.federated import fed_harness as H
    if name == "fedavg":
        return H.agg_fedavg(vecs)
    if name == "median":
        return H.agg_median(vecs)
    if name == "trimmed":
        return H.agg_trimmed(vecs, trim=1)
    if name == "geomedian":
        return H.agg_geomedian(vecs)
    if name == "krum":
        return H.agg_krum(vecs, n_malicious=1)[0]
    raise ValueError(name)


def main():
    from keeper.third_party.fedlab.serialization import SerializationTool
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    out_name = a.out or f"fed_laundering_{a.ds}_s{a.seed}.json"
    C.result_path(out_name, a.force)          # refuse an existing result before the work, not after it
    T0 = time.time()
    C.require_gpu()
    from keeper.laundering.probe4 import load_calibration
    hp = load_calibration(a.calibration or os.path.join(C.RES, f"calibration_{a.ds}_s0.json"))
    S0 = setup_fl(a.ds, a.seed, a.warmup)
    args, model, gs, theta, normal = S0["args"], S0["model"], S0["gs"], S0["theta"], S0["normal"]
    dev = S0["device"]
    n = args.num_nodes
    K = args.total_client

    # candidate targets whose distrust in-edges all sit in ONE client's training prefix
    owners = {}
    local = {c: client_graph(gs, c, n) for c in range(K)}
    # the §7 cohort is defined on the WHOLE training graph (union of client prefixes)
    allTr = C.Graph(np.concatenate([L[1].edges for L in local.values()]),
                    np.concatenate([L[1].labels for L in local.values()]),
                    np.zeros(sum(len(L[1].edges) for L in local.values()), np.int64),
                    np.zeros(sum(len(L[1].edges) for L in local.values()), np.int64), n)
    _gc, _gneg, gpos = C.pick_targets(allTr, max_neg=3, strict=True)
    global_cohort = set(_gc.tolist())
    for c, (Gc, Trc, ttc) in local.items():
        coh, neg, pos = C.pick_targets(Trc, max_neg=3, strict=False)
        for v in coh.tolist():
            if v not in global_cohort:
                continue
            # every distrust in-edge of v anywhere in training must be in this client
            tot = sum(int(((L[1].edges[:, 1] == v) & S.neg_mask(L[1].labels)).sum())
                      for L in local.values())
            if tot == len(neg[v]):
                owners[v] = (c, np.asarray(neg[v]), len(gpos.get(v, [])))
    rng = np.random.default_rng(7)
    tg = sorted(rng.choice(sorted(owners), min(a.n_targets, len(owners)), replace=False).tolist())
    print(f"[{a.ds} s{a.seed}] clients {K}, maax {gs.maax}, single-owner targets "
          f"{len(owners)}, probing {len(tg)} | hp {hp['neggrad']}", flush=True)

    def load(vec):
        SerializationTool.deserialize_model(model, vec.to(dev))
        return model

    recs = []
    for v in tg:
        c, rows, npos = owners[v]
        Gc, Trc, ttc = local[c]
        mask = np.zeros(len(Trc.edges), bool)
        mask[rows] = True
        served = Trc.subset(~mask)
        panel = C.Panel(Trc, n_probe=256, seed=1234)
        r_star = float(panel.reputation(load(theta), C.embed(load(theta), Trc, ttc), [v])[0])
        u = U.neggrad_plus(load(theta), Gc, Trc, mask, ttc, **hp["neggrad"])
        unl = SerializationTool.serialize_model(u.model).clone().cpu()
        d = unl - normal[c]
        base_vecs = [normal[i] for i in range(K)]
        unl_vecs = [unl if i == c else normal[i] for i in range(K)]
        rec = dict(target=v, owner=c, n_neg=int(len(rows)), n_pos=int(npos),
                   stratum=M.stratum_of(npos), r_star=r_star, aggs={})
        for agg in AGGS + ("exempt_fedavg",):
            name = "fedavg" if agg == "exempt_fedavg" else agg
            th_u = aggregate(name, unl_vecs).cpu()
            th_n = aggregate(name, base_vecs).cpu()
            fce_u = U.forget_ce_served(load(th_u), served, Trc, mask, ttc)
            rep_u = float(panel.reputation(load(th_u), C.embed(load(th_u), served, ttc), [v])[0])
            fce_n = U.forget_ce_served(load(th_n), served, Trc, mask, ttc)
            rec["aggs"][agg] = dict(forgetting=fce_u - fce_n,
                                    projection=float((th_u - th_n) @ d),
                                    laundering_gain=rep_u - r_star)
        ref_f = rec["aggs"]["fedavg"]["forgetting"]
        ref_p = rec["aggs"]["fedavg"]["projection"]
        for agg, x in rec["aggs"].items():
            x["retained_forgetting"] = M.safe_div(x["forgetting"], ref_f) if abs(ref_f) > 1e-9 else None
            x["retained_projection"] = M.safe_div(x["projection"], ref_p) if abs(ref_p) > 1e-12 else None
        recs.append(rec)
        print(f"  v={v} client {c} k={len(rows)} | retained (forget/proj): " + ", ".join(
            f"{g} {rec['aggs'][g]['retained_forgetting']}/{rec['aggs'][g]['retained_projection']}"
            for g in ("geomedian", "krum", "median")), flush=True)

    summ = {}
    for agg in AGGS + ("exempt_fedavg",):
        for key in ("retained_forgetting", "retained_projection", "laundering_gain", "forgetting"):
            vals = [r["aggs"][agg][key] for r in recs]
            summ.setdefault(agg, {})[key] = M.by_stratum(vals, [r["stratum"] for r in recs])
    h6 = {agg: summ[agg]["retained_projection"]["all"]["mean"] for agg in ("geomedian", "krum")}
    out = dict(ds=a.ds, seed=a.seed, warmup=a.warmup, clients=K, maax=int(gs.maax),
               calibration=hp, n_single_owner_targets=len(owners), targets=recs,
               summary=summ,
               h6=dict(retained_projection=h6,
                       verdict_single_seed=("SUPPORTED" if all(x is not None and x < 0.5
                                                               for x in h6.values())
                                            else "NOT SUPPORTED")),
               runtime_s=round(time.time() - T0, 1))
    path = C.dump(out, out_name, force=a.force)
    print(f"H6 (single seed) retained projection {h6} -> {out['h6']['verdict_single_seed']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
