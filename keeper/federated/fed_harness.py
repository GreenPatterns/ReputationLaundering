"""
fed_harness.py — attack/defense FL harness for T-FedTrust.

Self-contained federated loop that REUSES thecode/ (model, trainer, attack,
SPQSA) without modifying it.  Adds:
  * per-client attack gating (the stock run.py path attacks ALL clients);
  * a pluggable server-side robust aggregator
    {fedavg/nodefense, krum, median, trimmed, geomedian, normclip};
  * per-round oracle victim-trust evaluation (the targeted ASR metric).

Aggregation is done directly on the per-client serialized model vectors that
the trainer already produces, so the server-side defense plugs in cleanly.
"""
import os, sys, time, json
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("S3R_CHUNK_SIZE", "256")
os.environ.setdefault("S3R_ATTN_CHUNK", "2048")

import torch.nn.functional as F
from keeper.third_party.fedlab.serialization import SerializationTool
from keeper.federated.secure_compression import decrypt_and_decompress
from keeper.backbone.attacks import initiate_attack


# ----------------------------------------------------------------------------
# Oracle victim evaluation: the TRUE attack-success metric is the global model's
# trust toward the attacker's actual injected victims (a tiny, focused set the
# defender does not know). Measured for evaluation only; no defense sees it.
# ----------------------------------------------------------------------------
def identify_victims(gs, attacker_ids, attack_type, vp, seed, device):
    if attack_type == "slow_poisoning":
        atype, vfrac = "good_mouthing", 0.15
    else:
        atype, vfrac = attack_type, vp
    victims = set()
    for aid in attacker_ids:
        cd = gs.get_dataloader(aid)
        for e_np, l_np in zip(cd["client_edges"], cd["client_labels"]):
            if len(e_np) == 0:
                continue
            e = torch.from_numpy(np.array(e_np, dtype=np.int64).T).to(device)
            l = torch.from_numpy(np.array(l_np, dtype=np.float32)).to(device)
            try:
                pe, _ = initiate_attack(e, l, attack_type=atype, attack_percentage=1.0,
                                        victim_percentage=vfrac, seed=seed)
            except Exception:
                continue
            if pe.shape[1] > e.shape[1]:
                victims.update(int(v) for v in pe[1, e.shape[1]:].tolist())
    return sorted(victims)


def get_victims(dataset, attacker_ids, attack_type="good_mouthing", victim_pct=0.5, seed=42):
    """Compute the attacker's victim node set for a config (for consistent eval)."""
    from keeper.federated.run import load_config, setup_models_and_utils, get_device
    from keeper.third_party.fedlab.partitioned_HetroEdges import PartitioningCyberData
    cfg = load_config(dataset); device = get_device()
    *_, parameter_parser = setup_models_and_utils()
    args = build_args(cfg, dataset, parameter_parser, seed=seed, attack=True,
                      attacked_clients=attacker_ids, attack_type=attack_type, victim_pct=victim_pct)
    import numpy as _np, torch as _t
    _t.manual_seed(seed); _t.cuda.manual_seed_all(seed); _np.random.seed(seed)
    gs = PartitioningCyberData(args=args, wrkr=args.total_client)
    return identify_victims(gs, attacker_ids, attack_type, victim_pct, seed, device)


def build_victim_probe(graph_data, index_list, victims, max_seq, n=5, seed=0):
    """Build a late-context probe of (trusted source -> victim) edges."""
    edges = np.asarray(graph_data["edges"], dtype=np.int64)
    labels = np.asarray(graph_data["labels"], dtype=np.float32)
    T = len(index_list)
    n = min(n, max_seq, T)
    ids = list(range(T - n, T))
    c_start = 0 if ids[0] == 0 else int(index_list[ids[0] - 1]) + 1
    c_end = int(index_list[ids[-1]]) + 1
    ce = edges[c_start:c_end]; cl = labels[c_start:c_end]
    rel = [int(index_list[i]) - c_start for i in ids]; rel[-1] = len(ce) - 1
    present = set(ce.flatten().tolist())
    vlist = [v for v in victims if v in present]
    srcs = list(dict.fromkeys(ce[:, 0].tolist()))[:60]
    ps, pd = [], []
    for v in vlist:
        for u in srcs[:8]:
            if u != v:
                ps.append(u); pd.append(v)
    return {
        "context_edges": torch.from_numpy(ce.T.copy()).long(),
        "context_labels": torch.from_numpy(cl.copy()).float(),
        "context_index_list": rel, "context_T": len(ids),
        "probe_src": torch.tensor(ps, dtype=torch.long),
        "probe_dst": torch.tensor(pd, dtype=torch.long),
        "n_victims_present": len(vlist),
    }


@torch.no_grad()
def victim_trust(model, vprobe, device):
    if vprobe is None or vprobe["probe_src"].numel() == 0:
        return float("nan")
    model.eval()
    oi, oT = model.index_list, model.args.train_time_slots
    model.index_list = vprobe["context_index_list"]; model.args.train_time_slots = vprobe["context_T"]
    ce = vprobe["context_edges"].to(device); cl = vprobe["context_labels"].to(device)
    y = cl[:, 1].long()
    try:
        _, z = model(ce, y, cl, return_loss=False)
        W = model.regression_weights.to(device)
        s = vprobe["probe_src"].to(device); d = vprobe["probe_dst"].to(device)
        pair = torch.cat((z[s], z[d]), 1)
        tv = F.softmax(torch.mm(pair, W), 1)[:, 0].mean().item()
    finally:
        model.index_list, model.args.train_time_slots = oi, oT
    return tv


# ----------------------------------------------------------------------------
# Robust aggregators over per-client serialized model vectors
# ----------------------------------------------------------------------------
def agg_fedavg(params, weights=None):
    P = torch.stack(params)              # (N, D)
    if weights is None:
        return P.mean(dim=0)
    w = torch.as_tensor(weights, dtype=P.dtype, device=P.device).view(-1, 1)
    return (P * w).sum(dim=0)


def agg_median(params, **kw):
    return torch.stack(params).median(dim=0).values


def agg_trimmed(params, trim=1, **kw):
    P = torch.stack(params)
    n = P.shape[0]
    if 2 * trim >= n:
        return P.mean(dim=0)
    s, _ = torch.sort(P, dim=0)
    return s[trim:n - trim].mean(dim=0)


def agg_geomedian(params, iters=8, eps=1e-8, **kw):
    """Geometric median (RFA, Pillutla et al. 2022) via Weiszfeld iteration.
    Certified bounded influence / 0.5 breakdown point; no detection needed."""
    P = torch.stack(params)                       # (N, D)
    z = P.mean(dim=0)
    for _ in range(iters):
        d = torch.norm(P - z, dim=1).clamp(min=eps)   # (N,)
        w = 1.0 / d
        z = (w.view(-1, 1) * P).sum(dim=0) / w.sum()
    return z


def agg_krum(params, n_malicious=1, **kw):
    P = torch.stack(params)              # (N, D)
    n = P.shape[0]
    d = torch.cdist(P, P) ** 2           # (N,N) squared distances
    m = n - n_malicious - 2
    m = max(1, m)
    scores = []
    for i in range(n):
        di = d[i].clone()
        di[i] = float("inf")
        closest, _ = torch.sort(di)
        scores.append(closest[:m].sum().item())
    pick = int(np.argmin(scores))
    return P[pick], pick


# ----------------------------------------------------------------------------
# Attack-gated trainer
# ----------------------------------------------------------------------------
def make_trainer(model, cfg, args, device, attacked_clients):
    from keeper.federated.basic_client2 import SGDSerialClientTrainer

    class AttackGatedTrainer(SGDSerialClientTrainer):
        """Gate the slow-poisoning attack to ONLY the listed malicious clients."""
        def local_process(self, payload, id_list):
            self._gate_ids = set(self.malicious_clients or [])
            return super().local_process(payload, id_list)

        def train(self, model_parameters, cdata_dict, id, **kw):
            # toggle the model's attack flag per client
            self._model.attack = bool(self._args.attack) and (id in set(self.malicious_clients or []))
            return super().train(model_parameters, cdata_dict, id, **kw)

    tr = AttackGatedTrainer(model, cfg["fl"]["n_clients"], cuda=(device.type == "cuda"),
                            device=device, args=args)
    tr.setup_optim(args.epochs, args.lr, args.weight_decay)
    tr.set_malicious_clients(attacked_clients)
    return tr


def build_args(cfg, dataset, parameter_parser, **over):
    args = parameter_parser()
    args.dataset = dataset
    args.edge_path = os.path.join(PROJECT, cfg["edge_path"])
    args.data_path = os.path.join(PROJECT, cfg["data_path"])
    args.num_nodes = cfg["num_nodes"]
    args.attention_head = cfg["n_heads"]
    args.layers = cfg["layers"]
    args.lr = cfg["lr"]; args.weight_decay = cfg["weight_decay"]; args.dropout = cfg["dropout"]
    args.seed = over.get("seed", cfg["seed"])
    args.total_client = cfg["fl"]["n_clients"]
    args.time_slots = cfg["fl"]["time_slots"]
    args.train_time_slots = cfg["fl"]["time_slots"]
    args.epochs = cfg["fl"]["local_epochs"]
    args.sample_ratio = 1.0
    args.com_round = over.get("rounds", cfg["fl"]["rounds"])
    args.fedprox_mu = over.get("mu", 0.0)
    args.fed_algo = "fedavg"
    args.fedskip_interval = 0; args.fednova_p = 1.0; args.feddyn_alpha = 0.0
    args.moon_mu = 0.0; args.moon_tau = 0.5
    args.attack = over.get("attack", False)
    args.attack_type = over.get("attack_type", "slow_poisoning")
    args.victim_percentage = over.get("victim_pct", 0.15)
    args.attacked_clients = over.get("attacked_clients", [3])
    args.secure_compress = over.get("secure_compress", False)
    args.adaef = over.get("adaef", False)
    args.tap_noise = over.get("tap_noise", False)
    args.target_sparsity = over.get("target_sparsity", 0.9)
    args.num_bits = over.get("num_bits", 8)
    args.dp_epsilon = over.get("dp_epsilon", 8.0)
    args.adaef_K = over.get("adaef_K", 5)
    args.shared_secret = over.get("shared_secret", 98765)
    args.task = over.get("task", "Task1")
    args.ablation = False; args.defense = False
    return args


def _reconstruct(pack, global_params, round_id, secret):
    """Return a full client model vector from an uplink pack[0]."""
    raw = pack[0]
    if isinstance(raw, dict) and "payload" in raw:
        cid = int(pack[3])
        delta = decrypt_and_decompress(raw, round_id=round_id, client_id=cid, shared_secret=secret)
        return (global_params.cpu() + delta).to(global_params.device)
    return raw.to(global_params.device)


# ----------------------------------------------------------------------------
# Main federated loop with pluggable defense
# ----------------------------------------------------------------------------
def run_attack_fl(dataset, defense="fedavg", attack=False, attacker_id=3,
                  rounds=None, seed=42, secure_compress=False, adaef=False,
                  tap_noise=False, dp_epsilon=8.0, target_sparsity=0.9, num_bits=8,
                  victim_pct=0.15, attack_type="slow_poisoning", attacked_clients=None,
                  eval_every=1, eval_victims=None,
                  tag="", save=True, verbose=True):
    from keeper.federated.run import load_config, setup_models_and_utils, get_device
    cfg = load_config(dataset)
    device = get_device()
    DGTEN, read_graph, setup_features, calculate_auc, get_snapshot_index, parameter_parser = setup_models_and_utils()
    from keeper.third_party.fedlab.partitioned_HetroEdges import PartitioningCyberData

    if attacked_clients is None:
        attacked = [attacker_id] if attack else []
    else:
        attacked = attacked_clients if attack else []
    args = build_args(cfg, dataset, parameter_parser, seed=seed, rounds=rounds, attack=attack,
                      attacked_clients=attacked, secure_compress=secure_compress, adaef=adaef,
                      tap_noise=tap_noise, dp_epsilon=dp_epsilon, target_sparsity=target_sparsity,
                      num_bits=num_bits, victim_pct=victim_pct, attack_type=attack_type)
    n_rounds = args.com_round

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    gs = PartitioningCyberData(args=args, wrkr=args.total_client)
    args.num_nodes = gs.graph["ncount"]
    args.max_snapshots_inaClient = gs.maax

    graph_data = read_graph(args)
    index_list = get_snapshot_index(args.time_slots, data_path=args.data_path)
    X = torch.from_numpy(setup_features(graph_data, seed=seed, dim=args.layers[-1])).to(device)
    all_labels = graph_data["labels"][:, 1].astype(int)
    class_counts = np.bincount(all_labels)
    class_weight = torch.FloatTensor(1.0 / class_counts * len(all_labels)).to(device)

    model = DGTEN(device, args, X, num_labels=2, index_list=index_list, class_weight=class_weight)
    trainer = make_trainer(model, cfg, args, device, attacked)
    trainer.setup_dataset(gs)
    trainer.set_experimental_setting(args.task)

    # Oracle victim probe (evaluation only): trust toward the attacker's actual
    # injected victims; this is the honest ASR metric.
    # eval_victims (if given) fixes the victim set so clean and attacked runs are
    # measured against the SAME nodes; otherwise victims are auto-identified.
    vprobe = None
    victims = eval_victims
    if victims is None and (attack and attacked):
        victims = identify_victims(gs, attacked, attack_type, victim_pct, seed, device)
    if victims:
        vprobe = build_victim_probe(graph_data, index_list, victims, gs.maax)

    # Keep the broadcast vector on CPU, matching serialize_model(cpu=True) so the
    # client's SPQSA delta (current_params - model_parameters) stays same-device.
    global_params = SerializationTool.serialize_model(model).clone().cpu()

    best = []
    round_log = []
    t_train = time.time()
    for rnd in range(n_rounds):
        t0 = time.time()
        trainer._current_round = rnd
        trainer.local_process([global_params], list(range(args.total_client)))
        packs = trainer.uplink_package          # list of per-client packs
        # order packs by client id
        packs = sorted(packs, key=lambda p: int(p[3]))
        params = [_reconstruct(p, global_params, rnd, args.shared_secret) for p in packs]
        ids = [int(p[3]) for p in packs]

        # ---- aggregate per chosen defense ----
        if defense in ("fedavg", "nodefense"):
            new_global = agg_fedavg(params)
        elif defense == "median":
            new_global = agg_median(params)
        elif defense == "trimmed":
            new_global = agg_trimmed(params, trim=1)
        elif defense == "geomedian":
            new_global = agg_geomedian(params)
        elif defense == "krum":
            new_global, _ = agg_krum(params, n_malicious=1)
        elif defense == "normclip":
            # detection-free per-client norm clipping about the median model
            P = torch.stack(params)
            center = P.median(dim=0).values
            dev = P - center
            dn = dev.norm(dim=1, keepdim=True).clamp(min=1e-8)
            med_norm = dev.norm(dim=1).median()
            scale = torch.clamp(med_norm / dn, max=1.0)
            new_global = (center + dev * scale).mean(dim=0)
        else:
            raise ValueError(f"unknown defense {defense}")

        global_params = new_global.cpu()    # keep CPU-resident for SPQSA device match
        SerializationTool.deserialize_model(model, global_params)
        v_trust = victim_trust(model, vprobe, device) if vprobe is not None else float("nan")
        # Full MCC evaluation dominates cost; evaluate every eval_every rounds
        # (and always in the last 12 rounds so tail metrics are exact).
        do_eval = (rnd % eval_every == 0) or (rnd >= n_rounds - 12) or (rnd == n_rounds - 1)
        if do_eval:
            metrics = trainer.evaluate_global_model(global_params)
        else:
            metrics = np.full(14, np.nan)
        rt = time.time() - t0
        best.append([float(x) for x in metrics] + [rt])
        round_log.append({"round": rnd, "victim_trust": float(v_trust),
                          "T1_MCC": float(metrics[0]), "T2_MCC": float(metrics[7])})
        if verbose and (rnd % 5 == 0 or rnd == n_rounds - 1):
            print(f"  [{defense}{'+atk' if attack else ''}] R{rnd:02d} "
                  f"T1={metrics[0]:.3f} T2={metrics[7]:.3f} ({rt:.1f}s)", flush=True)

    wall = time.time() - t_train
    arr = np.array(best, dtype=np.float64)
    keys = ["MCC", "AUC", "BAcc", "AP", "F1_Micro", "F1_Macro"]
    mean_t1 = {k: round(float(np.nanmean(arr[:, i])), 4) for i, k in enumerate(keys)}
    std_t1 = {k: round(float(np.nanstd(arr[:, i])), 4) for i, k in enumerate(keys)}
    mean_t2 = {k: round(float(np.nanmean(arr[:, 7 + i])), 4) for i, k in enumerate(keys)}
    std_t2 = {k: round(float(np.nanstd(arr[:, 7 + i])), 4) for i, k in enumerate(keys)}
    # Oracle victim-trust (targeted ASR): tail mean over the last 10 rounds
    vt_series = [r.get("victim_trust", float("nan")) for r in round_log]
    vt_tail = float(np.nanmean(vt_series[-10:])) if len(vt_series) >= 1 else float("nan")

    result = {
        "mode": "fl_attack", "dataset": dataset, "defense": defense, "attack": attack,
        "attacker_id": attacker_id, "seed": seed, "rounds": n_rounds,
        "secure_compress": secure_compress, "adaef": adaef, "tap_noise": tap_noise,
        "dp_epsilon": dp_epsilon if secure_compress else None,
        "mean_task1": mean_t1, "std_task1": std_t1,
        "mean_task2": mean_t2, "std_task2": std_t2,
        "victim_trust_tail": round(vt_tail, 4),
        "n_victims": (vprobe["n_victims_present"] if vprobe else 0),
        "wall_total_s": round(wall, 1),
        "round_log": round_log,
        "per_round": [[round(v, 6) for v in row] for row in best],
    }
    if save:
        outdir = os.path.join(PROJECT, "claude_results")
        os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        atk = "atk" if attack else "clean"
        tg = f"_{tag}" if tag else ""
        sc = "_sc" if secure_compress else ""
        out = os.path.join(outdir, f"{dataset}_{defense}_{atk}{sc}_s{seed}{tg}_{ts}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        result["_path"] = out
        if verbose:
            print(f"  saved: {out}")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bitcoinalpha")
    p.add_argument("--defense", default="fedavg")
    p.add_argument("--attack", action="store_true")
    p.add_argument("--attacker", type=int, default=3)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    r = run_attack_fl(a.dataset, defense=a.defense, attack=a.attack,
                      attacker_id=a.attacker, rounds=a.rounds, seed=a.seed)
    print("T1_MCC", r["mean_task1"]["MCC"], "T2_MCC", r["mean_task2"]["MCC"])
