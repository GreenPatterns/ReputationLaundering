"""E2 driver (Task 4.5; docs/preregistration.md, Sections 2.1 and 2.4): one run per (network, seed, stream, model).

  python -m keeper.scem.e2_driver --ds DS --seed S --stream {clean,lag1,lag3} --model MODEL --epochs 150 \
         --prereg docs/preregistration.md --out ABS.json [--force]

A run loads the network's training graph (snapshots 0..tt-1), draws the seed's plan (keeper.scem.e2_streams: targets,
t_s and both lags' witnesses, from the clean graph and its causal capped tiers), builds the stream (the clean twin, or
the attack stream of the lag), trains one ungated model on it for --epochs epochs and scores the 30 targets.

Models. s3r: the default model of keeper.laundering.core (LocalGDTE with S3R). scem*: the archived LocalGDTE with SCEM
(keeper.backbone.local_model_scem) through stage_b.scem_localgdte(stage_b.configured_scem(...)), with M-B on: the
witness weights are witness_cred.level_weights of the run's stream levels (causal capped tiers under the calibration
frozen on the clean stream). Variants: scem (phi learned), scem_phi05 (phi fixed at 0.5, upgrade G5), scem_nohaz
(hazard off: lambda = 0, the protocol's lambda_max = 0) and scem_n1 (N = 1). Every model is built through
core.make_model with --seed, the same for the clean twin and the attack runs, so a paired difference is the attack's
effect. SCEM's constructor swallows unknown keywords, so the built module's settings are checked against SCEM_MODELS
(phi_fixed, N, the hazard switch, horizon_A) and recorded.

Scores. The proxy of a target is keeper.laundering.core.Panel.reputation from the final embedding, with the panel
built on the clean training graph (n_probe 256, seed 1234). Its percentile is the share of the clean graph's active
accounts (the panel's population) whose proxy, under the same model, is strictly below it
(keeper.analysis.cold_start.platform_percentiles). The test metrics are core.test_metrics on the network's held-out
snapshot. The attack gain, (proxy under attack - proxy on the clean twin) x direction, needs two runs and is computed
by keeper.analysis.e2_analysis.

Output: one JSON through core.dump (atomic; an existing file is replaced only with --force), with the plan, the
model settings and learned phi, the M-B levels' digest, tier counts and calibration, the scores, the determinism
warnings raised while building, training and embedding (also printed to stderr), the protocol's sha256 (--prereg) and
the provenance with the git state.
"""
import argparse
import hashlib
import os
import sys
import time
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from keeper.analysis.cold_start import platform_percentiles  # noqa: E402
from keeper.analysis.provenance import git_state  # noqa: E402
from keeper.cluster import protocol_settings as ps  # noqa: E402
from keeper.cppr.protocol import preregistration  # noqa: E402
from keeper.credibility.tiers import KEEPER_EDGE_CAP_UNITS, KEEPER_NEG_FROM, TIER_WEIGHTS  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.scem import e2_streams as E  # noqa: E402
from keeper.scem import scem as SC  # noqa: E402
from keeper.scem import stage_b  # noqa: E402
from keeper.scem.witness_cred import clean_calibration, level_weights  # noqa: E402

STREAM_LAGS = {"clean": None, "lag1": 1, "lag3": 3}
# model -> (options of stage_b.configured_scem, the settings the built module must show). horizon_A must equal
# erlang_horizon_A(N, horizon_phi): phi_fixed, or phi_min = N / 16 at the module defaults r_ratio = 4 and L_f = 1.
SCEM_MODELS = {
    "scem": ({}, dict(N=2, phi_fixed=None, hazard=True, horizon_phi=0.125)),
    "scem_phi05": ({"phi_fixed": 0.5}, dict(N=2, phi_fixed=0.5, hazard=True, horizon_phi=0.5)),
    "scem_nohaz": ({"hazard": False}, dict(N=2, phi_fixed=None, hazard=False, horizon_phi=0.125)),
    "scem_n1": ({"N_stages": 1}, dict(N=1, phi_fixed=None, hazard=True, horizon_phi=0.0625)),
}
assert set(STREAM_LAGS) == set(ps.E2_STREAMS) and set(SCEM_MODELS) | {"s3r"} == set(ps.E2_MODELS)


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def scem_settings(module) -> dict:
    """The settings of a built SCEM module that E2 varies, as recorded."""
    return dict(N=int(module.N), phi_fixed=module.phi_fixed, hazard=bool(module.hazard),
                horizon_A=float(module.horizon_A), lambda_max=float(module.lambda_max),
                phi_min=float(module.phi_min), phi_max=float(module.phi_max))


def check_settings(model: str, settings: dict) -> dict:
    """settings, after checking them against SCEM_MODELS[model]; RuntimeError on a mismatch (a swallowed option)."""
    want = SCEM_MODELS[model][1]
    expected = dict(N=want["N"], phi_fixed=want["phi_fixed"], hazard=want["hazard"],
                    horizon_A=SC.erlang_horizon_A(want["N"], want["horizon_phi"]))
    wrong = {k: (settings[k], v) for k, v in expected.items() if settings[k] != v}
    if wrong:
        raise RuntimeError(f"{model}: the built SCEM module has (got, expected) {wrong}")
    return settings


def build_model(D, Tr, seed: int, model: str, witness_weights=None) -> tuple:
    """(model, its checked SCEM settings or None for s3r), built through core.make_model with `seed`."""
    if model == "s3r":
        return C.make_model(D, Tr, seed, D.tt), None
    pipeline_class = C.LocalGDTE
    try:
        C.LocalGDTE = stage_b.scem_localgdte(
            stage_b.configured_scem(witness_weights=witness_weights, **SCEM_MODELS[model][0]))
        m = C.make_model(D, Tr, seed, D.tt)
    finally:
        C.LocalGDTE = pipeline_class
    return m, check_settings(model, scem_settings(m.scem))


def mb_record(levels: np.ndarray, clean_levels: np.ndarray, Tr_clean, eps: float, plan, lag) -> dict:
    """M-B's inputs as recorded: rule, calibration (frozen on the clean stream), the levels' digest and per-row tier
    counts; on an attack stream also the witnesses' levels at their rating rows and the count of honest cells whose
    level differs from the clean stream's."""
    cal = clean_calibration(*E.signed(Tr_clean), clean_levels.shape[1], eps, T=levels.shape[0])
    out = dict(eps=float(eps), edge_cap_units=KEEPER_EDGE_CAP_UNITS, neg_from=KEEPER_NEG_FROM, t_seed=cal.t_seed,
               T=int(cal.T), tier_weights=list(TIER_WEIGHTS),
               calibration=dict(K=list(cal.K), theta0=list(cal.theta0), n_ref=list(cal.n_ref)),
               levels_sha256=sha256_array(levels),
               tier_counts=[np.bincount(row, minlength=len(TIER_WEIGHTS)).tolist() for row in levels])
    if lag is not None:
        rows = (plan.t_s + lag)[:, None, None]
        out.update(witness_levels=levels[rows, plan.witnesses[lag]].tolist(),
                   honest_levels_changed=int((levels[:, :clean_levels.shape[1]] != clean_levels).sum()))
    return out


def score(m, z, Tr_clean, plan) -> dict:
    """Proxies and percentiles of the plan's targets, on the panel of the clean training graph."""
    panel = C.Panel(Tr_clean)
    proxy = panel.reputation(m, z, plan.targets)
    population = panel.reputation(m, z, panel.active)
    return dict(proxy=proxy.tolist(), percentile=platform_percentiles(proxy, population).tolist(),
                panel=dict(n_probe=int(len(panel.sources)), sources_sha256=sha256_array(panel.sources),
                           n_population=int(len(panel.active))))


def run(ds: str, seed: int, stream: str, model: str, epochs: int, out: str, prereg: str, force: bool = False) -> dict:
    """One E2 run; writes its JSON and returns the record."""
    t0 = time.time()
    lag = STREAM_LAGS[stream]
    if model not in ps.E2_MODELS:
        raise ValueError(f"model must be one of {ps.E2_MODELS}, got {model!r}")
    path = C.result_path(out, force)              # refuse an existing result before the work, not after it
    prereg_info = preregistration(prereg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(ds)
    mem = C.apply_memory_profile(D)
    G = C.full_graph(D)
    Tr = C.train_split(G, D.tt)
    eps = ps.EPS_CAPPED[ds]
    clean_levels = E.stream_levels(Tr, Tr, eps, D.tt)
    plan = E.draw(Tr, clean_levels, seed)
    Tr_run, meta = (Tr, dict(lag=None, n_nodes=int(Tr.n), n_rows=int(len(Tr.edges)), n_injected=0)) \
        if lag is None else E.attack_stream(Tr, plan, lag)
    mb, weights = None, None
    if model != "s3r":
        levels = clean_levels if lag is None else E.stream_levels(Tr_run, Tr, eps, D.tt)
        weights = level_weights(levels)
        mb = mb_record(levels, clean_levels, Tr, eps, plan, lag)
    with warnings.catch_warnings(record=True) as caught:      # protocol 2.1: log determinism warnings
        warnings.simplefilter("always")
        m, settings = build_model(D, Tr_run, seed, model, weights)
        t1 = time.time()
        m = C.train(m, Tr_run, D.tt, epochs=epochs)
        train_s = time.time() - t1
        z = C.embed(m, Tr_run, D.tt)
    determinism = sorted({str(w.message) for w in caught if "determinis" in str(w.message)})
    for message in determinism:
        print(f"determinism warning: {message}", file=sys.stderr)
    test = C.test_metrics(m, z, D, G)
    record = dict(
        experiment="E2", ds=ds, seed=seed, model_seed=seed, stream=stream, lag=lag, model=model, epochs=epochs,
        temporal_encoder="S3R" if model == "s3r" else "SCEM", settings=settings,
        learned_phi=None if settings is None else float(m.scem.phi),
        plan=dict(plan.to_dict(), eps=float(eps), edge_cap_units=KEEPER_EDGE_CAP_UNITS, neg_from=KEEPER_NEG_FROM,
                  clean_levels_sha256=sha256_array(clean_levels)),
        stream_info=meta, targets=plan.targets.tolist(), kind=plan.kinds, direction=plan.direction.tolist(),
        **score(m, z, Tr, plan), test_metrics=test, test_mcc=test["t1"]["mcc"], mb=mb, memory_profile=mem,
        determinism_warnings=determinism, prereg=prereg_info, train_s=round(train_s, 1),
        runtime_s=round(time.time() - t0, 1))
    C.dump(dict(record, provenance=dict(C.provenance(D), **git_state())), path, D, force=force)
    print(f"[E2 {ds} s={seed} {stream} {model}] test MCC {record['test_mcc']} learned phi {record['learned_phi']} "
          f"({record['runtime_s']} s); wrote {path}")
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m keeper.scem.e2_driver", description="One E2 run (Task 4.5).")
    ap.add_argument("--ds", required=True, choices=ps.NETWORKS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--stream", required=True, choices=ps.E2_STREAMS)
    ap.add_argument("--model", required=True, choices=ps.E2_MODELS)
    ap.add_argument("--epochs", type=int, default=ps.EPOCHS)
    ap.add_argument("--prereg", required=True, help="the protocol file; its sha256 is recorded")
    ap.add_argument("--out", required=True, help="the result JSON (absolute, or a name under results/laundering/)")
    ap.add_argument("--force", action="store_true", help="replace an existing result (or set KEEPER_OVERWRITE=1)")
    return ap


def main(argv=None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        run(a.ds, a.seed, a.stream, a.model, a.epochs, a.out, a.prereg, a.force)
    except (FileExistsError, FileNotFoundError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
