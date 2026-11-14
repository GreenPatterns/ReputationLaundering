"""E7 analysis (Task 4.7; protocol docs/preregistration.md Sections 2.9, 3 and 4): KEEPER combined, descriptive.

  CUDA_VISIBLE_DEVICES="" python -m keeper.analysis.e7_analysis --results-root ABS_REPO_ROOT \
         [--prereg docs/preregistration.md] [--n-boot 10000] [--force]

Inputs, read-only: the runs of keeper.cluster.e7_tasks under <results root>/results/cppr/e7/, the CPPR runs (result and
target dump, keeper.cppr.dumps) and the witness-arm runs (keeper.cppr.witness_arm). A missing run is listed and its
cells are left out, never imputed.

Integrity (exit 2, nothing written; every failure is listed). Each run: the protocol's sha256 is --prereg's; a git
commit is recorded and no tracked file was modified (CPPR runs: no untracked path either); the registered settings:
epochs, seed, capped BTP with the network's eps, K, n_ref, KEEPER's cap and guard, the rating-level backbone and its
SCEM settings (keeper.scem.e2_driver.check_settings), and its M-B eps, cap and guard. CPPR runs: rho, granularity,
k_fake, gate norm, the arms and methods of protocol_settings.E7_CPPR, the held test snapshot, the attack (g, random
endorsers, ring) and its reference calibration; the target dump belongs to its result and has one finite row per arm,
method, kind and target. Witness runs: stream, g, M-B, k = 5, lag 1, the ring, the endorser stream. Across runs: one
frozen M-B calibration per network; per network and seed, one witness plan whose targets are the CPPR runs' direct edge
targets, and the same whitewash snap_f and r0 in the g = 0 and g = 1 CPPR runs (one base model); per network, seed and
M-B setting, one code commit, panel and SCEM settings (a clean twin and its attacks); per attack stream, the same stream
and endorsers with M-B on and off.

Fields per network, each with Section 4's headline interval (keeper.analysis.intervals.Block: hull of the crossed
bootstrap over seeds and target ids with missing cells, and the seed-t interval; unclipped), per-seed values and cells:
  direct_edge, direct_account  gain_<m>: mean r_after - r0 of the direct arm's edge-level (account-level) targets, m in
                               plain (Plain Erasure), vpe, cppr; from the g = 0 run, not control-adjusted
  whitewash_g0, whitewash_g1   gain_<m>: the fakes' gain on the whitewash targets, m in vpe, cppr
  mcc                          mcc_base, mcc_direct_<m>, mcc_whitewash_g<g>_<m>: Task 1 test MCC on the held test
                               snapshot, one value per seed
  witness_g0, witness_g1       gain_mb_on, gain_mb_off: per direct target, proxy under attack minus proxy on the clean
                               twin with the same M-B setting (+ is laundering); off_minus_on: gain_mb_off - gain_mb_on,
                               paired by seed and target
  witness_g<g>_attacked        the same over the direct targets with at least one attacked warning
  mcc_witness                  mcc_<stream>_mb<on|off>
E7 is descriptive: no test and no decision rule (Section 1).

Outputs: <results root>/results/analysis/e7.json (keeper.laundering.core.dump: provenance, overwrite guard) and the
generated e7.md beside it. An existing output is refused without --force before any work. Exit status: 0; 1 when an
output exists without --force; 2 for an integrity failure or a usage error.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

from keeper.analysis import cppr_runs as R
from keeper.analysis.bootstrap import N_BOOT
from keeper.analysis.intervals import BOOT_SEED, Block, Boot
from keeper.analysis.provenance import git_state
from keeper.cluster import e7_tasks as T
from keeper.cluster import protocol_settings as ps
from keeper.cppr import rating_backbone as RB
from keeper.cppr import witness_arm as WA
from keeper.cppr.protocol import ATTACK_RNG_ENTROPY, preregistration
from keeper.laundering import core as C
from keeper.scem.e2_driver import check_settings

SCHEMA = "keeper.analysis.e7_analysis/1"
K_FAKE = 3                                   # the driver's default region size, which E7 takes (Section 2.9: s = 3)
MB_ARMS = ("base", "whitewash")              # the arms whose M-B levels each E7 CPPR run records
GAIN_BLOCKS = (("direct_edge", 0, "direct", "edge", ("plain", "vpe", "cppr")),
               ("direct_account", 0, "direct", "account", ("plain", "vpe", "cppr")),
               ("whitewash_g0", 0, "whitewash", "edge", ("vpe", "cppr")),
               ("whitewash_g1", 1, "whitewash", "edge", ("vpe", "cppr")))
EXIT_OK, EXIT_EXISTS, EXIT_INTEGRITY = 0, 1, 2


def folder(root: str) -> str:
    return os.path.join(os.path.abspath(root), *T.FOLDER)


def _read(path: str):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------ integrity
def _settings_failures(name: str, obj: dict, want: dict) -> list:
    return [f"{name}: {path} is {R.lookup(obj, path)!r}, not the registered {value!r}"
            for path, value in want.items() if not R._same(R.lookup(obj, path), value)]


def _mb_failures(name: str, mb: dict | None, ds: str) -> list:
    registered = dict(eps=ps.EPS_CAPPED[ds], edge_cap_units=ps.EDGE_CAP_UNITS, neg_from=ps.NEG_FROM)
    if not isinstance(mb, dict):
        return [f"{name}: no M-B record"]
    return [f"{name}: M-B {k} is {mb.get(k)!r}, not the registered {v!r}" for k, v in registered.items()
            if mb.get(k) != v]


def _scem_failures(name: str, settings) -> list:
    try:
        check_settings(RB.SCEM_MODEL, settings)
    except (RuntimeError, KeyError, TypeError) as err:
        return [f"{name}: SCEM settings {settings!r} are not the registered model's ({err})"]
    return []


def cppr_settings(ds: str, seed: int, g: int, prereg_sha: str) -> dict:
    """Field path -> the registered value of the E7 CPPR run (ds, seed, g)."""
    arms, methods = ps.E7_CPPR[g]
    want = {"ds": ds, "seed": seed, "epochs": ps.EPOCHS, "rho": R.RHO, "granularity": R.GRANULARITY, "k_fake": K_FAKE,
            "gate_norm": R.GATE_NORM, "config.gate_norm": R.GATE_NORM, "config.arms": arms.split(","),
            "config.methods": methods.split(","), "config.hold_test_snapshot": True, "config.ungated_baseline": False,
            "config.attack.attack_edges": g, "config.attack.endorser_select": "random",
            "config.attack.sybil_topology": "ring", "config.attack.k_fake": K_FAKE,
            "config.backbone.name": ps.E7_BACKBONE, "config.backbone.model": RB.SCEM_MODEL,
            "config.preregistration.sha256": prereg_sha, "provenance.git_dirty": False, "provenance.git_untracked": 0}
    want.update({f"config.tiers.{f}": v for f, v in R.expected_tiers(ps.CAPPED, ds).items()})
    if g:
        want.update({f"config.attack.reference_calibration.{f}": v
                     for f, v in R.calibration(ps.EPS_CAPPED[ds], ps.EDGE_CAP_UNITS, ds).items()})
    else:
        want["config.attack.reference_calibration"] = None
    return want


def cppr_failures(key: tuple, result: dict, dump: dict, prereg_sha: str) -> list:
    """The integrity failures of the CPPR run key = (ds, seed, g) and its target dump."""
    ds, seed, g = key
    name = T.cppr_id(*key)
    out = _settings_failures(name, result, cppr_settings(ds, seed, g, prereg_sha))
    if not R.lookup(result, "provenance.git_commit"):
        out.append(f"{name}: no git commit recorded")
    backbone = R.lookup(result, "config.backbone") or {}
    arms = backbone.get("mb", {}).get("arms", {})
    if set(arms) != set(MB_ARMS):
        out.append(f"{name}: M-B levels recorded for the arms {sorted(arms)}, not {list(MB_ARMS)}")
    for arm, mb in arms.items():
        out += _mb_failures(f"{name} ({arm} arm)", mb, ds)
    out += _scem_failures(name, backbone.get("settings"))
    out += [f"{name}: {f['detail']}" for f in R._dump_failures(SimpleNamespace(stem=name), result, dump)]
    return out + [f"{name}: {f}" for f in R.index_rows(dump)[1]]


def witness_settings(ds: str, seed: int, stream: str, mb: str, prereg_sha: str) -> dict:
    """Field path -> the registered value of the witness-arm run (ds, seed, stream, mb)."""
    g = ps.E7_WITNESS_STREAMS[stream]
    want = {"experiment": "E7", "arm": "witness", "ds": ds, "seed": seed, "model_seed": seed, "stream": stream, "g": g,
            "mb": mb, "epochs": ps.EPOCHS, "backbone": ps.E7_BACKBONE, "model": RB.SCEM_MODEL,
            "gate_norm": WA.GATE_NORM, "prereg.sha256": prereg_sha, "provenance.git_dirty": False,
            "gate.method": "btp", "gate.eps": ps.EPS_CAPPED[ds], "gate.edge_cap_units": ps.EDGE_CAP_UNITS,
            "gate.neg_from": ps.NEG_FROM}
    if g is None:
        want["attack"] = None
    else:
        want.update({"attack.g": g, "attack.k": WA.K_SYBILS, "attack.lag": WA.LAG, "attack.sybil_ring": True,
                     "attack.rng": [ATTACK_RNG_ENTROPY, seed, WA.ARM_INDEX, g]})
    if mb == "off":
        want["mb_info"] = None
    return want


def witness_failures(key: tuple, r: dict, prereg_sha: str) -> list:
    """The integrity failures of the witness-arm run key = (ds, seed, stream, mb)."""
    name = T.witness_id(*key)
    out = _settings_failures(name, r, witness_settings(*key, prereg_sha))
    if not R.lookup(r, "provenance.git_commit"):
        out.append(f"{name}: no git commit recorded")
    if key[3] == "on":
        out += _mb_failures(name, r.get("mb_info"), key[0])
    out += _scem_failures(name, r.get("settings"))
    if r.get("targets") != R.lookup(r, "plan.targets") or len(r.get("proxy") or ()) != len(r.get("targets") or ()):
        out.append(f"{name}: targets, plan and proxies are inconsistent")
    return out


def _same(groups: dict, fn, label: str) -> list:
    """A failure per group whose members disagree on fn(member); members are (name, object)."""
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len({json.dumps(fn(obj), sort_keys=True) for _, obj in members}) > 1:
            out.append(f"{'/'.join(map(str, key))}: {label} differs across {sorted(n for n, _ in members)}")
    return out


def _direct_targets(index: dict) -> list:
    return sorted(index.get(("direct", "cppr", "edge"), {}))


def group_failures(cppr: dict, witness: dict) -> list:
    """The cross-run checks of the module docstring over the loaded runs."""
    calib, plans, twins, attacks = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for (ds, seed, g), (result, _) in cppr.items():
        for arm, mb in result["config"]["backbone"]["mb"]["arms"].items():
            calib[(ds,)].append((f"{T.cppr_id(ds, seed, g)} ({arm})", mb["calibration"]))
    for (ds, seed, stream, mb), r in witness.items():
        name = T.witness_id(ds, seed, stream, mb)
        if r["mb_info"] is not None:
            calib[(ds,)].append((name, r["mb_info"]["calibration"]))
        plans[(ds, seed)].append((name, r["plan"]))
        twins[(ds, seed, mb)].append((name, r))
        if r["attack"] is not None:
            attacks[(ds, seed, stream)].append((name, r))
    out = _same(calib, lambda c: c, "the frozen M-B calibration")
    out += _same(plans, lambda p: p, "the witness plan")
    out += _same(twins, lambda r: [r["provenance"]["git_commit"], r["panel"], r["settings"]],
                 "the code commit, panel or SCEM settings of a clean twin and its attacks")
    out += _same(attacks, lambda r: [r["stream_info"], r["attack"]], "the attacked stream with M-B on and off")
    for (ds, seed), members in plans.items():
        g0 = cppr.get((ds, seed, 0))
        if g0 is not None and sorted(members[0][1]["targets"]) != _direct_targets(g0[1]):  # plan lists draw order
            out.append(f"{ds}/{seed}: the witness targets are not the CPPR runs' direct edge targets")
    for (ds, seed, g), (result, index) in cppr.items():
        other = cppr.get((ds, seed, 1)) if g == 0 else None
        if other is None:
            continue
        where = f"{T.cppr_id(ds, seed, 0)} vs {T.cppr_id(ds, seed, 1)}"
        if result["sybil_admission"]["whitewash"]["snap_f"] != other[0]["sybil_admission"]["whitewash"]["snap_f"]:
            out.append(f"{where}: the whitewash snap_f differ")
        r0 = [{t: row["r0"] for t, row in idx.get(("whitewash", "cppr", "edge"), {}).items()}
              for idx in (index, other[1])]
        if r0[0] != r0[1]:
            out.append(f"{where}: the whitewash targets' r0 differ (two base models)")
    return out


def load(root: str, prereg_sha: str, networks=ps.NETWORKS) -> tuple:
    """(cppr {key: (result, dump index)}, witness {key: record}, missing task ids, integrity failures)."""
    base, cppr, witness, missing, failures = folder(root), {}, {}, [], []
    for key in (k for k in T.cppr_runs() if k[0] in networks):
        name = T.cppr_id(*key)
        paths = [os.path.join(base, f"{name}.json"), os.path.join(base, "targets", f"{name}_targets.json")]
        if not all(os.path.isfile(p) for p in paths):
            missing.append(name)
            continue
        result, dump = (_read(p) for p in paths)
        failures += cppr_failures(key, result, dump, prereg_sha)
        cppr[key] = (result, R.index_rows(dump)[0])
    for key in (k for k in T.witness_runs() if k[0] in networks):
        path = os.path.join(base, f"{T.witness_id(*key)}.json")
        if not os.path.isfile(path):
            missing.append(T.witness_id(*key))
            continue
        witness[key] = _read(path)
        failures += witness_failures(key, witness[key], prereg_sha)
    if not failures:
        failures = group_failures(cppr, witness)
    return cppr, witness, missing, failures


# ------------------------------------------------------------------------ cells
def cppr_cells(cppr: dict, ds: str) -> tuple:
    """(gains, mccs): gains[block][method][seed] = {target: gain}; mccs[field][seed] = [mcc]."""
    gains, mccs = defaultdict(dict), defaultdict(dict)
    for (d, seed, g), (result, index) in sorted(cppr.items()):
        if d != ds:
            continue
        for block, bg, arm, kind, methods in GAIN_BLOCKS:
            for m in methods:
                cell = index.get((arm, m, kind)) if bg == g else None
                if cell:
                    gains[block].setdefault(f"gain_{m}", {})[seed] = {t: row["gain"] for t, row in cell.items()}
        held = result.get("test_metrics", {})
        fields = [("mcc_base", held.get("base"))] if g == 0 else []
        for arm, by_method in held.items():
            if arm != "base":
                fields += [(f"mcc_{arm}{'' if arm == 'direct' else f'_g{g}'}_{m}", tm) for m, tm in by_method.items()]
        for field, tm in fields:
            if tm is not None and tm["t1"]["mcc"] is not None:
                mccs[field][seed] = [tm["t1"]["mcc"]]
    return gains, mccs


def witness_gains(witness: dict, ds: str, seed: int, stream: str, mb: str):
    """{target: proxy under attack - proxy on the clean twin} of one attack run, or None if a run is missing."""
    attack, clean = witness.get((ds, seed, stream, mb)), witness.get((ds, seed, "clean", mb))
    if attack is None or clean is None:
        return None
    return {int(t): float(a - c) for t, a, c in zip(attack["targets"], attack["proxy"], clean["proxy"])}


def witness_cells(witness: dict, ds: str) -> tuple:
    """(gains, mccs): gains[block][field][seed] = {target: value} for witness_g<g>[_attacked]; mccs[field][seed]."""
    gains, mccs = defaultdict(dict), defaultdict(dict)
    for (d, seed, stream, mb), r in sorted(witness.items()):
        if d == ds and r["test_mcc"] is not None:
            mccs[f"mcc_{stream}_mb{mb}"][seed] = [r["test_mcc"]]
    seeds = sorted({k[1] for k in witness if k[0] == ds})
    for stream, g in ((s, g) for s, g in ps.E7_WITNESS_STREAMS.items() if g is not None):
        for seed in seeds:
            by_mb = {mb: witness_gains(witness, ds, seed, stream, mb) for mb in ps.E7_MB}
            attack = next((witness[(ds, seed, stream, mb)] for mb in ps.E7_MB if (ds, seed, stream, mb) in witness),
                          None)
            attacked = set() if attack is None else set(attack["attack"]["attacked_targets"])
            fields = {f"gain_mb_{mb}": cells for mb, cells in by_mb.items() if cells is not None}
            if len(fields) == 2:
                fields["off_minus_on"] = {t: by_mb["off"][t] - by_mb["on"][t] for t in by_mb["on"]}
            for field, cells in fields.items():
                gains[f"witness_g{g}"].setdefault(field, {})[seed] = cells
                kept = {t: v for t, v in cells.items() if t in attacked}
                if kept:
                    gains[f"witness_g{g}_attacked"].setdefault(field, {})[seed] = kept
    return gains, mccs


def _block(name: str, fields: dict, boot: Boot) -> dict:
    block = Block(name, boot)
    for field, groups in fields.items():
        block.stat(field, groups, support=None)
    block.extra = dict(n_cells={f: sum(len(v) for v in g.values()) for f, g in fields.items()},
                       seeds={f: sorted(g) for f, g in fields.items()})
    return block.to_dict()


def analyse(root: str, prereg: dict, networks=ps.NETWORKS, n_boot: int = N_BOOT) -> dict:
    """The E7 analysis; ValueError lists the integrity failures."""
    cppr, witness, missing, failures = load(root, prereg["sha256"], networks)
    if failures:
        raise ValueError("E7 runs fail the integrity checks:\n  " + "\n  ".join(failures))
    out = {}
    for ds in networks:
        boot = Boot(ds, n_boot=n_boot, design="crossed")
        (gains, mccs), (w_gains, w_mccs) = cppr_cells(cppr, ds), witness_cells(witness, ds)
        blocks = {name: _block(name, gains[name], boot) for name, *_ in GAIN_BLOCKS if gains.get(name)}
        blocks.update({name: _block(name, fields, boot) for name, fields in sorted(w_gains.items())})
        for name, fields in (("mcc", mccs), ("mcc_witness", w_mccs)):
            if fields:
                blocks[name] = _block(name, dict(fields), boot)
        out[ds] = blocks
    return dict(schema=SCHEMA, experiment="E7", protocol_section="2.9",
                descriptive="no superiority conclusion is drawn from E7 (Section 1); intervals only", prereg=prereg,
                n_boot=int(n_boot), boot_seed=BOOT_SEED,
                runs=dict(expected=sum(k[0] in networks for k in T.cppr_runs() + T.witness_runs()),
                          used=len(cppr) + len(witness), missing=missing,
                          commits=sorted({r["provenance"]["git_commit"] for r, _ in cppr.values()}
                                         | {r["provenance"]["git_commit"] for r in witness.values()})),
                networks=out, analysis_git=git_state())


# ------------------------------------------------------------------------ report
def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def markdown(report: dict) -> str:
    """e7.md, generated from the report: per network and block, each field's estimate, 95% interval and seeds."""
    runs = report["runs"]
    lines = ["# E7: KEEPER combined (descriptive; generated by keeper.analysis.e7_analysis)", "",
             f"Runs used: {runs['used']}; missing: {len(runs['missing'])}. Protocol sha256 "
             f"`{report['prereg']['sha256']}`; {report['n_boot']} bootstrap replicates. Gains are r_after - r0 (CPPR "
             "arms; whitewash: the fakes' gain), not control-adjusted, and proxy under attack - proxy on the clean "
             "twin with the same M-B setting (witness arm; off_minus_on = M-B off minus M-B on). Intervals: hull of "
             "the crossed bootstrap and the seed-t interval (protocol Section 4). No superiority conclusion is drawn.",
             ""]
    for ds, blocks in report["networks"].items():
        lines += [f"## {ds}", ""]
        for name, block in blocks.items():
            lines += [f"### {name}", "", "| field | estimate | 95% interval | seeds | cells |",
                      "| --- | --- | --- | --- | --- |"]
            for field, ci in block["ci95"].items():
                lines.append(f"| {field} | {_fmt(block[field])} | "
                             f"{'n/a' if ci is None else f'[{_fmt(ci[0])}, {_fmt(ci[1])}]'} | "
                             f"{len(block['seeds'][field])} | {block['n_cells'][field]} |")
            lines.append("")
    lines += ["## Missing runs", "", ", ".join(runs["missing"]) or "none", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.analysis.e7_analysis", description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="the main repository (not a git worktree); reads results/cppr/e7, writes results/analysis")
    ap.add_argument("--prereg", default=ps.PREREG, help="the protocol file whose sha256 every run must carry")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true", help="replace existing outputs")
    args = ap.parse_args(argv)
    out_dir = os.path.join(os.path.abspath(args.results_root), "results", "analysis")
    paths = [os.path.join(out_dir, f"e7.{ext}") for ext in ("json", "md")]
    try:
        for path in paths:
            C.result_path(path, args.force)
        report = analyse(args.results_root, preregistration(args.prereg), n_boot=args.n_boot)
    except FileExistsError as error:
        print(f"e7_analysis: {error}", file=sys.stderr)
        return EXIT_EXISTS
    except (ValueError, FileNotFoundError) as error:
        print(f"e7_analysis: {error}", file=sys.stderr)
        return EXIT_INTEGRITY
    os.makedirs(out_dir, exist_ok=True)
    C.dump(report, paths[0], force=args.force)
    C.write_atomic(paths[1], lambda f: f.write(markdown(report)), replace=C.overwrite_allowed(args.force))
    print(f"wrote {paths[0]} and {paths[1]} ({report['runs']['used']} runs, {len(report['runs']['missing'])} missing)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
