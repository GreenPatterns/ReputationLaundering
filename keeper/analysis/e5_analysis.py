"""E5, descriptive: the Regime-2 baselines against CPPR-BTP, Plain VPE and Plain Erasure (Task 4.3; protocol
docs/preregistration.md Sections 2.7, 3 and 4). E5's outcomes are secondary: intervals, no decision rule.

  CUDA_VISIBLE_DEVICES="" python -m keeper.analysis.e5_analysis --results-root ABS_REPO_ROOT [--force]

Inputs, read-only: every pair of keeper.cluster.e5_tasks.pairs, an E5 run and its partner of the same network, seed
and flags (E3's capped-BTP run at g = 0; the E4 run E4_PAIRED_RUN at g = 1), each a result and its target dump
(keeper.cppr.dumps, <output folder>/targets/<id>_targets.json). A pair with a missing file is listed and skipped,
never imputed.

Pairing, checked on every pair before any number: the same tier and attack configuration; per arm and target kind of
the E5 run the same targets with the same r0, exactly; per whitewash group the same snap_f and endorsers. Any mismatch
stops the analysis (exit 2) with every mismatch listed, and nothing is written.

Fields per network. Gain blocks direct_edge and direct_account (the direct arm's edge-level and account-level targets,
g = 0), whitewash_g0 and whitewash_g1 (the whitewash targets, whose gain is the fakes' gain):
  gain_<m>         mean target gain r_after - r0 of method m: the baselines author, singleton and withdrawal (E5 run)
                   and the comparators cppr, vpe and, on the direct arm, plain (partner run)
  <b>_minus_<c>    the paired difference baseline b minus comparator c, same seed and target
MCC blocks mcc_direct, mcc_whitewash_g0 and mcc_whitewash_g1: mcc_<m>, Task 1 test MCC on the held test snapshot (one
value per seed), and <b>_minus_<c>. Gains are not control-adjusted: no baseline has a control run. author's direct gains
are 0 by construction (it is the base model). Each field has Section 4's headline interval (keeper.analysis.intervals
.Block: hull of the crossed bootstrap over seeds and target ids with missing cells, and the seed-t interval), unclipped,
with its per-seed values and cell count. Linkage of the per-rating pseudonym is E6's.

Outputs: <results root>/results/analysis/e5.json (keeper.laundering.core.dump: provenance, overwrite guard) and the
generated e5.md beside it. An existing output is refused without --force before any work. Exit status: 0; 1 when an
output exists without --force; 2 for a pairing failure or a usage error.
"""
import argparse
import json
import os
import sys

from keeper.analysis.bootstrap import N_BOOT
from keeper.analysis.intervals import BOOT_SEED, Block, Boot
from keeper.cluster import e5_tasks as E5
from keeper.cluster import protocol_settings as PS
from keeper.cppr import dumps as DP
from keeper.cppr.protocol import BASELINES
from keeper.laundering import core as C

SCHEMA = "keeper.analysis.e5_analysis/1"
COMPARATORS = ("cppr", "vpe", "plain")
GAIN_BLOCKS = (("direct_edge", 0, "direct", "edge"), ("direct_account", 0, "direct", "account"),
               ("whitewash_g0", 0, "whitewash", "edge"), ("whitewash_g1", 1, "whitewash", "edge"))
MCC_BLOCKS = (("mcc_direct", 0, "direct"), ("mcc_whitewash_g0", 0, "whitewash"), ("mcc_whitewash_g1", 1, "whitewash"))
EXIT_OK, EXIT_EXISTS, EXIT_PAIRING = 0, 1, 2


class PairingError(ValueError):
    """An E5 run and its partner disagree on what Section 2.7 requires to be identical."""


def target_dump_path(row) -> str:
    return os.path.join(os.path.dirname(row.done_path), "targets", f"{row.id}_targets.json")


def load_run(row):
    """(result, {(arm, method, kind): {target: (r0, gain)}}) of a task row, or None when a file is missing."""
    paths = (row.done_path, target_dump_path(row))
    if not all(os.path.isfile(p) for p in paths):
        return None
    with open(paths[0]) as f:
        result = json.load(f)
    with open(paths[1]) as f:
        dump = json.load(f)
    col = {c: i for i, c in enumerate(dump["columns"])}
    index = {}
    for r in dump["rows"]:
        cell = index.setdefault((r[col["arm"]], r[col["method"]], r[col["kind"]]), {})
        cell[int(r[col["target"]])] = (r[col["r0"]], r[col["gain"]])
    return result, index


def _r0(index: dict, arm: str, kind: str) -> tuple:
    """({target: r0} of one arm and kind over every method of a run, [targets whose r0 differs between methods])."""
    r0, clash = {}, []
    for (a, _, k), cell in index.items():
        if (a, k) != (arm, kind):
            continue
        for t, (value, _) in cell.items():
            if r0.setdefault(t, value) != value:
                clash.append(t)
    return r0, clash


def pairing_errors(pair, e5, partner) -> list:
    """Every way the E5 run ``e5`` and its partner ``partner`` (load_run's tuples) break Section 2.7's pairing."""
    (res5, idx5), (resp, idxp) = e5, partner
    where = f"{pair.e5.id} vs {pair.partner.id}"
    errors = [f"{where}: config.{key} differs" for key in ("tiers", "attack")
              if res5["config"][key] != resp["config"][key]]
    for arm, kind in sorted({(a, k) for a, _, k in idx5}):
        mine, clash = _r0(idx5, arm, kind)
        theirs, _ = _r0(idxp, arm, kind)
        errors += [f"{where}: {arm}/{kind} target {t} has two r0 in {pair.e5.id}" for t in clash]
        if set(mine) != set(theirs):
            errors.append(f"{where}: {arm}/{kind} targets differ: {sorted(set(mine) ^ set(theirs))}")
        errors += [f"{where}: {arm}/{kind} target {t}: r0 {mine[t]!r} != {theirs[t]!r}"
                   for t in sorted(set(mine) & set(theirs)) if mine[t] != theirs[t]]
    adm5, admp = res5["sybil_admission"].get("whitewash"), resp["sybil_admission"].get("whitewash")
    for key in ("snap_f", "endorsers"):
        if adm5 is not None and (admp is None or adm5[key] != admp[key]):
            errors.append(f"{where}: whitewash {key} {adm5[key]} != {None if admp is None else admp[key]}")
    return errors


def collect(pairs: list, runs: dict) -> tuple:
    """(gains, mccs) per network: gains[ds][block][method][seed] = {target: gain}; mccs[ds][block][method][seed] =
    [mcc]. ``runs`` maps a task id to load_run's tuple; a baseline is read from the E5 run, a comparator from the
    partner."""
    gains, mccs = {}, {}
    for pair in pairs:
        source = {m: runs[pair.e5.id] if m in BASELINES else runs[pair.partner.id] for m in BASELINES + COMPARATORS}
        for name, _, arm, kind in (b for b in GAIN_BLOCKS if b[1] == pair.g):
            for method, (_, index) in source.items():
                cell = index.get((arm, method, kind))
                if cell:
                    gains.setdefault(pair.ds, {}).setdefault(name, {}).setdefault(method, {})[pair.seed] = {
                        t: gain for t, (_, gain) in cell.items()}
        for name, _, arm in (b for b in MCC_BLOCKS if b[1] == pair.g):
            for method, (result, _) in source.items():
                held = result.get("test_metrics", {}).get(arm, {}).get(method)
                if held is not None and held["t1"]["mcc"] is not None:
                    mccs.setdefault(pair.ds, {}).setdefault(name, {}).setdefault(method, {})[pair.seed] = [
                        held["t1"]["mcc"]]
    return gains, mccs


def _difference(a: dict, b: dict) -> dict:
    """seed -> {target: a - b} over the cells present in both (a missing pair is dropped)."""
    out = {}
    for seed in sorted(set(a) & set(b)):
        if isinstance(a[seed], dict):
            cells = {t: a[seed][t] - b[seed][t] for t in a[seed].keys() & b[seed].keys()}
        else:
            cells = [a[seed][0] - b[seed][0]]
        if cells:
            out[seed] = cells
    return out


def _block(name: str, by_method: dict, boot: Boot, prefix: str) -> dict:
    block = Block(name, boot)
    cells = {}
    fields = [(f"{prefix}_{m}", by_method[m]) for m in BASELINES + COMPARATORS if m in by_method]
    fields += [(f"{b}_minus_{c}", _difference(by_method[b], by_method[c]))
               for b in BASELINES for c in COMPARATORS if b in by_method and c in by_method]
    for field, groups in fields:
        if groups:
            block.stat(field, groups, support=None)
            cells[field] = sum(len(v) for v in groups.values())
    block.extra = dict(n_cells=cells, seeds={f: sorted(g) for f, g in fields if g})
    return block.to_dict()


def analyse(results_root: str, networks=PS.NETWORKS, n_boot: int = N_BOOT) -> dict:
    """The E5 analysis of the pairs of ``networks``; PairingError lists every mismatch."""
    pairs = [p for p in E5.pairs(results_root) if p.ds in networks]
    runs = {row.id: load_run(row) for p in pairs for row in (p.e5, p.partner)}
    missing = [dict(e5=p.e5.id, partner=p.partner.id, missing=[r.id for r in (p.e5, p.partner) if runs[r.id] is None])
               for p in pairs if runs[p.e5.id] is None or runs[p.partner.id] is None]
    present = [p for p in pairs if runs[p.e5.id] is not None and runs[p.partner.id] is not None]
    errors = [e for p in present for e in pairing_errors(p, runs[p.e5.id], runs[p.partner.id])]
    if errors:
        raise PairingError("E5 runs do not pair with their E3/E4 partners (Section 2.7):\n  " + "\n  ".join(errors))
    gains, mccs = collect(present, runs)
    out = {}
    for ds in networks:
        boot = Boot(ds, n_boot=n_boot, design="crossed")
        out[ds] = {name: _block(name, gains[ds][name], boot, "gain") for name, *_ in GAIN_BLOCKS
                   if name in gains.get(ds, {})}
        out[ds].update({name: _block(name, mccs[ds][name], boot, "mcc") for name, *_ in MCC_BLOCKS
                        if name in mccs.get(ds, {})})
    return dict(schema=SCHEMA, protocol_section="2.7", n_boot=n_boot, boot_seed=BOOT_SEED,
                pairs=[dict(e5=p.e5.id, partner=p.partner.id, ds=p.ds, seed=p.seed, g=p.g) for p in present],
                missing=missing, target_columns=list(DP.TARGET_COLUMNS), networks=out)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def markdown(report: dict) -> str:
    """The generated summary: per network and block, each field's point, 95% interval and seeds."""
    lines = ["# E5: Regime-2 baselines (descriptive; generated by keeper.analysis.e5_analysis)", "",
             f"Pairs analysed: {len(report['pairs'])}; missing: {len(report['missing'])}. Gains are r_after - r0 "
             "(whitewash: the fakes' gain), not control-adjusted; author's direct gains are 0 by construction. "
             "Intervals: hull of the crossed bootstrap and the seed-t interval (protocol Section 4).", ""]
    for ds, blocks in report["networks"].items():
        lines += [f"## {ds}", ""]
        for name, block in blocks.items():
            lines += [f"### {name}", "", "| field | estimate | 95% interval | seeds | cells |",
                      "| --- | --- | --- | --- | --- |"]
            for field in block["ci95"]:
                ci = block["ci95"][field]
                lines.append(f"| {field} | {_fmt(block[field])} | "
                             f"{'n/a' if ci is None else f'[{_fmt(ci[0])}, {_fmt(ci[1])}]'} | "
                             f"{len(block['seeds'][field])} | {block['n_cells'][field]} |")
            lines.append("")
    if report["missing"]:
        lines += ["## Missing pairs", ""] + [f"- {m['e5']} / {m['partner']}: missing {', '.join(m['missing'])}"
                                             for m in report["missing"]] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.analysis.e5_analysis", description=__doc__.split("\n")[0])
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository (not a git worktree); reads and writes under results/")
    ap.add_argument("--force", action="store_true", help="replace existing outputs")
    args = ap.parse_args(argv)
    folder = os.path.join(os.path.normpath(args.results_root), "results", "analysis")
    paths = [os.path.join(folder, f"e5.{ext}") for ext in ("json", "md")]
    try:
        for path in paths:
            C.result_path(path, args.force)
        report = analyse(args.results_root)
    except FileExistsError as error:
        print(f"e5_analysis: {error}", file=sys.stderr)
        return EXIT_EXISTS
    except ValueError as error:                       # PairingError, or a results root e5_tasks refuses
        print(f"e5_analysis: {error}", file=sys.stderr)
        return EXIT_PAIRING
    os.makedirs(folder, exist_ok=True)
    C.dump(report, paths[0], force=args.force)
    C.write_atomic(paths[1], lambda f: f.write(markdown(report)), replace=C.overwrite_allowed(args.force))
    print(f"wrote {paths[0]} and {paths[1]} ({len(report['pairs'])} pairs, {len(report['missing'])} missing)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
