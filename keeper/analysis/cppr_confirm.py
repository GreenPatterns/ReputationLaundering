"""E3/E4 confirmatory analysis of the CPPR runs (Task 4.2b): H1, H2w, H2b, H3a and H3b, utility and sensitivities.

Run CPU only, with the main repository as the results root (keeper.cluster.cppr_tasks refuses a git worktree):
  CUDA_VISIBLE_DEVICES="" python -m keeper.analysis.cppr_confirm --results-root /path/to/main/repo --force

Inputs, read-only: the expected E3 and E4 runs and their target dumps (keeper.analysis.cppr_runs), the registered
settings (keeper.cluster.protocol_settings) and the protocol file docs/preregistration.md. The endpoints and statistics
are keeper.analysis.cppr_endpoints'.

Integrity: every run must pass keeper.analysis.cppr_runs' checks (each violated field is listed), and every E4 target
must have the r0 of the E3 run with the same seed and tier setting (exact equality). A run that fails is excluded and
counted, never used; any failure, or a missing protocol file, exits 2. A missing run (result or dump) is listed, never
imputed, and is not a failure: the analysis runs on any partial run set, and a component with results for fewer than
80% of its registered seeds is not evaluable.

H3a is an exact property: in every capped-BTP run used, arm and group, the fakes in Tier >= 1 are at most
guarantee_per_group = floor(K g edge_cap_units). A violation is a code defect and exits 3 (2 takes precedence).

Outputs: results/analysis/cppr_confirm.json, written through keeper.laundering.core.dump (the overwrite guard of
ruling R27; atomic), and the generated summary results/analysis/cppr_confirm.md (keeper.analysis.cppr_confirm_md). Both
are written in every case; an existing output is refused without --force before any work. Holm's correction over the
eight hypotheses is not applied here: the hypothesis-level p-values feed the decisions step. Exit status: 0, 2 or 3 as
above; argparse's usage errors also exit 2.
"""
import argparse
import collections
import datetime
import json
import os
import platform
import socket
import sys

import numpy as np

from keeper.analysis import cppr_endpoints as E
from keeper.analysis import cppr_runs as R
from keeper.analysis.bootstrap import ALPHA, N_BOOT
from keeper.analysis.intervals import BOOT_SEED, PROPORTION, Boot
from keeper.analysis.provenance import git_state
from keeper.cluster import protocol_settings as PS
from keeper.cluster.cppr_tasks import ENDORSER_TAGS
from keeper.laundering import core as C

H3B_METHODS = (PS.BFS, PS.CAPPED)           # endpoint = BFS's contribution minus capped BTP's
HYPOTHESES = ("H1", "H2w", "H2b", "H3b")
KINDS = ("edge", "trusted")                 # edge-level targets and bad-mouth victims
EXIT_OK, EXIT_INTEGRITY, EXIT_CODE_DEFECT = 0, 2, 3
SCHEMA = "keeper.analysis.cppr_confirm/2"
ANALYSIS = os.path.join(C.ROOT, "results", "analysis")
MODULE_FILES = ("keeper/analysis/cppr_confirm.py", "keeper/analysis/cppr_runs.py", "keeper/analysis/cppr_endpoints.py",
                "keeper/analysis/cppr_confirm_md.py", "keeper/analysis/bootstrap.py", "keeper/analysis/intervals.py",
                "keeper/cluster/protocol_settings.py", "keeper/cluster/cppr_tasks.py", "keeper/cppr/dumps.py",
                "keeper/cppr/vpe_frontier.py")
CLAIMS = dict(H1="CPPR removes direct laundering", H2w="CPPR confines whitewash fakes without attack edges",
              H2b="CPPR confines bad-mouth fakes without attack edges",
              H3b="Under one attack edge, the attacker's best whitewash gain is smaller against capped BTP than BFS")
RULES = dict(H1="(a) interval > 0 and (b) cut lower bound >= 0.8, on each network",
             H2w="difference interval > 0 and suppression lower bound >= 0.8, on each network",
             H2b="difference interval > 0 and suppression lower bound >= 0.8, on each network",
             H3b="interval > 0 on each network")
G1_LABELS = tuple(R.cell_label(c) for c in PS.E4_CELLS if c.g == 1)
G2_LABELS = tuple(R.cell_label(c) for c in PS.E4_CELLS if c.g == 2)


# ------------------------------------------------------------------------ one network
class Network:
    """The used runs of one network: e3[tier][seed] and e4[(tier, cell label)][seed] -> Run."""

    def __init__(self, ds: str, runs: dict):
        self.ds, self.e3, self.e4 = ds, {tier: {} for tier in R.TIERS}, {}
        for key, run in runs.items():
            if key.ds != ds:
                continue
            if key.family == "e3":
                self.e3[key.tier][key.seed] = run
            else:
                self.e4.setdefault((key.tier, R.cell_label(key.cell)), {})[key.seed] = run

    def indexes(self, tier: str) -> dict:
        return {seed: run.index for seed, run in sorted(self.e3[tier].items())}


def adjusted_by_cell(net: Network) -> tuple:
    """(adjusted, failures, unpaired): adjusted maps (tier, cell label) -> seed -> {target: control-adjusted CPPR
    whitewash gain} for every E4 run whose E3 run (same seed and tier setting) is present and agrees with it; failures
    are the r0 problems (the run then contributes nothing); unpaired lists the E4 runs whose E3 run is not used."""
    adjusted, failures, unpaired = {}, [], []
    for (tier, label), by_seed in sorted(net.e4.items()):
        for seed, run in sorted(by_seed.items()):
            partner = net.e3[tier].get(seed)
            if partner is None:
                unpaired.append(run.key.stem)
                continue
            values, problems = E.adjusted_gains(run.index, partner.index)
            failures += [dict(run=run.key.stem, field="r0 against E3", detail=f"{p} ({partner.key.stem})")
                         for p in problems]
            if values and not problems:
                adjusted.setdefault((tier, label), {})[seed] = values
    return adjusted, failures, unpaired


def replay_check(net: Network, drawn: dict) -> list:
    """The E3 runs whose dumped edge-level targets or bad-mouth victims differ from the replayed draws of their seed."""
    out = []
    for tier, by_seed in net.e3.items():
        for seed, run in sorted(by_seed.items()):
            for kind in KINDS:
                dumped = set().union(*(set(t) for (_, _, k), t in run.index.items() if k == kind))
                if dumped != set(drawn[seed][kind]):
                    out.append(dict(run=run.key.stem, kind=kind, n_dumped=len(dumped),
                                    n_replayed=len(drawn[seed][kind]),
                                    n_common=len(dumped & set(drawn[seed][kind]))))
    return out


def _restricted(component, paired: E.Paired, exclude, *args, **kwargs) -> dict:
    kept, removed = E.restrict(paired, exclude or ())
    out = component(kept, *args, **kwargs)
    if exclude is not None:
        out["n_pairs_removed"] = removed
    return out


def hypotheses_for(net: Network, adjusted: dict, boot: Boot, prefix: str = "", exclude=None,
                   tier: str = PS.CAPPED, confirmatory: bool = True) -> dict:
    """{hypothesis: {component: result}} of one network on the E3 runs of ``tier`` (capped BTP for the confirmatory
    hypotheses) and, for the capped tier, H3b; every cell is restricted to targets outside ``exclude``."""
    by_seed = net.indexes(tier)
    n = len(PS.SEEDS_5 if tier == PS.UNCAPPED else PS.SEEDS_E3)
    h1, h2w, h2b = E.h1_cells(by_seed), E.h2w_cells(by_seed), E.h2b_cells(by_seed)

    def diff(paired, name):
        return _restricted(E.difference_component, paired, exclude, prefix + name, boot, n, confirmatory=confirmatory)

    def ratio(paired, name):
        return _restricted(E.ratio_component, paired, exclude, prefix + name, boot, n, confirmatory=confirmatory)
    out = {"H1": {"a_difference": diff(h1["a"], "H1/a"), "b_cut": ratio(h1["b"], "H1/b")},
           "H2w": {"a_difference": diff(h2w["a"], "H2w/a"), "b_suppression": ratio(h2w["b"], "H2w/b")},
           "H2b": {"a_difference": diff(h2b["a"], "H2b/a"), "b_suppression": ratio(h2b["b"], "H2b/b")}}
    if tier == PS.CAPPED:
        cells = E.method_cell_cells(adjusted, H3B_METHODS, G1_LABELS)
        out["H3b"] = {"difference": _restricted(E.h3b_component, cells, exclude, prefix + "H3b", boot,
                                                len(PS.SEEDS_5), methods=H3B_METHODS, cell_labels=G1_LABELS,
                                                confirmatory=confirmatory)}
    return out


def _mcc(block) -> float | None:
    """Task 1 MCC of a keeper.laundering.core.test_metrics block."""
    t1 = (block or {}).get("t1") or {}
    return None if t1.get("mcc") is None else float(t1["mcc"])


def utility_for(runs_by_seed: dict, boot: Boot, prefix: str, n: int, ungated: bool) -> dict:
    """Test-MCC differences per seed (Task 1, original test snapshot): the direct arm's CPPR minus its Plain Erasure,
    and minus the ungated backbone where trained (capped-BTP runs)."""
    def direct(run, method):
        return ((run.result.get("test_metrics") or {}).get("direct") or {}).get(method)
    pairs = {"cppr_minus_plain": lambda run: (direct(run, "cppr"), direct(run, "plain"))}
    if ungated:
        pairs["cppr_minus_ungated"] = lambda run: (direct(run, "cppr"), run.result.get("ungated_baseline"))
    out = {}
    for name, get in pairs.items():
        cells = {}
        for seed, run in sorted(runs_by_seed.items()):
            a, b = (_mcc(x) for x in get(run))
            if a is not None and b is not None:
                cells[seed] = {"run": [a - b]}
        out[name] = dict(E.difference_component(E.Paired(cells), prefix + name, boot, n, confirmatory=False),
                         unit_of_analysis="seed")
    return out


def secondary_for(net: Network, adjusted: dict, boot: Boot) -> dict:
    """The secondary outcomes of one network (intervals, no decision rule): CPPR-BFS and uncapped CPPR-BTP on the
    H1/H2 endpoints, test-MCC differences, account-level direct laundering, CPPR's overshoot share and the
    attack-minus-control gains, per E3 tier setting."""
    out = dict(tier_variants={}, utility_mcc={}, account_level={}, overshoot_share={}, attack_minus_control={})
    for tier in R.TIERS:
        by_seed, prefix = net.indexes(tier), f"secondary/{tier}/"
        n = len(PS.SEEDS_5 if tier == PS.UNCAPPED else PS.SEEDS_E3)
        if tier != PS.CAPPED:
            out["tier_variants"][tier] = hypotheses_for(net, adjusted, boot, prefix, tier=tier, confirmatory=False)
        out["utility_mcc"][tier] = utility_for(net.e3[tier], boot, prefix + "mcc/", n, tier == PS.CAPPED)
        out["account_level"][tier] = {name: E.difference_component(p, f"{prefix}account/{name}", boot, n,
                                                                   confirmatory=False)
                                      for name, p in E.account_cells(by_seed).items()}
        share, left_out = E.overshoot_cells(by_seed)
        out["overshoot_share"][tier] = dict(E.difference_component(share, prefix + "overshoot", boot, n,
                                                                   confirmatory=False, support=PROPORTION,
                                                                   unit="targets"), n_targets_left_out=left_out)
        out["attack_minus_control"][tier] = {name: E.difference_component(p, f"{prefix}amc/{name}", boot, n,
                                                                          confirmatory=False)
                                             for name, p in E.attack_minus_control_cells(by_seed).items()}
    return out


def admission_by_rank(net: Network) -> dict:
    """Per tier setting and E4 cell, the fakes in Tier >= 1 of each target group by seed and their mean over seeds; with
    max-B(u) endorsers group i holds the accounts ranked (i - 1) g + 1 .. i g by B(u) (ruling R44). Exact counts."""
    out = {}
    for (tier, label), by_seed in sorted(net.e4.items()):
        per_seed, guarantees = {}, []
        for seed, run in sorted(by_seed.items()):
            record = (run.result.get("sybil_admission") or {}).get("whitewash") or {}
            per_seed[seed] = [int(x) for x in record.get("admitted_per_group") or []]
            guarantees.append(record.get("guarantee_per_group"))
        width = max(map(len, per_seed.values()), default=0)
        out.setdefault(tier, {})[label] = dict(
            ranked=label.startswith(ENDORSER_TAGS["max_trust"]), admitted_per_group_by_seed=per_seed,
            mean_admitted_per_group=[float(np.mean([v[i] for v in per_seed.values() if i < len(v)]))
                                     for i in range(width)],
            guarantee_per_group=guarantees[0] if len(set(map(str, guarantees))) == 1 else guarantees)
    return out


def descriptive_for(net: Network, adjusted: dict, boot: Boot) -> dict:
    """E4, descriptive: every cell's mean control-adjusted CPPR whitewash gain per tier setting, BFS minus capped BTP
    per cell (paired by seed and target), the g = 2 best-response difference, and admission by endorser rank."""
    n = len(PS.SEEDS_5)
    cells = {}
    for (tier, label), by_seed in sorted(adjusted.items()):
        paired = E.Paired({s: {t: [v] for t, v in sorted(g.items())} for s, g in sorted(by_seed.items())})
        cells.setdefault(tier, {})[label] = E.difference_component(paired, f"e4/{tier}/{label}", boot, n,
                                                                   confirmatory=False)
    differences = {label: E.difference_component(E.difference_of(adjusted.get((PS.BFS, label), {}),
                                                                  adjusted.get((PS.CAPPED, label), {})),
                                                 f"e4/bfs_minus_btpc/{label}", boot, n, confirmatory=False)
                   for label in G1_LABELS + G2_LABELS}
    g2 = E.h3b_component(E.method_cell_cells(adjusted, H3B_METHODS, G2_LABELS), "e4/best_response_g2", boot, n,
                         methods=H3B_METHODS, cell_labels=G2_LABELS, confirmatory=False)
    return dict(e4_cells=cells, e4_bfs_minus_capped=differences, best_response_g2=g2,
                admission_by_rank=admission_by_rank(net))


# ------------------------------------------------------------------------ analysis
def h3a_block(runs: dict) -> dict:
    """H3a over every used capped-BTP run of E3 (g = 0: no fake may reach Tier >= 1) and E4."""
    n_runs = n_groups = 0
    violations = []
    for key, run in sorted(runs.items(), key=lambda item: item[0].stem):
        if key.tier != PS.CAPPED:
            continue
        groups, found = E.h3a_check(key.stem, run.result.get("sybil_admission") or {}, K=PS.K_EXPECTED[key.ds],
                                    attack_edges=0 if key.cell is None else key.cell.g,
                                    edge_cap_units=PS.EDGE_CAP_UNITS,
                                    arms=("whitewash", "badmouth") if key.family == "e3" else R.E4_ARMS)
        n_runs, n_groups, violations = n_runs + 1, n_groups + groups, violations + found
    return dict(ok=not violations, code_defect=bool(violations), n_runs=n_runs, n_groups=n_groups,
                violations=violations)


def _load_all(keys: list, results_root: str, manifest: R.Manifest, prereg_sha256) -> tuple:
    """(runs that pass every check, missing runs, failures as dict(run, field, detail))."""
    runs, missing, failures = {}, [], []
    for key in keys:
        run, reason, problems = R.load_run(key, results_root, manifest, prereg_sha256)
        if reason is not None:
            missing.append(dict(run=key.stem, family=key.family, reason=reason))
        failures += [dict(run=key.stem, **p) for p in problems]
        if run is not None and not problems:
            runs[key] = run
    return runs, missing, failures


def _pair_checks(runs: dict, networks) -> tuple:
    """(runs without the E4 runs whose r0 differs from their E3 run's, those r0 failures, the unpaired E4 runs)."""
    failures, unpaired = [], []
    for ds in networks:
        _, found, alone = adjusted_by_cell(Network(ds, runs))
        failures, unpaired = failures + found, unpaired + alone
    bad = {f["run"] for f in failures}
    return {k: r for k, r in runs.items() if k.stem not in bad}, failures, unpaired


def integrity_block(keys: list, runs: dict, missing: list, failures: list, unpaired: list) -> dict:
    """What was expected, missing, excluded (with every violated field per run) and used."""
    def count(pool):
        return {family: sum(k.family == family for k in pool) for family in ("e3", "e4")}
    gone = {m["run"] for m in missing}
    excluded = {}
    for f in failures:
        if f["run"] is not None:
            excluded.setdefault(f["run"], set()).add(f["field"])
    commits = collections.Counter(str((r.result.get("provenance") or {}).get("git_commit")) for r in runs.values())
    return dict(ok=not failures, complete=not missing and not excluded, failures=failures,
                excluded_runs={run: sorted(fields) for run, fields in sorted(excluded.items())},
                missing_runs=missing, n_expected=count(keys), n_missing=count([k for k in keys if k.stem in gone]),
                n_excluded=count([k for k in keys if k.stem in excluded]), n_used=count(runs),
                e4_runs_without_e3_control=unpaired, git_commits=dict(sorted(commits.items())))


def _assemble(per_network: dict, with_claims: bool) -> dict:
    out = {}
    for h in HYPOTHESES:
        nets = {ds: dict(components=comps[h], **E.network_summary(comps[h])) for ds, comps in per_network.items()}
        out[h] = (dict(claim=CLAIMS[h], rule=RULES[h], networks=nets, **E.hypothesis_summary(nets, PS.NETWORKS))
                  if with_claims else nets)
    return out


def _nested_summary(per_network: dict) -> dict:
    """Each hypothesis and network read with the nested bootstrap in the headline hull instead of the crossed one."""
    return {h: {ds: E.network_summary({name: c["sensitivity_nested"] for name, c in comps[h].items()})
                for ds, comps in per_network.items()} for h in HYPOTHESES}


def analyse(results_root: str, prereg: str, networks=PS.NETWORKS, n_boot: int = N_BOOT, replay=None,
            manifest: R.Manifest | None = None) -> dict:
    """The whole analysis (no provenance block): integrity, H3a, the hypotheses, sensitivities and secondary
    outcomes. ``replay(ds, seeds, manifest)`` gives each seed's drawn targets (default cppr_runs.seed_targets)."""
    results_root, networks = os.path.abspath(results_root), tuple(networks)
    manifest = R.Manifest(results_root) if manifest is None else manifest
    replay = R.seed_targets if replay is None else replay
    prereg_sha256 = manifest.add(prereg) if os.path.isfile(prereg) else None
    failures = [] if prereg_sha256 else [dict(run=None, field="protocol", detail=f"protocol file {prereg} not found")]
    keys = R.expected_runs(networks)
    R.check_run_set(keys, results_root, networks)
    runs, missing, run_failures = _load_all(keys, results_root, manifest, prereg_sha256)
    runs, r0_failures, unpaired = _pair_checks(runs, networks)
    failures += run_failures + r0_failures
    main, unused, secondary, descriptive, stored, mismatches = {}, {}, {}, {}, {}, []
    for ds in networks:
        net, boot = Network(ds, runs), Boot(ds, n_boot)
        adjusted = adjusted_by_cell(net)[0]
        drawn = replay(ds, list(R.STORED_SEEDS) + list(PS.SEEDS_E3), manifest)
        old = {kind: set().union(*(set(drawn[s][kind]) for s in R.STORED_SEEDS)) for kind in KINDS}
        stored[ds] = {kind: len(old[kind]) for kind in KINDS}
        mismatches += replay_check(net, drawn)
        main[ds] = hypotheses_for(net, adjusted, boot)
        unused[ds] = hypotheses_for(net, adjusted, boot, prefix="unused/", exclude=old["edge"] | old["trusted"])
        secondary[ds] = secondary_for(net, adjusted, boot)
        descriptive[ds] = descriptive_for(net, adjusted, boot)
    h3a = h3a_block(runs)
    return dict(
        schema=SCHEMA,
        status="integrity_failure" if failures else "code_defect" if h3a["violations"] else "ok",
        protocol=dict(path=manifest.key(prereg), sha256=prereg_sha256),
        settings=dict(networks=list(networks), networks_registered=list(PS.NETWORKS), seeds_e3=list(PS.SEEDS_E3),
                      seeds_e3_uncapped=list(PS.SEEDS_5), seeds_e4=list(PS.SEEDS_5),
                      stored_seeds=list(R.STORED_SEEDS), n_boot=int(n_boot), n_boot_registered=N_BOOT, alpha=ALPHA,
                      threshold=E.THRESHOLD, min_seed_share=str(E.MIN_SEED_SHARE), overshoot_cut=E.OVERSHOOT_CUT,
                      boot_seed=BOOT_SEED, h3b_methods=list(H3B_METHODS), h3b_cells=list(G1_LABELS),
                      g2_cells=list(G2_LABELS), neg_from=PS.NEG_FROM),
        integrity=integrity_block(keys, runs, missing, failures, unpaired),
        h3a=h3a,
        hypotheses=_assemble(main, with_claims=True),
        sensitivities=dict(nested=_nested_summary(main),
                           unused_targets=dict(stored_seed_targets=stored, replay_mismatches=mismatches,
                                               hypotheses=_assemble(unused, False))),
        secondary=secondary,
        descriptive=descriptive,
        method=METHOD)


def exit_code(result: dict) -> int:
    """2 on an integrity failure, else 3 on an H3a violation, else 0."""
    if result["integrity"]["failures"]:
        return EXIT_INTEGRITY
    return EXIT_CODE_DEFECT if result["h3a"]["violations"] else EXIT_OK


METHOD = dict(
    inputs="The expected runs are keeper.cluster.cppr_tasks' E3 and E4 task lists. A run is used only when every "
           "recorded setting equals the registered one (keeper.analysis.cppr_runs) and, for E4, when its targets' r0 "
           "equal its E3 run's; a run that fails is excluded, with every violated field listed. A missing run is "
           "listed, never imputed, and its seed simply has no result.",
    endpoints="Unit: the target; pair: the same seed and target in every arm used (a missing pair is dropped and "
              "counted). Gains are the dump's gain = r_after - r0 (keeper.cppr.dumps). H1, H2w and H2b use the "
              "capped-BTP E3 runs; H3b the g = 1 E4 cells under BFS and capped BTP, control-adjusted with the E3 run "
              "of the same seed and tier setting (control arm, kind edge), whose r0 must equal the E4 run's exactly. "
              "See keeper.analysis.cppr_endpoints for each formula.",
    statistics="Estimate: the mean of per-seed means (equal seed weights); ratios: 1 - the ratio of seed-averaged "
               "means. Headline 95% interval: the hull of the crossed bootstrap percentile interval (seeds and target "
               "ids resampled independently, missing cells allowed; n_boot replicates) and the seed-t interval (df = "
               "seeds - 1; linearized for ratios), clipped to the support (signed quantities and ratios are "
               "unbounded). p-value of a difference: max(one-sided seed-t p, (1 + #{replicates <= 0}) / (B + 1)); of "
               "a ratio against 0.8: max(linearized one-sided seed-t p of (ratio - 0.8), (1 + #{replicates < 0.8}) / "
               "(B + 1)); B counts the defined replicates. Hypothesis p-value: the largest component p-value over "
               "components and networks (intersection-union); Holm's correction is left to the decisions step.",
    readings="Difference: met when the interval lies above 0, contradicted when it lies below 0. Threshold (0.8), "
             "read three ways (Section 1): met when the lower bound reaches it, inconclusive when only the upper bound "
             "does ('removal not established'), partial when the upper bound is below it ('reduces', with the "
             "measured share). A threshold component is contradicted only when its interval lies wholly below 0, the "
             "method then doing worse than nothing (ruling R62). A component with results for fewer than 80% of its "
             "registered seeds is not evaluable, and so is its hypothesis.",
    h3b="Per method and seed s, the best-response cell is the g = 1 cell with the largest mean control-adjusted CPPR "
        "whitewash gain over the other seeds (ties: the first cell in E4_CELLS order); s contributes its own mean gain "
        "in that cell; the endpoint is BFS's contribution minus capped BTP's. Each bootstrap replicate recomputes the "
        "cell means on its resampled targets, then the choices: a drawn seed chooses from the drawn seeds that differ "
        "from it, so a seed drawn twice never chooses with its own data, and a replicate that drew one distinct seed "
        "only is left out (no leave-one-seed-out choice exists).",
    sensitivities="Nested: the nested bootstrap (seeds, then each drawn seed's own targets) replaces the crossed one in "
                  "the hull and the p-value. Unused targets: every endpoint without the edge-level targets and the "
                  "bad-mouth victims that the driver draws for the stored seeds 0-4 (ruling R62), replayed in the "
                  "driver's order on the network's training graph without training (never read from stored files); "
                  "the same replay is compared with every E3 run's dump, and a mismatch is listed.",
    secondary="Intervals only, no decision rule: CPPR-BFS and uncapped CPPR-BTP on the H1/H2 endpoints; test MCC "
              "(Task 1, original test snapshot) of the direct arm's CPPR minus Plain Erasure, and minus the ungated "
              "backbone (capped runs); account-level gains (not control-adjusted: the control arm scores no account "
              "target); overshoot share: targets with a positive Plain Erasure gain whose cut exceeds 1.2; "
              "attack-minus-control gains; every E4 cell, BFS minus capped BTP per cell, the g = 2 best response, and "
              "admission by endorser rank (exact counts).")


# ------------------------------------------------------------------------ provenance and CLI
def provenance(manifest: R.Manifest, args) -> dict:
    """Code state, host, versions, the bootstrap settings and the sha256 of every input (manifest) and module."""
    import scipy
    modules = R.Manifest(C.ROOT)
    return dict(**git_state(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                host=socket.gethostname(), argv=list(sys.argv), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), n_boot=int(args.n_boot),
                boot_seed=BOOT_SEED, results_root=os.path.abspath(args.results_root),
                code_sha256=C.CODE_SHA256_AT_IMPORT,
                module_sha256={rel: modules.add(os.path.join(C.ROOT, rel)) for rel in MODULE_FILES},
                inputs_sha256=dict(sorted(manifest.sha256.items())))


def parse(argv=None):
    ap = argparse.ArgumentParser(description="E3/E4 confirmatory analysis of the CPPR runs (CPU only).")
    ap.add_argument("--results-root", default=C.ROOT, help="the main repository, whose results/cppr holds the runs")
    ap.add_argument("--prereg", default=os.path.join(C.ROOT, PS.PREREG), help="the protocol file")
    ap.add_argument("--networks", nargs="+", default=list(PS.NETWORKS), choices=PS.NETWORKS,
                    help="networks to analyse (default: all three; a subset leaves every hypothesis not evaluable)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap replicates (the protocol's 10,000)")
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "cppr_confirm.json"))
    ap.add_argument("--md", default=os.path.join(ANALYSIS, "cppr_confirm.md"))
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    from keeper.analysis.cppr_confirm_md import render_markdown
    a = parse(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)      # refuse before the work, not after it
    md_path = C.result_path(os.path.abspath(a.md), a.force)
    manifest = R.Manifest(a.results_root)
    result = analyse(a.results_root, a.prereg, a.networks, a.n_boot, manifest=manifest)
    result["provenance"] = provenance(manifest, a)
    for path in (out_path, md_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    text = render_markdown(json.loads(json.dumps(C._finite(result), default=C._json_default)))
    C.write_atomic(md_path, lambda f: f.write(text), replace=C.overwrite_allowed(a.force))
    code = exit_code(result)
    for failure in result["integrity"]["failures"]:
        print(f"integrity failure: {failure['run']}: {failure['detail']}", file=sys.stderr)
    for violation in result["h3a"]["violations"]:
        print(f"H3a violation (code defect): {violation['run']} {violation['arm']} group {violation['group']}: "
              f"{violation['reason']}", file=sys.stderr)
    print(f"wrote {out_path} and {md_path} (status {result['status']}, exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
