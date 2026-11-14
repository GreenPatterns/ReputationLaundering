"""E5 task list: the Regime-2 baselines (Task 4.3; protocol docs/preregistration.md Sections 2.7 and 6).

  python -m keeper.cluster.e5_tasks --results-root ABS_REPO_ROOT --out TASKS.tsv [--force]

Writes the task file of keeper.cluster.taskpool as keeper.cluster.cppr_tasks does for E3 and E4: the same line format,
results-root guard, command template (Section 6, settings read from keeper.cluster.protocol_settings when it runs) and
exit status. Per network (largest first) and seed in SEEDS_5, two capped-BTP runs that train the baselines alone
(--methods E5_METHODS) and dump targets and verdicts:
  g = 0  --arms E5_ARMS_G0 and the code defaults for the attack; paired with E3's capped run of the seed.
         id e5_btpc_g0_{ds}_s{seed}
  g = 1  --arms E5_ARMS_G1 and the attack flags, eps and base cache of the E4 run E4_PAIRED_RUN (g = 1, random
         endorsers, ring, s = 3), its partner. id e5_btpc_rand_ring_k3_g1_{ds}_s{seed}
30 tasks. Under --results-root: outputs results/cppr/e5/<id>.json, logs results/cppr/e5/logs/<id>.log, target dumps
results/cppr/e5/targets, verdict dumps results/dumps/e5 (for E6), and the base cache .keeper-cache/base as in E4.
pairs() gives each E5 task with its partner's task row; keeper.analysis.e5_analysis reads it. Standard library only.
"""
import argparse
import sys
from typing import NamedTuple

from keeper.cluster import cppr_tasks as T
from keeper.cluster import protocol_settings as ps


class Pair(NamedTuple):
    """An E5 task, the E3 or E4 task it pairs with, and its network, seed and attack edges per group g."""
    e5: T.TaskRow
    partner: T.TaskRow
    ds: str
    seed: int
    g: int


def _pairs_of(root: str, ds: str, seed: int) -> list:
    dumps = ["--dump-targets", T._quoted(root, "results", "cppr", "e5", "targets"),
             "--dump-verdicts", T._quoted(root, "results", "dumps", "e5")]
    g0 = T._row(root, "e5", f"e5_{ps.CAPPED}_g0_{ds}_s{seed}", ds, seed, ps.CAPPED,
                ["--arms", ps.E5_ARMS_G0, "--methods", ps.E5_METHODS, *dumps])
    cell, label = ps.E4_PAIRED_RUN
    partner = T._e4_row(root, ds, seed, cell, label)
    g1 = T._row(root, "e5", "e5_" + partner.id[len("e4_"):], ds, seed, label,
                ["--pool-eps", str(ps.EPS_CAPPED[ds]), "--arms", ps.E5_ARMS_G1, "--methods", ps.E5_METHODS,
                 "--attack-edges", str(cell.g), "--k-fake", str(cell.s), "--sybil-topology", cell.topology,
                 "--endorser-select", cell.endorsers, *dumps, "--base-cache", T._quoted(root, *T.BASE_CACHE_PARTS)])
    return [Pair(g0, T._e3_row(root, ds, seed, ps.CAPPED), ds, seed, 0), Pair(g1, partner, ds, seed, cell.g)]


def pairs(results_root: str) -> list[Pair]:
    """Every E5 task with its partner, in task-file order: networks largest first, seeds ascending, g = 0 then g = 1."""
    root = T._checked_root(results_root)
    return [pair for ds in T._largest_first() for seed in ps.SEEDS_5 for pair in _pairs_of(root, ds, seed)]


def e5_tasks(results_root: str) -> list[T.TaskRow]:
    """The 30 tasks of E5, in task-file order."""
    return [pair.e5 for pair in pairs(results_root)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.e5_tasks",
                                 description="Write the task file of the E5 runs (Regime-2 baselines), for "
                                             "keeper.cluster.taskpool.")
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository, not of a git worktree")
    ap.add_argument("--out", required=True, metavar="TASKS.tsv", help="the task file to write")
    ap.add_argument("--force", action="store_true", help="replace an existing task file")
    args = ap.parse_args(argv)
    try:
        rows = e5_tasks(args.results_root)
    except ValueError as error:
        ap.error(str(error))
    try:
        T.write_task_file(args.out, T.render(rows), args.force)
    except FileExistsError:
        print(f"e5_tasks: {args.out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"e5_tasks: cannot write {args.out}: {error.strerror}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} tasks (E5) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
