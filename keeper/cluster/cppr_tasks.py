"""E3 and E4 task lists of the confirmatory CPPR runs (Task 4.2a).

  python -m keeper.cluster.cppr_tasks {e3,e4} --results-root ABS_REPO_ROOT --out TASKS.tsv [--force]

Writes the task file that keeper.cluster.taskpool reads: one task per line, `id<TAB>done_path<TAB>log_path<TAB>command`.
Each command is a template of the registered protocol (docs/preregistration.md, Section 6) filled in with the settings
of Sections 2.2, 2.5 and 2.6. The generator holds no registered value of its own: it reads them from
keeper.cluster.protocol_settings when it runs. The pool runs the commands with its working directory set to a git
worktree fixed at the run commit, so `--prereg` is the protocol's path inside that worktree. The runs write by absolute
path into the results folder of the main repository, `--results-root`, which must not itself be a git worktree.

The id of a task is the stem of its output, and the output is the task's done path. Under --results-root:
  E3, 75 tasks   results/cppr/e3/e3_{btpc|bfs|btpu}_{ds}_s{seed}.json
  E4, 270 tasks  results/cppr/e4/e4_{btpc|bfs|btpu}_{rand|maxb}_{ring|star}_k{S}_g{G}_{ds}_s{seed}.json
where btpc is capped BTP, btpu is uncapped BTP and bfs is the legacy tiers; rand and maxb are random and max-B(u)
endorsers. The log of a task is results/cppr/{e3,e4}/logs/<id>.log. Target dumps go to results/cppr/{e3,e4}/targets
and E3's verdict dumps to results/dumps/e3. The generator creates no folder: the pool creates the output and log
folders, and the driver creates its dump folders and its base-model cache folder.

E3 (Section 2.5). Per network and seed in SEEDS_E3: capped BTP (also `--ungated-baseline` and verdict dumps), BFS (also
verdict dumps), and, on SEEDS_5 only, uncapped BTP. Every run takes `--arms direct,whitewash,badmouth,control` and the
code defaults for the attack edges (g = 0). No E3 run takes a base-model cache: no two of them share a base model.

E4 (Section 2.6). Per network and seed in SEEDS_5, the eight cells of E4_CELLS under BFS and capped BTP, and the two
g = 1 max-B(u) star cells also under uncapped BTP: 18 runs. Every run takes `--arms whitewash --methods cppr` and
`--pool-eps` set to the capped eps of its network. The capped cell g = 1, random endorsers, ring, s = 3 takes
`--methods plain_vpe,cppr` instead, because E5 pairs with it. Cells that differ only in the attack share one base
model per network, seed and tier setting, so every E4 run takes `--base-cache <results-root>/.keeper-cache/base`
(rulings R59 and R61): a folder outside results/ that git ignores.

Order. The networks come largest first by registered n_ref (Wiki-RfA, then OTC, then Alpha), so that the long tasks
start early. Within a network the seeds ascend. Within a seed, E3 runs capped BTP, BFS, then uncapped BTP; E4 follows
E4_CELLS, and each cell runs BFS, capped BTP, then uncapped BTP where it applies.

Exit status: 0 when the file is written; 1 when it cannot be written (it exists and --force is missing, or its folder
is missing); 2 for a usage error, such as a --results-root that is relative or a git worktree. Standard library only.
"""
import argparse
import os
import re
import shlex
import sys
from typing import NamedTuple

from keeper.cluster import protocol_settings as ps
# The registered grid lives in protocol_settings. These names stay importable from here as well, which
# keeper.analysis.cppr_confirm (Task 4.2b) relies on.
from keeper.cluster.protocol_settings import BFS, CAPPED, UNCAPPED, Cell  # noqa: F401
from keeper.cluster.protocol_settings import E4_CELLS, E4_PAIRED_RUN, E4_UNCAPPED_CELLS  # noqa: F401

DRIVER = "keeper.cppr.cp_vpe_gpu"
ENDORSER_TAGS = {"random": "rand", "max_trust": "maxb"}      # the endorser selection as the E4 file names spell it
BASE_CACHE_PARTS = (".keeper-cache", "base")                 # below the results root, outside results/ (R59, R61)
UNSAFE_IN_ROOT = re.compile(r"[\t\r\n]")                     # would break the task file's lines or fields


class TaskRow(NamedTuple):
    """One line of the task file, with the four fields keeper.cluster.taskpool reads."""
    id: str
    done_path: str
    log_path: str
    command: str


def _checked_root(results_root: str) -> str:
    """The results root, normalised. ValueError unless it is absolute, free of tabs and line breaks, and not a git
    worktree: a worktree's .git is a file, and results go to the main repository."""
    if not os.path.isabs(results_root):
        raise ValueError(f"the results root {results_root!r} is not an absolute path")
    if UNSAFE_IN_ROOT.search(results_root):
        raise ValueError(f"the results root {results_root!r} holds a tab or a line break")
    root = os.path.normpath(results_root)
    if os.path.isfile(os.path.join(root, ".git")):
        raise ValueError(f"the results root {results_root!r} is a git worktree; results go to the main repository")
    return root


def _largest_first() -> list:
    """The networks by registered n_ref, largest first (ties keep the order of NETWORKS)."""
    return sorted(ps.NETWORKS, key=lambda ds: -ps.N_REF_EXPECTED[ds])


def _when(condition: bool, *words: str) -> list:
    return list(words) if condition else []


def _quoted(root: str, *parts: str) -> str:
    """A path under the results root, quoted for the shell (a plain path is unchanged)."""
    return shlex.quote(os.path.join(root, *parts))


def _command(ds: str, seed: int, done_path: str, variant: str, extra: list) -> str:
    """The driver command: the common part of Section 6, the tier flags, then the experiment's own flags."""
    tier_words = [word for pair in ps.tier_flags(variant, ds).items() for word in pair]
    words = [ps.PYTHON, "-m", DRIVER, "--ds", ds, "--seed", str(seed), "--epochs", str(ps.EPOCHS),
             "--hold-test-snapshot", "--prereg", ps.PREREG, "--out", shlex.quote(done_path), *tier_words, *extra]
    return " ".join(words)


def _row(root: str, family: str, task_id: str, ds: str, seed: int, variant: str, extra: list) -> TaskRow:
    folder = os.path.join(root, "results", "cppr", family)
    done_path = os.path.join(folder, f"{task_id}.json")
    return TaskRow(task_id, done_path, os.path.join(folder, "logs", f"{task_id}.log"),
                   _command(ds, seed, done_path, variant, extra))


def _e3_row(root: str, ds: str, seed: int, variant: str) -> TaskRow:
    extra = ["--arms", ps.E3_ARMS, "--dump-targets", _quoted(root, "results", "cppr", "e3", "targets"),
             *_when(variant != ps.UNCAPPED, "--dump-verdicts", _quoted(root, "results", "dumps", "e3")),
             *_when(variant == ps.CAPPED, "--ungated-baseline")]
    return _row(root, "e3", f"e3_{variant}_{ds}_s{seed}", ds, seed, variant, extra)


def _e4_row(root: str, ds: str, seed: int, cell: ps.Cell, variant: str) -> TaskRow:
    task_id = f"e4_{variant}_{ENDORSER_TAGS[cell.endorsers]}_{cell.topology}_k{cell.s}_g{cell.g}_{ds}_s{seed}"
    methods = ps.E4_PAIRED_METHODS if (cell, variant) == ps.E4_PAIRED_RUN else ps.E4_METHODS
    extra = ["--pool-eps", str(ps.EPS_CAPPED[ds]), "--arms", ps.E4_ARMS, "--methods", methods,
             "--attack-edges", str(cell.g), "--k-fake", str(cell.s), "--sybil-topology", cell.topology,
             "--endorser-select", cell.endorsers, "--dump-targets", _quoted(root, "results", "cppr", "e4", "targets"),
             "--base-cache", _quoted(root, *BASE_CACHE_PARTS)]
    return _row(root, "e4", task_id, ds, seed, variant, extra)


def _e3_variants(seed: int) -> tuple:
    """The tier settings E3 runs for a seed: uncapped BTP only on SEEDS_5."""
    return (ps.CAPPED, ps.BFS, ps.UNCAPPED) if seed in ps.SEEDS_5 else (ps.CAPPED, ps.BFS)


def _e4_variants(cell: ps.Cell) -> tuple:
    """The tier settings E4 runs for a cell: uncapped BTP only in E4_UNCAPPED_CELLS."""
    return (ps.BFS, ps.CAPPED, ps.UNCAPPED) if cell in ps.E4_UNCAPPED_CELLS else (ps.BFS, ps.CAPPED)


def e3_tasks(results_root: str) -> list[TaskRow]:
    """The 75 tasks of E3, in task-file order: capped BTP, BFS and, on SEEDS_5, uncapped BTP per network and seed."""
    root = _checked_root(results_root)
    return [_e3_row(root, ds, seed, variant)
            for ds in _largest_first()
            for seed in ps.SEEDS_E3
            for variant in _e3_variants(seed)]


def e4_tasks(results_root: str) -> list[TaskRow]:
    """The 270 tasks of E4, in task-file order: the 18 runs of the grid per network and seed."""
    root = _checked_root(results_root)
    return [_e4_row(root, ds, seed, cell, variant)
            for ds in _largest_first()
            for seed in ps.SEEDS_5
            for cell in ps.E4_CELLS
            for variant in _e4_variants(cell)]


def render(rows: list[TaskRow]) -> str:
    """The text of the task file: one tab-separated line per task."""
    return "".join("\t".join(row) + "\n" for row in rows)


def write_task_file(path: str, text: str, force: bool) -> None:
    """Write the task file. Without force the file is created exclusively, so an existing one raises
    FileExistsError and is left as it was."""
    with open(path, "w" if force else "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


GENERATORS = {"e3": e3_tasks, "e4": e4_tasks}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.cppr_tasks",
                                 description="Write the task file of the E3 or E4 CPPR runs, for "
                                             "keeper.cluster.taskpool.")
    ap.add_argument("experiment", choices=sorted(GENERATORS), help="e3 (75 tasks) or e4 (270 tasks)")
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository, not of a git worktree: every output, log and dump "
                         "goes under its results/, and the E4 base-model cache under its .keeper-cache/")
    ap.add_argument("--out", required=True, metavar="TASKS.tsv", help="the task file to write")
    ap.add_argument("--force", action="store_true", help="replace an existing task file")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        rows = GENERATORS[args.experiment](args.results_root)
    except ValueError as error:
        ap.error(str(error))
    try:
        write_task_file(args.out, render(rows), args.force)
    except FileExistsError:
        print(f"cppr_tasks: {args.out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"cppr_tasks: cannot write {args.out}: {error.strerror}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} tasks ({args.experiment.upper()}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
