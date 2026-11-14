"""E1 task list (Task 4.4): CPU tasks of keeper.scem.e1_runner for keeper.cluster.taskpool.

  python -m keeper.cluster.e1_tasks --results-root ABS_REPO_ROOT --out TASKS.tsv [--single] [--force]

--single writes only the tasks of H5's single-record studies, e1c_single and e1d_h5_single (SINGLE_SHARDS), which
were added after the other E1 tasks had run.

One task per network, study and shard (SHARDS), with the task-file format, the results-root checks and the writer of
keeper.cluster.cppr_tasks. Outputs: <root>/results/scem/e1/<id>.json, logs in <root>/results/scem/e1/logs/<id>.log,
where id is `{study}_{ds}` or `{study}_{ds}_part{i}of{n}`. Networks come largest first, and within a network the
longest studies first.

CPU slots. The pool needs GPU ids; E1 passes dummy ones, one per CPU slot, and every command blanks
CUDA_VISIBLE_DEVICES itself (and pins numpy to one thread), so no task can see a GPU whatever id the pool hands it:
  CUDA_VISIBLE_DEVICES= python -m keeper.cluster.taskpool TASKS.tsv --cwd WORKTREE --slots-per-gpu 1 \\
      --gpus $(seq -s, -f cpu%g 0 29)
The pool must start with CUDA_VISIBLE_DEVICES empty or unset, because it refuses --gpus ids that a non-empty
CUDA_VISIBLE_DEVICES does not list.
"""
import argparse
import os
import shlex
import sys

from keeper.cluster import protocol_settings as ps
from keeper.cluster.cppr_tasks import TaskRow, _checked_root, _largest_first, render, write_task_file

RUNNER = "keeper.scem.e1_runner"
CPU_ENV = "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
# Shards per study and network, for tasks of at most about 20 minutes, from the measured time per unit (one CPU):
# e1b 2 / 3.2 / 16 s over 3500 / 7382 / 1482 units (Alpha / OTC / Wiki-RfA); e1c 1.5 / 3.9 / 53 s over 220 / 386 / 45;
# e1d_h5 15 / 5.6 / 69 s over ~50 / 467 / 46. Order within a network: longest study first.
SHARDS = {
    "e1b": {"wiki-rfa": 24, "bitcoinotc": 24, "bitcoinalpha": 8},
    "e1d_h5": {"wiki-rfa": 3, "bitcoinotc": 3, "bitcoinalpha": 1},
    "e1c": {"wiki-rfa": 3, "bitcoinotc": 2, "bitcoinalpha": 1},
    "e1d": {"wiki-rfa": 1, "bitcoinotc": 1, "bitcoinalpha": 1},
    "e1a": {"wiki-rfa": 1, "bitcoinotc": 1, "bitcoinalpha": 1},
}
# H5's single-record studies: one unit per eligible (lag, record) pair, 234 / 598 / 528 units on 21 snapshots and
# 465 / 1362 / 949 on 88 (Alpha / OTC / Wiki-RfA). A unit costs what a batched unit of e1c / e1d_h5 cost (six engine
# runs on the stream); per unit the larger of the estimate above and the batched run's runtime / units:
# 4.7 / 7.3 / 53 s on 21 snapshots, 15 / 9.6 / 69 s on 88; shards for about 1000 s of units per task.
SINGLE_SHARDS = {
    "e1d_h5_single": {"wiki-rfa": 66, "bitcoinotc": 14, "bitcoinalpha": 7},
    "e1c_single": {"wiki-rfa": 28, "bitcoinotc": 5, "bitcoinalpha": 2},
}


def _row(root: str, study: str, ds: str, index: int, count: int) -> TaskRow:
    task_id = f"{study}_{ds}" if count == 1 else f"{study}_{ds}_part{index}of{count}"
    folder = os.path.join(root, "results", "scem", "e1")
    done_path = os.path.join(folder, f"{task_id}.json")
    command = " ".join([CPU_ENV, ps.PYTHON, "-m", RUNNER, study, "--ds", ds, "--shard", f"{index}/{count}",
                        "--prereg", ps.PREREG, "--out", shlex.quote(done_path)])
    return TaskRow(task_id, done_path, os.path.join(folder, "logs", f"{task_id}.log"), command)


def e1_tasks(results_root: str, table: dict = SHARDS) -> list[TaskRow]:
    root = _checked_root(results_root)
    return [_row(root, study, ds, i, shards[ds])
            for ds in _largest_first() for study, shards in table.items() for i in range(shards[ds])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.e1_tasks", description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository, not of a git worktree")
    ap.add_argument("--out", required=True, metavar="TASKS.tsv", help="the task file to write")
    ap.add_argument("--single", action="store_true", help="only e1c_single and e1d_h5_single (SINGLE_SHARDS)")
    ap.add_argument("--force", action="store_true", help="replace an existing task file")
    args = ap.parse_args(argv)
    try:
        rows = e1_tasks(args.results_root, SINGLE_SHARDS if args.single else SHARDS)
    except ValueError as error:
        ap.error(str(error))
    try:
        write_task_file(args.out, render(rows), args.force)
    except FileExistsError:
        print(f"e1_tasks: {args.out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"e1_tasks: cannot write {args.out}: {error.strerror}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} tasks (E1) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
