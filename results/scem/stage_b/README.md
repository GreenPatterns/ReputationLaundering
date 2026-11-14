# Stage B results (2026-10-17): the user's SCEM code

All files here were written on 2026-10-17 by `scem_eval/stageB_eval.py` (md5 d8e1515f686d). Since the
2026-10-26 consolidation, that script is `keeper/scem/stage_b.py`, run as `python -m keeper.scem.stage_b`.
The old paths below name the files as they were when these records were written; the tag
`pre-revision-2026-10-26` keeps them.

## What produced each file

| Files | Code at the time | Runs | Command now |
|---|---|---|---|
| `engine_<ds>.json` | `thecode/scem.py` = `updated/scem.py` (md5 e1959ac9c01f), function `run_scem_streaming_weights_np`, unchanged | CPU, deterministic, no seed | `python -m keeper.scem.stage_b engine --ds <ds>` |
| `trace_symbolic.json` | same file, `verify_symbolic_example()`, unchanged | CPU | `python -m keeper.scem.stage_b trace` |
| `equiv_bitcoinalpha_s0.json` | original SCEM vs `scem_eval/scem_autograd_fix.py`, forward pass only | GPU, seed 0 | `python -m keeper.scem.stage_b equiv --ds bitcoinalpha --seed 0 --original FILE` (FILE: `git show pre-revision-2026-10-26:thecode/scem.py`) |
| `model_<ds>_s<seed>.json`, `model_summary.json` | `updated/archive_old_versions/local_model.py` (md5 7687d835a558) with `SCEM` taken from `scem_eval/scem_autograd_fix.py` (md5 82e93a05c6e3), trained by `laundering.core.train` for 150 epochs | GPU, seeds 0-4; Slurm job 18442 plus the seed-0 smoke run in job 18409 | `python -m keeper.scem.stage_b model --ds <ds> --seed <s>`, then `python -m keeper.scem.stage_b aggregate` |

Where these files live now: `thecode/scem.py` is `keeper/scem/scem.py`; the archived LocalGDTE-SCEM class
is `keeper/backbone/local_model_scem.py`; `laundering.core` is `keeper.laundering.core`.

Every subcommand writes here by default and refuses to replace an existing record unless it gets
`--force` or runs with `KEEPER_OVERWRITE=1`. `--out DIR` writes the records to another folder;
`aggregate` then also reads its model records from `DIR`.

## The SCEM copy

`scem_eval/scem_autograd_fix.py` was `updated/scem.py` with four lines changed (1105-1106, 1112-1113).
In-place `index_add_` on `n_dur`, `m_dur`, `n_closed` and `m_closed` became out-of-place `index_add`. The
user approved this change on 2026-10-17. Without it, training fails at the first backward pass. On
2026-10-27 the four lines were merged into `keeper/scem/scem.py`, the one SCEM engine, and the copy moved
to `attic/scem_eval/legacy/scem_autograd_fix.py`.

The forward pass is bit-identical to the original: same initial parameters, embeddings and auxiliary loss
on Bitcoin-Alpha, seed 0 (`equiv_bitcoinalpha_s0.json`). `tests/test_scem_equivalence.py` checks this on
the CPU against the original file of the tag, and on an NVIDIA RTX A6000 against the stored auxiliary loss
(`python -m pytest -m gpu tests/test_scem_equivalence.py`).

The first equivalence attempt failed because `torch.utils.checkpoint` was not imported (log
`equiv_bitcoinalpha_s0.attempt1-import-error.log`, not kept in this repository). One import line in the
glue fixed it.

## Code hash in `provenance`

The model files record code hash 2c6ab61927c1. At the time the hash covered `laundering/*.py`,
`thecode/*.py` and `configs/*.json`. It differs from the original 7db11a6df9db because the user's CPPR
files were installed for Stage C (`thecode/scem.py`, `laundering/cp_vpe_gpu.py`).
`scem_autograd_fix.py` was not covered by the hash. Since the consolidation the hash covers
`keeper/**/*.py` and `configs/*`, so new records carry hashes over a different file set.

## Definitions

Each metric is labelled `[paper]` or `[proposed]` in the docstring of `keeper/scem/stage_b.py`. The user
accepted the proposed definitions on 2026-10-17. They are still not stated in the paper or in the user's
code.

## Caveats

- The SCEM horizon used by every model run is A = 53.1 snapshots, the `updated/scem.py` default derived
  from phi_min. It is not the paper's 13.28. Training covers 21 snapshots, so no rating reaches its forced
  close inside these models.
- Four models shared one GPU at a time, on a node that was also running other jobs. Training time and
  latency are therefore upper bounds, not clean measurements. Peak memory is per process and unaffected.
- `ordinary_mcc` is Task 1 (both endpoints seen in training) at p(trust) < 0.5, which is the zero-logit
  threshold, from `keeper.laundering.core.test_metrics`.
