# KEEPER: Defending Signed Trust Networks Against Erasure-Based Reputation Laundering

KEEPER shows that erasing a few authentic negative ratings through legitimate deletion requests raises a
distrusted account's reputation in a signed temporal trust network, and it defends against this
laundering with Source-Clocked Escrow Memory (SCEM) and Credibility-Preserving Pseudonymous Retention
(CPPR). This repository holds all code as one installable Python package (`keeper`), the four signed
networks, every stored result, and the LaTeX source of the paper, one self-contained manuscript without a
supplement (`paper/main_scem.pdf`).

Table V of the paper lists the implementation settings. This README maps each paper
result to the command that produces it and to the file that stores it. It quotes no result number: the
numbers are in the files it names.

## Layout

```text
keeper/                  all code, one installable package
├── backbone/            LocalGDTE: signed convolution, S3R temporal encoder, FrugalSSM, diffusion
│                        features, and the archived LocalGDTE-SCEM class (local_model_scem.py)
├── scem/                SCEM: trainable module and fixed-rule streaming engine (scem.py), Stage B
│                        evaluation (stage_b.py), release-rule ablation (release_rule.py), audit (audit.py)
├── cppr/                Plain VPE (vpe.py, vpe_frontier.py), CPPR (cp_vpe_gpu.py), linkage attacker (linkage.py)
├── laundering/          loading and training (core.py, signed.py), unlearning operators U1 to U5 (unlearn.py),
│                        deletion channels (attack.py), laundering probe (probe4.py), operator calibration,
│                        external scorers, incidental churn, request filters, multi-seed aggregation
├── federated/           federated placement experiment (fed_laundering.py) and its harness
├── data/                dataset builders (Wiki-RfA, Epinions-Dated), dataset census, provenance check
├── paperkit/            table and figure generators, and read-only analyses quoted in the paper
├── cluster/             Slurm launchers
├── credibility/         credibility tiers shared by SCEM and CPPR: legacy tiers and budgeted trust
│                        propagation (tiers.py)
├── analysis/            CPU analyses of the stored runs and of the confirmatory experiments E1 to E9a, Holm
│                        decisions (decisions.py), the paper's value map (values.py)
└── third_party/fedlab/  vendored FedLab modules (Apache-2.0; LICENSE and NOTICE inside)
tests/                   CPU tests (pytest) and the GPU regression suite (test_core.py)
configs/                 one configuration per dataset
data/cyberdata/          the four signed networks
results/                 every stored result: laundering/, scem/, cppr/, legacy/, analysis/, paper/ (value maps and
                         the generated evidence tables in paper/tables/) and dumps/ (SHA-256 manifests only;
                         the dump files are git-ignored)
paper/                   main_scem.tex and its PDF, figures/, the IEEE class file and the packaging scripts
docs/                    registered protocol, its deviation log, and the key list of the value map
```

This repository is a snapshot of the authors' development repository, without its history. The stored
result files record the commit of the development repository that produced them and the SHA-256 of the
protocol (`docs/preregistration.md`), which ties each result to the protocol text released here.

## Install

```bash
conda env create -f environment.yml     # Python 3.10.14, PyTorch 2.4.0 (CUDA 12.4), PyG 2.6.1
conda activate reputation-laundering
pip install -e .                        # the keeper package, editable; it declares no further dependencies
```

Without conda, install the CUDA-matched PyTorch and torch-scatter wheels first, as the header of
`requirements.txt` shows, then `pip install -r requirements.txt` and `pip install -e .`.

Run every command below from the repository root. The stored results were produced with the cluster's
conda environment `ps`, whose package versions equal every pin in `environment.yml` and `requirements.txt`
(checked on 2026-10-27); there, `python` below stands for `env -u PYTHONPATH -u VIRTUAL_ENV
~/.conda/envs/ps/bin/python`.

| Component | Version used |
| --- | --- |
| Python | 3.10.14 |
| CUDA / cuDNN | 12.4 / 9.1 |
| PyTorch, torch-scatter, PyTorch Geometric | 2.4.0, 2.1.2 (CUDA 12.4 wheel), 2.6.1 |
| NumPy, SciPy, scikit-learn, NetworkX, pandas | 1.26.4, 1.15.3, 1.6.1, 3.4.2, 2.3.3 |
| matplotlib, seaborn (figures only) | 3.10.0, 0.13.2 |
| einops, easydict, madgrad, texttable, Pillow, nvidia-ml-py, tqdm, PyYAML | see `requirements.txt` |
| pytest (tests only) | 9.0.2 |
| pdfTeX (paper only) | 1.40.29, TeX Live |

**Hardware.** Every model needs a CUDA GPU: `keeper.laundering.core.require_gpu` refuses the CPU unless
`FEDUNL_ALLOW_CPU=1` is set, which only the tests do. The SCEM streaming engine, the tests, and the table
and figure generators run on the CPU. The stored runs used 48 GB NVIDIA L40S and NVIDIA RTX A6000 GPUs
(see [Determinism and provenance](#determinism-and-provenance)).

## Data

`data/cyberdata/` holds the four signed networks in the loader's format. Their sizes are in Table III of
the paper and in `results/legacy/dataset_census.json`; their file digests are in
`results/legacy/provenance_verification.json`.

| Dataset (`--ds`) | Source |
| --- | --- |
| Bitcoin-Alpha (`bitcoinalpha`) | Kumar et al., ICDM 2016 (SNAP `soc-sign-bitcoin-alpha`) |
| Bitcoin-OTC (`bitcoinotc`) | Kumar et al., WSDM 2018 (SNAP `soc-sign-bitcoin-otc`) |
| Wiki-RfA (`wiki-rfa`) | West et al., TACL 2014 (SNAP `wiki-RfA`) |
| Epinions-Dated (`epinions-dated`) | Massa and Avesani, AAAI 2005 (Epinions trust network with timestamps) |

The loader (`keeper/laundering/signed.py`) drops zero ratings and self-loops and keeps only the sign of
each rating. `python -m keeper.data.build_wikirfa --download` rebuilds the Wiki-RfA files from the SNAP
archive. `python -m keeper.data.build_epinions --konect FILE` rebuilds Epinions-Dated from the KONECT
edge list, which carries each rating's sign and timestamp; `--snap` and `--times` instead join two
fallback sources. `python -m keeper.data.provenance --out FILE` re-checks the digests and the Epinions
join. The datasets are redistributed for reproducibility: cite the original sources and check their terms
of use before reusing them.

## Reproduce the paper's results

`<ds>` is one of the four dataset names above and `<s>` a training seed from 0 to 4, the seeds of the stored
runs behind the laundering results; the confirmatory experiments E1 to E9a use seeds 5 to 14. Rows marked CPU
need no GPU; every other experiment needs one. The experiments of seeds 0 to 4 write
`results/laundering/<name>_<ds>_s<s>.json`, except the Stage B commands (`python -m keeper.scem.stage_b`),
which write `results/scem/stage_b/`. The confirmatory experiments write `results/cppr/`, `results/scem/e1/`,
`results/scem/e2/` and `results/laundering/e9a/`.

**Stored results are protected.** A writer refuses to replace an existing result file unless it gets
`--force` or runs with `KEEPER_OVERWRITE=1`, and it checks before it starts its work. Most experiment
CLIs take `--out FILE` (an absolute path, or a name under `results/laundering/`) to write elsewhere;
`python -m keeper.scem.stage_b` takes `--out DIR`.

**Prerequisite: operator calibration**, once per dataset on seed 0. It selects the step sizes and damping
of U3 to U5 on separate calibration targets, and the laundering probe, the request filters and the
federated experiment refuse to start without it:

```bash
python -m keeper.laundering.calibrate_operators --ds <ds> --seed 0   # -> results/laundering/calibration_<ds>_s0.json
```

### Paper items

The laundering study (Section IV: Table III and Fig. 2) comes from the stored runs of seeds 0 to 4 on the
unmodified backbone, and Fig. 4a comes from a graph-only report of the credibility layer. Tables IV and VI,
Figs. 4b, 5 and 6 and the outcome of each hypothesis come from the confirmatory experiments E1 to E9a on
Bitcoin-Alpha, Bitcoin-OTC and Wiki-RfA. The protocol of these experiments is `docs/preregistration.md`: its Section 2 fixes each experiment, and its Section 6 gives the command templates
of the CPPR driver. `docs/preregistration_deviations.md` logs every change made after registration.

**Running a confirmatory experiment.** Each experiment has a task-list generator in `keeper/cluster/`. The
generator writes one task per line: an id, the result file, the log file and one command, built from the
registered settings in `keeper/cluster/protocol_settings.py`. `keeper.cluster.taskpool` runs a task list on
the GPUs of one Slurm allocation, one GPU per task. It skips a task whose result file exists and reruns a
failed task once. E3 serves as the example:

```bash
ROOT=$PWD        # the main repository as an absolute path; a git worktree is refused
python -m keeper.cluster.cppr_tasks e3 --results-root "$ROOT" --out TASKS.tsv
python -m keeper.cluster.taskpool TASKS.tsv --cwd WORKTREE      # inside a Slurm GPU allocation
python -m keeper.analysis.cppr_confirm                          # CPU
# E1 runs on the CPU, so its pool takes dummy GPU ids:
CUDA_VISIBLE_DEVICES= python -m keeper.cluster.taskpool TASKS.tsv --cwd WORKTREE --slots-per-gpu 1 --gpus $(seq -s, -f cpu%g 0 29)
```

`WORKTREE` is a git worktree of this repository fixed at one commit. The pool starts every command there,
and each result records its commit and the SHA-256 of the protocol. The generated commands call the
cluster's interpreter (`env -u PYTHONPATH -u VIRTUAL_ENV ~/.conda/envs/ps/bin/python`) and write by
absolute path under `--results-root`. The analysis of an experiment runs on the CPU and first checks its
runs: the registered settings and the protocol's SHA-256, and for E5 the pairing with the E3 and E4 runs
(each module's docstring lists its checks). It writes `results/analysis/<name>.json` and `<name>.md` and
refuses to replace an existing output without `--force`.

In the table, "the pool" stands for the `keeper.cluster.taskpool` command above, and `<seed>` is a seed from
5 to 14 (E3, and the H6 models of E2) or from 5 to 9 (the other experiments).

| Paper item | Command | Stored results |
| --- | --- | --- |
| Figs. 1 and 3, diagrams without result numbers | `python -m keeper.paperkit.make_diagrams` (Fig. 1 `fig_overview`, Fig. 3 `fig_mechanisms`) | `paper/figures/` |
| Tables I, II and V (notation, adversaries, settings) | none: hand-written in `paper/main_scem.tex` | none |
| Table III, signed networks and cohorts | `python -m keeper.data.dataset_census bitcoinalpha bitcoinotc wiki-rfa epinions-dated --json FILE` (CPU); the Days column is the range of the `time` column of `data/cyberdata/<ds>.csv`, which no result file stores | `results/legacy/dataset_census.json` (`nodes`, `edges`, and `majority_le3` for the cohort) |
| Section IV text and Fig. 2a, the laundering study (seeds 0 to 4) | `python -m keeper.laundering.probe4 --ds <ds> --seed <s>` (Epinions-Dated: add `--no-full-cohort`) and `python -m keeper.laundering.probe_baselines --ds <ds> --seed <s>` (the three networks without Epinions-Dated); then `python -m keeper.analysis.laundering_extra` and `python -m keeper.paperkit.make_glance_table` (both CPU) | `results/laundering/probe4_<ds>_s<s>.json`, `baselines_<ds>_s<s>.json`; `results/analysis/laundering_extra.{json,md}`; `results/paper/tables/tab_laundering_glance.tex`, `results/paper/glance_values.json`. The SGCN row reads `results/legacy/probe4_aggregate.json` (provenance note 5) |
| Section IV text, self-term ablation (E9a; H8) and calibrated cold start (E8) | `python -m keeper.cluster.e9a_tasks --results-root "$ROOT" --out TASKS.tsv` (driver `keeper.laundering.e9a_runner`), the pool, then `python -m keeper.analysis.e9a_e8` | `results/laundering/e9a/e9a_<ds>_s<seed>_{off,on}.json`, `logs/`; `results/dumps/e9a/` (git-ignored, see below); `results/analysis/e9a_e8.{json,md}` |
| Fig. 4a and RQ1 text, credibility layer | `python -m keeper.credibility.tiers --report` (CPU; no model is trained) | `results/analysis/tiers_report.json` |
| RQ1 text, SybilRank comparison (exploratory, post hoc; not registered) | `python -m keeper.analysis.sybilrank_probe` (CPU, about 7 minutes; no model is trained) | `results/analysis/sybilrank_probe.json` (`networks.<ds>.sybilrank.probe.matched.max_admitted`) |
| Figs. 4b and 5 and RQ2 text, release policies, commitment window and fake-witness attack (E1; H4, H5) | `python -m keeper.cluster.e1_tasks --results-root "$ROOT" --out TASKS.tsv`, and a second list with `--single` for the single-record studies of H5 (driver `keeper.scem.e1_runner`, CPU), the pool, then `python -m keeper.analysis.e1_analysis` | `results/scem/e1/<study>_<ds>.json` or `<study>_<ds>_part<i>of<n>.json`, `logs/`; `results/analysis/e1.{json,md}` |
| RQ2 text, escrow end to end (E2; H6a, H6b) | `python -m keeper.cluster.e2_tasks --results-root "$ROOT" --out TASKS.tsv` (driver `keeper.scem.e2_driver`), the pool, then `python -m keeper.analysis.e2_analysis` | `results/scem/e2/e2_<model>_<stream>_<ds>_s<seed>.json`, `logs/`; `results/analysis/e2.{json,md}` |
| Fig. 6a, 6b and RQ3 text, direct laundering and fake accounts without and with attack edges (E3, E4; H1, H2w, H2b, H3a, H3b) | `python -m keeper.cluster.cppr_tasks e3 --results-root "$ROOT" --out TASKS.tsv`, the same with `e4` (driver `keeper.cppr.cp_vpe_gpu`), the pool, then `python -m keeper.analysis.cppr_confirm` | `results/cppr/e3/`, `results/cppr/e4/` (results, `targets/`, `logs/`); `results/analysis/cppr_confirm.{json,md}` |
| Table VI, retention methods (E5) | `python -m keeper.cluster.e5_tasks --results-root "$ROOT" --out TASKS.tsv` (driver `keeper.cppr.cp_vpe_gpu` with `--methods author,singleton,withdrawal`), the pool, then `python -m keeper.analysis.e5_analysis --results-root "$ROOT"`. The analysis pairs each E5 run with the E3 or E4 run of the same seed, which supplies plain erasure, the single pool and CPPR | `results/cppr/e5/` (results, `targets/`, `logs/`); `results/analysis/e5.{json,md}` |
| Fig. 6c and RQ3 text, insider linkage (E6; H7) | `python -m keeper.analysis.e6_linkage` (CPU); it reads the per-verdict dumps that the E3 runs with capped BTP and with the legacy tiers and the E5 runs without attack edges write | `results/analysis/e6.{json,md}`; `results/dumps/e3/`, `results/dumps/e5/` (git-ignored, see below) |
| RQ4 text, KEEPER as one system (E7, descriptive) | `python -m keeper.cluster.e7_tasks --results-root "$ROOT" --out TASKS.tsv` (drivers `keeper.cppr.cp_vpe_gpu` with `--backbone rating_scem`, and `keeper.cppr.witness_arm`), the pool, then `python -m keeper.analysis.e7_analysis --results-root "$ROOT"` | `results/cppr/e7/` (`e7_cppr_*.json`, `e7_witness_*.json`, `targets/`, `logs/`); `results/analysis/e7.{json,md}` |
| Table IV, the registered tests at a glance, and the outcome of each hypothesis | `python -m keeper.analysis.decisions` (CPU): Holm's correction over the eight statistical hypotheses, from the analysis outputs above | `results/analysis/decisions.{json,md}` |
| Figs. 2, 4, 5 and 6 | `python -m keeper.paperkit.make_figures` (CPU). Fig. 2a reads `results/analysis/laundering_extra.json` and the probe records, Fig. 5 reads `e1.json`, and Fig. 6 reads `cppr_confirm.json` and `e6.json`; these panels check the plotted means against the stored values and stop on a mismatch. Fig. 2b to 2d and Fig. 4 plot the numbers of the evidence tables below (`tab_strata_cutoff`, `s_partial_deletion`, `tab_placement`, `s_credibility`, `s_release_policies`, `s_window`), held as constants at the top of the script | `paper/figures/fig_{laundering,streams,witness,retention}.pdf` |
| Numbers quoted only in the text | `python -m keeper.analysis.values` (CPU) writes the value map: one key per value, with its source file and JSON pointer | `results/paper/keeper_values.json`, `docs/keeper_values_keys.md`, `paper/keeper_values.tex` |

The run files of E1 to E9a are tracked with their logs and target dumps. The large dump files under
`results/dumps/` are git-ignored, and only the `MANIFEST_SHA256.txt` of each dump folder is tracked: the
per-verdict dumps of E3 and E5 (`results/dumps/e3/`, `results/dumps/e5/`) and the per-run arrays of E9a
(`results/dumps/e9a/`). `e6_linkage` and `e9a_e8` read these dumps, and `e6_linkage` compares each dump with
its manifest entry. In a fresh clone these two analyses therefore need the dumps first: rerun E3 with capped
BTP and with the legacy tiers, the E5 runs without attack edges, and E9a. The other analyses read tracked
files only. `python -m keeper.analysis.dump_manifest results/dumps/e3 --check` compares a dump folder with
its manifest.

The trained runs of E2, E3, E4, E5 and E7 ran on NVIDIA RTX A6000 GPUs and those of E9a on an NVIDIA L40S;
E1 and E6 run on the CPU. Each run record names its host and, for a GPU run, its GPU. The text and the
tables of the manuscript carry literal numbers: the manuscript inputs no generated file, and no script
compares the LaTeX source with the value map. To trace a number, look up its key in
`docs/keeper_values_keys.md` or its row in the evidence tables below.

### Evidence tables for the numbers quoted in the text

The single manuscript quotes many results in its text and figures instead of printing a table for each. The
generated LaTeX tables behind these numbers are kept in `results/paper/tables/`, outside the manuscript; no
document inputs them. `make_supp_tables` and `make_revision_supp_tables` stand for
`python -m keeper.paperkit.make_supp_tables` and `python -m keeper.paperkit.make_revision_supp_tables` (both
CPU, next subsection). The experiments E1 to E9a are run as the table above shows.

| Evidence | Command | Stored results |
| --- | --- | --- |
| Dataset provenance (file digests) | `python -m keeper.data.provenance --out FILE` (CPU) and the dataset census | `results/legacy/provenance_verification.json`, `results/legacy/dataset_census.json` |
| Per-stratum gains under U1 to U5 (Fig. 2b) | the laundering probe, then `make_supp_tables` | `results/paper/tables/tab_strata_cutoff.tex` |
| Incidental churn (Section IV text) | `python -m keeper.laundering.incidental --ds <ds> --seed <s>`, then `make_supp_tables` | `results/laundering/incidental_<ds>_s<s>.json`; `results/paper/tables/tab_incidental.tex` |
| Defense placement: request filters (Fig. 2d) and federated aggregation (Section IV text) | `python -m keeper.laundering.verification --ds <ds> --seed <s>` and `python -m keeper.federated.fed_laundering --ds <ds> --seed <s>` (Bitcoin-Alpha and Bitcoin-OTC), then `make_supp_tables` | `results/laundering/verification_<ds>_s<s>.json`, `fed_laundering_<ds>_s<s>.json`; `results/paper/tables/tab_placement.tex` |
| Single-pool operating points (not quoted in the paper) | `python -m keeper.cppr.vpe_frontier --ds <ds> --seed <s>` for the three networks without Epinions-Dated, then `make_supp_tables` | `results/laundering/vpe_<ds>_s<s>.json`; `results/paper/tables/tab_vpe_frontier.tex` |
| Credibility layer: calibration, tier shares, demotion census (Fig. 4a, RQ1 text, Table V) | `python -m keeper.credibility.tiers --report` (CPU), then `make_revision_supp_tables` | `results/analysis/tiers_report.json`; `results/paper/tables/s_credibility.tex` |
| Release policies, fake-witness attack, commitment window on 21 and on 88 snapshots, granularity (Figs. 4b and 5, RQ2 text) | E1 and `python -m keeper.analysis.e1_analysis`, then `make_revision_supp_tables` | `results/analysis/e1.json`; `results/paper/tables/s_{release_policies,witness_attack,window,granularity}.tex` |
| Gate normalization in the stored runs (seeds 0 to 4, legacy tiers; RQ3 text) | `python -m keeper.cppr.cp_vpe_gpu --ds <ds> --seed <s> --gate-norm count`, the same with `--gate-norm snapshot`, for the three networks without Epinions-Dated; aggregates: `python -m keeper.cppr.cp_vpe_gpu --aggregate --gate-norm count`, the same with `snapshot`; rows: `python -m keeper.paperkit.cppr_gate_values` (CPU), checked against `results/paper/tables/gate_ablation.tex` with `--check` | `results/laundering/cpgate_count_<ds>_s<s>.json`, `cpgate_count_aggregate.json`; `cp_vpe_<ds>_s<s>.json`, `cp_vpe_gpu_aggregate.json` |
| Protocol and outcomes (Table IV) | hypotheses and rules: `docs/preregistration.md`; outcomes: `python -m keeper.analysis.decisions` (CPU) | `results/analysis/decisions.{json,md}` |
| Yardsticks, partial deletion, percentiles and targets without ratings (Section IV text, Fig. 2a and 2c, Table III) | the laundering probe and `python -m keeper.analysis.laundering_extra` (CPU), then `make_revision_supp_tables` | `results/analysis/laundering_extra.json`; `results/paper/tables/s_{yardsticks,partial_deletion,percentiles}.tex` |
| Self-term ablation (E9a) and calibrated cold start (E8) (Section IV text) | E9a and `python -m keeper.analysis.e9a_e8`, then `make_revision_supp_tables` | `results/analysis/e9a_e8.json`; `results/paper/tables/s_self_term.tex` |
| Stored engine runs: attenuation, order and component ablations (not quoted in the paper) | `python -m keeper.scem.stage_b engine --ds <ds>` for `bitcoinalpha`, `bitcoinotc`, `wiki-rfa`, and `python -m keeper.scem.release_rule`, which prints the content of the stored file (both CPU) | `results/scem/stage_b/engine_<ds>.json` (`engine`, `rolling`); `results/scem/release_rule/release_rule_ablation_exact.txt` |
| Lag census, memory census (overhead paragraph of RQ4) and silent-sender trace | `keeper.scem.scem.measure_dataset_evidence_lags` and `keeper.scem.audit.silent_sender_trace()`, both recomputed by `tests/test_scem_audit.py`; memory fields returned by `keeper.scem.scem.run_scem_streaming_weights_np` (all CPU) | `results/legacy/probe4_scem_aggregate.json` (`scem_verification.dataset_evidence_lags`), `results/legacy/round5_audit_results.json`; no file for the memory census |
| Hazard of a single warning (Section V-B text) | `python -m keeper.analysis.defense_extra` (CPU) | `results/analysis/defense_extra.{json,md}` |
| Workload counts of the stored runs (Section V-C text) | `python -m keeper.paperkit.empty_snapshot_share`, `panel_churn_overlap`, `cppr_workload_facts` (CPU, print only) | none |

`cp_vpe_gpu` defaults to `--gate-norm count`, the paper's CPPR gate of Eq. (12) (ruling G1, 2026-10-27). The
stored `cp_vpe_*` runs relied on the older default, so they are reproduced with `--gate-norm snapshot`; the
stored `cpgate_count_*` runs passed `--gate-norm count`. The help text of `cp_vpe_gpu`, the `gate_weights`
docstring and `keeper/paperkit/cppr_gate_values.py` still use the numbers of older drafts (Eq. (15), Table S7).
`python -m keeper.paperkit.cppr_gate_values --check` prints the six rows of the gate-normalization ablation
and compares them with `results/paper/tables/gate_ablation.tex`.

The LocalGDTE-SCEM training records of seeds 0 to 4 stay in `results/scem/stage_b/`
(`model_<ds>_s<s>.json`, `model_summary.json`; `python -m keeper.scem.stage_b model --ds <ds> --seed <s>`,
then `python -m keeper.scem.stage_b aggregate`), although the paper has no table for them.

### Tables, figures and PDFs from the stored results (CPU)

```bash
python -m keeper.paperkit.make_supp_tables            # results/paper/tables/tab_{strata_cutoff,incidental,placement,vpe_frontier}.tex, results/paper/values.json
python -m keeper.paperkit.make_glance_table           # results/paper/tables/tab_laundering_glance.tex, results/paper/glance_values.json
python -m keeper.paperkit.make_revision_supp_tables   # results/paper/tables/s_*.tex, results/paper/revision_supp_tables.json
python -m keeper.analysis.values                      # results/paper/keeper_values.json, results/paper/keeper_values.tex, docs/keeper_values_keys.md
python -m keeper.paperkit.make_figures                # fig_laundering, fig_streams, fig_witness, fig_retention
python -m keeper.paperkit.make_diagrams               # fig_overview, fig_mechanisms
python paper/package_overleaf.py                      # paper/overleaf/ (the upload folder) and paper/main_scem.pdf
```

These scripts read stored results only and run no experiment. `make_glance_table`,
`make_revision_supp_tables` and `values` take `--check`, which regenerates in memory and exits with status 1
when a file on disk differs. `make_supp_tables --out DIR --values FILE` writes elsewhere.

**Building the PDF.** The manuscript is one file, `paper/main_scem.tex`, with an inline bibliography; it
needs only `paper/IEEEtran.cls` and the six figures in `paper/figures/`. `python paper/package_overleaf.py`
writes these files into the self-contained folder `paper/overleaf/` (git-ignored; this is the folder to
upload to Overleaf), compiles it from its own root and copies the PDF to `paper/main_scem.pdf`. It fails on
a LaTeX error, an undefined reference, an overfull `\hbox`, "??" in the PDF text, or more than 18 pages
(`--max-pages`). `python paper/build_isolated.py` builds the manuscript alone in a temporary folder with the
same checks and prints the list of files it needs. Both scripts take `pdflatex` from the folder named by
`TEXBIN` (default `~/texlive/2024/bin/x86_64-linux`) and read the PDF text with PyMuPDF. Plain `pdflatex`
run twice in `paper/` also works; no bibtex step is needed.

Checked on 2026-11-13 in this checkout: `package_overleaf.py` built the manuscript from `paper/overleaf/` at 18
pages without a LaTeX error, an undefined reference, an overfull `\hbox` or "??", and the three `--check`
commands above passed.

### Cluster launchers

`keeper/cluster/` holds the Slurm launchers of the stored runs of seeds 0 to 4: `run_stageC_bgfix.sh` (all
CPPR runs of these seeds, both gates), `run_cp_vpe_gpu.sh` (per-snapshot gate runs) and
`stageB_full_slurm.sh` (LocalGDTE-SCEM training). The confirmatory experiments use the task pool instead
(see [Main paper](#main-paper)). Each launcher skips runs whose result file exists. Each skips its
aggregate step with a message when the aggregate exists, unless it runs with `KEEPER_OVERWRITE=1`. When
every model record already exists, `stageB_full_slurm.sh` still starts one malformed model run (an `xargs`
call without `-r`), which ends in a usage error and writes a stray log; run it only when model records are
missing.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q                      # CPU suite
CUDA_VISIBLE_DEVICES=0 python -m tests.test_core                 # GPU regression suite (script style)
CUDA_VISIBLE_DEVICES=0 python -m pytest -m gpu tests/test_scem_equivalence.py   # GPU: stored SCEM auxiliary loss
```

`tests/conftest.py` keeps the script-style `tests/test_core.py` out of pytest's collection, because its
checks run at import time and need a GPU. On the CPU, the GPU-marked test of the pytest suite is skipped.
`tests/__init__.py` makes `tests` a regular package, so that `python -m tests.test_core` finds this folder
even when another installed package (bitsandbytes 0.43.3, for example) ships a top-level `tests` package.

**Rerun checks of 2026-10-27.** After the move into `keeper/`, these commands reran stored results on the
GPU model each was produced on, writing to a scratch folder:

```bash
# one laundering-probe target of the same draw (NVIDIA L40S); compare targets[0] with the stored target 3509
python -m keeper.laundering.probe4 --ds bitcoinalpha --seed 0 --only-target 3509 --no-full-cohort --out /tmp/probe4_t3509.json
# one CPPR seed (NVIDIA RTX A6000); compare with results/laundering/cpgate_count_bitcoinalpha_s0.json
python -m keeper.cppr.cp_vpe_gpu --ds bitcoinalpha --seed 0 --epochs 150 --gate-norm count --out /tmp/cpgate_count_bitcoinalpha_s0.json
# the SCEM streaming engine (CPU); compare with results/scem/stage_b/engine_<ds>.json
python -m keeper.scem.stage_b engine --ds <ds> --out /tmp/engine
```

Each rerun matched its stored record. The probe target's record and the CPPR record agreed in every
numeric field exactly, ignoring wall-clock times and provenance. Each engine record was byte-identical
apart from its wall-clock `seconds` fields and the run-specific provenance fields (`host`, `utc`, `argv`
and the code hashes). On an NVIDIA RTX A6000, `python -m pytest -m gpu tests/test_scem_equivalence.py`
also reproduces the stored auxiliary loss of `results/scem/stage_b/equiv_bitcoinalpha_s0.json` exactly.

## Determinism and provenance

- **Seeds.** A training seed (0 to 4 in the stored laundering runs, 5 to 14 in the confirmatory experiments)
  sets the Python, NumPy and PyTorch generators and the Gaussian draw of the diffusion features. In the
  laundering probe, the targets, the calibration targets and the 256-account probe panel use fixed
  generator seeds, so every seed evaluates the same targets.
- **Deterministic mode.** Training runs in PyTorch's deterministic mode with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the scripts set themselves. On the same GPU model a rerun
  reproduces a stored record; results are not bitwise portable across GPU models.
- **Where each result family ran.** Every per-run record names its host, GPU, library versions and code
  hash in `provenance`, and all except the federated records also name the data digests. All GPU records
  use torch 2.4.0, CUDA 12.4 and deterministic mode.

  | Results | GPU | Host | Dates |
  | --- | --- | --- | --- |
  | `probe4`: Bitcoin-Alpha and Bitcoin-OTC (all seeds), Wiki-RfA seeds 0 to 3 | NVIDIA L40S | kc-sse-ml-rn04 | 2026-10-09 to 2026-10-10 |
  | `probe4`: Wiki-RfA seed 4 | NVIDIA RTX A6000 | kc-sse-ml-rn02 | 2026-10-17 |
  | `probe4`: Epinions-Dated, all seeds | NVIDIA RTX A6000 | kc-sse-ml-rn03 (seeds 0 to 3), kc-sse-ml-rn00 (seed 4) | 2026-10-18 to 2026-10-19 |
  | `calibration`, `baselines`, `incidental`, `verification`, `vpe`, `fed_laundering` | NVIDIA L40S | kc-sse-ml-rn04 | 2026-10-09 to 2026-10-11 |
  | `cpgate_count`, `cp_vpe` (CPPR, both gates) | NVIDIA RTX A6000 | kc-sse-ml-rn03 | 2026-10-20 |
  | `stage_b` `model`, `equiv` | NVIDIA RTX A6000 | kc-sse-ml-rn02 | 2026-10-17 |
  | `stage_b` `engine`, `trace` | CPU | kc-sse-ml-rn02 | 2026-10-17 |

- **Runtimes**, from the `runtime_s`, `train_seconds` and `seconds` fields of the stored records:

  | Experiment | Runtime | Hardware |
  | --- | --- | --- |
  | Laundering probe, one dataset and seed, with the full-cohort census | 8 to 18 h | L40S |
  | Laundering probe, one target (`--only-target ... --no-full-cohort`) | about 7 min | L40S |
  | Operator calibration, one dataset | 13 to 15 min (Epinions-Dated about 50 min) | L40S |
  | SGCN and EigenTrust, one seed | 20 to 30 min (Wiki-RfA about 2 h) | L40S |
  | Incidental churn, one seed | 22 to 31 min (Epinions-Dated about 2.7 h) | L40S |
  | Request filters, one seed | 27 min to 3.2 h | L40S |
  | Plain VPE operating points, one seed | 1 to 1.8 h | L40S |
  | Federated placement, one seed | 3 to 4 min | L40S |
  | CPPR, one seed, all arms (three runs sharing each GPU) | 36 to 42 min | A6000 |
  | LocalGDTE-SCEM training, one seed (four runs sharing one GPU) | 6 to 9 min | A6000 |
  | SCEM streaming engine, one configuration | under 7 s | CPU |
  | Release-rule ablation (18 engine runs) | about 1 min | CPU |
  | GPU regression suite, `python -m tests.test_core` | about 11 min | A6000 |

  The Epinions-Dated probe records were resumed after interruptions, so their `runtime_s` covers only the
  last session and is not listed.

### Provenance notes

1. **Code hashes.** Each result records `code_sha256`, a hash over the code tree at run time. Since the
   move into the package (2026-10-26) the hash covers `keeper/**/*.py` and `configs/*`; before, it covered
   `laundering/*.py`, `thecode/*.py` and `configs/*.json`. The hashes in the stored records therefore do
   not match the current tree. The tree before the move is not part of this snapshot.
   `MANIFEST_SHA256.txt` lists the SHA-256 of every tracked file except itself, sorted by path, as of
   2026-11-13 (the end of the November 2026 revision). To regenerate it after a change:
   `git ls-files | grep -vx MANIFEST_SHA256.txt | LC_ALL=C sort | sed 's|^|./|' | xargs -d '\n' sha256sum > MANIFEST_SHA256.txt`.
2. **Result-file labels.** In the laundering result files, `C1` labels the subject-side request for all
   negative in-edges of a target (the paper's C2), not the paper's single deletion. The files also hold
   analyses the paper does not report: a greedy cost curve, a grey-box C3 plan built on a surrogate model,
   and a matched trust-erasure control.
3. **CPPR background-deletion fix (2026-10-20).** Earlier CPPR runs re-used the background deletion
   requests on the graph with injected fake accounts without re-pointing them, so part of those deletions
   missed their intended edges. `build_inject_then_erase_graph` now re-points them by edge identifier
   (test: `tests/test_cp_vpe_inject.py`), and `results/paper/tables/gate_ablation.tex` reports the corrected runs.
   `python -m keeper.paperkit.cppr_workload_facts` prints how many requests still hit their edges. The
   superseded runs stay in `results/laundering/cp_vpe_before_bgfix_2026-10-20/`; their direct-laundering
   arm is identical to that of the corrected runs.
4. **SCEM autograd fix.** Four in-place index additions of the SCEM module became out-of-place, so the
   module can back-propagate; the user approved this on 2026-10-17, and it is merged into
   `keeper/scem/scem.py`. The forward pass is unchanged:
   `results/scem/stage_b/equiv_bitcoinalpha_s0.json` and `tests/test_scem_equivalence.py` (see
   `results/scem/stage_b/README.md`).
5. **Laundering aggregate.** The SGCN gains quoted in Section IV read
   `results/legacy/probe4_aggregate.json`, an aggregate of 2026-10-14 whose writer is not in this
   repository. `python -m keeper.laundering.aggregate4` pools the probe records into
   `results/laundering/probe4_aggregate.json`, which is not stored.
6. **Dataset census.** `results/legacy/dataset_census.json` is what the census command above writes for
   the four networks (identical on 2026-10-27). `results/laundering/dataset_census.json` is a second,
   three-network census without Wiki-RfA. A later revision phase resolves the pair.
7. **Audit.** `keeper.scem.audit` holds the Round-5 audit behind `results/legacy/round5_audit_results.json`.
   Its `main` cannot run end to end: two of its items need `cp_vpe_scem.py`, which this repository does not contain,
   and its first item refuses to overwrite the stored provenance record. The trace and the lag census are
   recomputed from code by `tests/test_scem_audit.py`.
8. **Inputs outside this repository.** `results/paper/values.json` keys its input-file hashes by absolute
   path, and four of its inputs, the calibration files named in the probe records'
   `calibration.source`, were read from the sibling folder `../FedUnlearning/laundering/results/`. Each is
   byte-identical to its copy in `results/laundering/`, which `make_supp_tables` reads when the sibling
   folder is absent. The folder was absent on 2026-11-12, and the regenerated `values.json` then differed
   from the stored file in these path keys only. Likewise, the Epinions artifact block of
   `results/legacy/provenance_verification.json` digests probe partial files that live only in that
   folder, so `python -m keeper.data.provenance` run here records them as missing.
9. **Typeset paths.** Captions of the evidence tables in `results/paper/tables/` still name the pre-package
   locations of some files (`laundering/results/`).

## Citation and license

Cite the paper (citation to be added on acceptance) and the dataset papers listed under [Data](#data).
No license has been chosen yet; until a `LICENSE` file is added, all rights are reserved by the authors.
The vendored FedLab modules keep their Apache-2.0 license (`keeper/third_party/fedlab/LICENSE`).
