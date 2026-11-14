"""Retention results quoted in the paper, for either credibility-gate normalization
(read-only, no training).

Reads the stored per-seed runs of keeper/cppr/cp_vpe_gpu.py and their aggregates in results/laundering/:
  count gate (the CPPR gate of Eq. (15)):   cpgate_count_<ds>_s<s>.json, cpgate_count_aggregate.json
  per-snapshot gate (the ablation):         cp_vpe_<ds>_s<s>.json,        cp_vpe_gpu_aggregate.json
Every arm of a run (Plain Erasure, Plain VPE, CPPR) is trained with the run's gate.

Prints the stored runs' rows for the count gate (the retention table of the paper before the 2026-11
revision), the numbers quoted for both gates, the SGCN inference-only share, and the rows of the supplement's
gate-ablation table (tab:gate_ablation). With --check it asserts that the latter equal those in
paper/supplement/supplement_final.tex; the main paper's retention table now comes from the confirmatory
runs (results/analysis/e5.json).

Run from the repository root:
    ~/.conda/envs/ps/bin/python -m keeper.paperkit.cppr_gate_values [--check]
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "results", "laundering")
DS3 = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")
NAME = {"bitcoinalpha": "Bitcoin-Alpha", "bitcoinotc": "Bitcoin-OTC", "wiki-rfa": "Wiki-RfA"}
RUNS = {"count": ("cpgate_count_{ds}_s{s}.json", "cpgate_count_aggregate.json"),
        "snapshot": ("cp_vpe_{ds}_s{s}.json", "cp_vpe_gpu_aggregate.json")}


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def gate_values(gate):
    """Per-dataset values of Table tab:cppr_main under one gate normalization."""
    per_seed, agg_name = RUNS[gate]
    agg = load(agg_name)
    out = {}
    for ds in DS3:
        runs = [load(per_seed.format(ds=ds, s=s)) for s in range(5)]
        a = agg[ds]
        dl, ww, bm = a["direct_laundering"], a["inject_then_erase_whitewash"], a["inject_then_erase_badmouth"]
        assert all(dl[k]["n"] == 5 for k in ("gain_plain_edge", "gain_vpe_edge", "gain_cp_vpe_edge"))
        link = {}
        for arm in ("plain_vpe", "cp_vpe"):
            L = [r["direct_laundering"][f"linkage_{arm}"] for r in runs]
            link[arm] = dict(auc={k: float(np.mean([x["auc"][k] for x in L])) for k in L[0]["auc"]},
                             top1=float(np.mean([x["top1"]["combined"] for x in L])),
                             chance=float(np.mean([x["chance_top1"] for x in L])))
        # A cut is 1 - g_def / g_ref with both gains averaged over the five seeds. The per-seed
        # ratios that the aggregates also store are undefined when a seed's reference gain is
        # zero, as for one count-gate seed of the Bitcoin-Alpha whitewash reference.
        def cut(block, ref, defense):
            return 1.0 - block[defense]["mean"] / block[ref]["mean"]

        out[ds] = dict(
            plain=(dl["gain_plain_edge"]["mean"], dl["gain_plain_edge"]["sd"]),
            vpe=(dl["gain_vpe_edge"]["mean"], dl["gain_vpe_edge"]["sd"]),
            cppr=(dl["gain_cp_vpe_edge"]["mean"], dl["gain_cp_vpe_edge"]["sd"]),
            cut_vpe=cut(dl, "gain_plain_edge", "gain_vpe_edge"),
            cut_cppr=cut(dl, "gain_plain_edge", "gain_cp_vpe_edge"),
            d_mcc=dl["d_t1_mcc_cp_vs_plain"]["mean"],
            link=link,
            link_sd=dl["linkage_auc_cp_vpe"]["sd"],
            fake_vpe=(ww["gain_plain_vpe"]["mean"], ww["gain_plain_vpe"]["sd"]),
            fake_cppr=(ww["gain_cp_vpe"]["mean"], ww["gain_cp_vpe"]["sd"]),
            attack_cut=cut(ww, "gain_plain_vpe", "gain_cp_vpe"),
            drop_vpe=(bm["drop_plain_vpe"]["mean"], bm["drop_plain_vpe"]["sd"]),
            drop_cppr=(bm["drop_cp_vpe"]["mean"], bm["drop_cp_vpe"]["sd"]),
            damage_cut=cut(bm, "drop_plain_vpe", "drop_cp_vpe"),
            mcc0=float(np.mean([r["base_t1"]["mcc"] for r in runs])))
    return out


def pm(v):
    return f"${v[0]:+.3f} \\pm {v[1]:.3f}$".replace("+-", "-").replace("$+-", "$-")


def pct(x):
    return f"${100 * x:.1f}\\%$"


def table_rows(vals):
    rows = []
    for ds in DS3:
        v = vals[ds]
        lv, lc = v["link"]["plain_vpe"], v["link"]["cp_vpe"]
        rows.append(f"{NAME[ds]} & Plain Erasure & {pm(v['plain'])} & ref. & ref. & -- & -- & -- & -- & -- & -- \\\\")
        rows.append(f"($75$ evals) & Plain VPE ($P^{{(t)}}$) & {pm(v['vpe'])} & {pct(v['cut_vpe'])} & -- & "
                    f"${lv['auc']['structure']:.3f}\\;/\\;{lv['auc']['combined']:.3f}$ & "
                    f"${100 * lv['top1']:.1f}\\%\\;({100 * lv['chance']:.1f}\\%)$ & {pm(v['fake_vpe'])} & ref. & "
                    f"{pm(v['drop_vpe'])} & ref. \\\\")
        rows.append(f" & \\textbf{{CPPR}} & {pm(v['cppr'])} & {pct(v['cut_cppr'])} & ${v['d_mcc']:+.3f}$ & "
                    f"${lc['auc']['structure']:.3f}\\;/\\;{lc['auc']['combined']:.3f}$ & "
                    f"${100 * lc['top1']:.1f}\\%\\;({100 * lc['chance']:.1f}\\%)$ & {pm(v['fake_cppr'])} & "
                    f"{pct(v['attack_cut'])} & {pm(v['drop_cppr'])} & {pct(v['damage_cut'])} \\\\")
    return rows


def s7_rows(count, snap):
    """Rows of Supplementary Table S7 (tab:gate_ablation): the count gate, then the ablation."""
    rows = []
    for i, ds in enumerate(DS3):
        for j, (gate, v) in enumerate((("Count", count[ds]), ("Snapshot", snap[ds]))):
            rows.append(f"{NAME[ds] if j == 0 else ''} & {gate} & ${v['mcc0']:.3f}$ & {pm(v['plain'])} & "
                        f"{pm(v['vpe'])} & {pm(v['cppr'])} & {pm(v['fake_vpe'])} & {pm(v['fake_cppr'])} & "
                        f"{pct(v['attack_cut'])} & {pm(v['drop_vpe'])} & {pm(v['drop_cppr'])} & "
                        f"{pct(v['damage_cut'])} & ${v['link']['cp_vpe']['auc']['combined']:.3f}$ \\\\")
    return rows


def sgcn_inference_share():
    """SGCN: seed-mean unlink gain over seed-mean exact-retrain gain (direct deletion)."""
    out = {}
    for ds in DS3:
        fs = sorted(glob.glob(os.path.join(RES, f"baselines_{ds}_s[0-9].json")))
        assert len(fs) == 5, ds
        s = [json.load(open(f))["summary"] for f in fs]
        u = np.mean([x["unlink"]["direct"]["all"]["mean"] for x in s])
        e = np.mean([x["exact_retrain"]["direct"]["all"]["mean"] for x in s])
        out[ds] = (float(u), float(e), float(u / e))
    return out


def main():
    count, snap = gate_values("count"), gate_values("snapshot")
    rows = table_rows(count)
    print("Stored runs, count gate (retention table before the 2026-11 revision):")
    print("\n".join(rows))
    for gate, vals in (("count", count), ("snapshot", snap)):
        print(f"\n[{gate} gate]")
        for ds in DS3:
            v = vals[ds]
            lc = v["link"]["cp_vpe"]
            lv = v["link"]["plain_vpe"]
            print(f"  {NAME[ds]:13s} plain {v['plain'][0]:+.3f}  VPE {v['vpe'][0]:+.3f}  CPPR {v['cppr'][0]:+.3f}  "
                  f"cuts {100 * v['cut_vpe']:.1f}%/{100 * v['cut_cppr']:.1f}%  dMCC {v['d_mcc']:+.3f}  "
                  f"fake gain VPE {v['fake_vpe'][0]:+.3f} CPPR {v['fake_cppr'][0]:+.3f}  "
                  f"attack {100 * v['attack_cut']:.1f}%  damage {100 * v['damage_cut']:.1f}%  MCC0 {v['mcc0']:.3f}")
            print(f"  {'':13s} AUC comb VPE {lv['auc']['combined']:.3f}  CPPR {lc['auc']['combined']:.3f} "
                  f"+/- {v['link_sd']:.3f}  CPPR timing {lc['auc']['timing']:.3f}  "
                  f"Acc@1 VPE {100 * lv['top1']:.1f}% ({100 * lv['chance']:.1f}%)  CPPR {100 * lc['top1']:.1f}% "
                  f"({100 * lc['chance']:.1f}%) = {lc['top1'] / lc['chance']:.1f}x chance")
    print("\nMCC0 count minus snapshot:", " / ".join(f"{count[d]['mcc0'] - snap[d]['mcc0']:+.3f}" for d in DS3))
    print("\nSGCN inference-only share of the exact-retraining gain:")
    for ds, (u, e, r) in sgcn_inference_share().items():
        print(f"  {NAME[ds]:13s} unlink {u:+.4f}  retrain {e:+.4f}  share {100 * r:.1f}%")
    supp = s7_rows(count, snap)
    print("\nSupplement, gate-ablation table (tab:gate_ablation) rows:")
    print("\n".join(supp))
    if "--check" in sys.argv:
        for path, want in (("results/paper/tables/gate_ablation.tex", supp),):
            tex = open(os.path.join(REPO, path)).read()
            missing = [r for r in want if r not in tex]
            if missing:
                raise SystemExit(f"STOP: rows not in {path}:\n" + "\n".join(missing))
            print(f"check: all {len(want)} rows match {path}")


if __name__ == "__main__":
    main()
