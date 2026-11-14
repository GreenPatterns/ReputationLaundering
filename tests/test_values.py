"""Task 4.9: the paper's values map and LaTeX value macros (keeper.analysis.values): formatting, key resolution against
the stored results, partial inputs, the generic flattening, --check and a pdflatex compile of the macros."""
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from keeper.analysis import values as V
from keeper.laundering import core as C

PDFLATEX = os.path.expanduser("~/texlive/2024/bin/x86_64-linux/pdflatex")


@pytest.fixture
def scratch():
    """A temporary folder inside the repository (nothing is written outside it)."""
    with tempfile.TemporaryDirectory(dir=C.ROOT, prefix=".scratch_test_") as folder:
        yield folder


@pytest.fixture(scope="module")
def stored():
    """The values map of this repository's stored results."""
    return V.build(C.ROOT)[0]


def copy(rel, root):
    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
    shutil.copy(os.path.join(C.ROOT, rel), os.path.join(root, rel))


# ------------------------------------------------------------------------ formatting
@pytest.mark.parametrize("value, fmt, text", [
    (0.2681, "signed3", "+0.268"), (-0.19945, "signed3", "-0.199"), (-0.0001, "signed3", "+0.000"),
    (0.0001, "fixed3", "0.000"), (0.9924, "pct1", "99.2"), (0.6482, "pct0", "65"), (-0.4171, "spct1", "-41.7"),
    (0.057712, "p2", "0.058"), (9.999e-05, "p2", "0.00010"), (1.73e-11, "p2", r"1.7\times10^{-11}"), (1.0, "p2", "1.0"),
    (0.0800, "g3", "0.0800"), (12949, "int", "12{,}949"), (3629.0, "int", "3{,}629"), (0.02, "g", "0.02"),
    ("not_supported", "str", r"not\_supported"), (True, "str", "true"), (None, "pct1", "n/a")])
def test_text_formats(value, fmt, text):
    assert V.text_of(value, fmt) == text


def test_json_pointers_round_trip_keys_with_slashes_dots_and_tildes():
    doc = {"a/b": {"x.y": [1, {"~z": 7}]}}
    ptr = V.pointer(["a/b", "x.y", 1, "~z"])
    assert ptr == "/a~1b/x.y/1/~0z" and V.resolve(doc, ptr) == 7


# ------------------------------------------------------------------------ key resolution on the stored results
def test_every_pointer_key_resolves_to_its_source_value(stored):
    docs = {}
    for key, e in stored.items():
        assert V.KEY.match(key)
        if e["kind"] != "pointer":
            continue
        if e["source"] not in docs:
            with open(os.path.join(C.ROOT, e["source"])) as f:
                docs[e["source"]] = json.load(f)
        assert V.resolve(docs[e["source"]], e["path"]) == e["value"], key
        assert e["text"] == V.text_of(e["value"], e["format"])


def test_derived_ranges_are_the_extremes_of_their_members(stored):
    for prefix, pattern, nets in V.DERIVED:
        members = [stored[pattern.format(net=V.NET[ds])]["value"] for ds in nets]
        assert (stored[f"{prefix}.min"]["value"], stored[f"{prefix}.max"]["value"]) == (min(members), max(members))
    assert (stored["range.h8.gain_on.min"]["text"], stored["range.h8.gain_on.max"]["text"]) == ("+0.268", "+0.435")


def test_stored_attack_numbers_and_the_sign_control(stored):
    nets = ("alpha", "otc", "wiki", "epinions")
    for op in ("unlink", "unlink_frozen_x"):
        assert [stored[f"stored.{n}.{op}_share"]["text"] for n in nets] == ["99.2", "100.2", "99.7", "100.3"]
    assert [stored[f"stored.{n}.sign_control.mean"]["text"] for n in nets] == ["-0.131", "-0.168", "-0.265", "-0.142"]
    assert [stored[f"stored.{n}.sign_control.n.s0"]["value"] for n in nets] == [24, 25, 22, 25]
    assert stored["stored.epinions.influence_newton.gain.mean"]["text"] == "n/a"
    assert stored["glance.edgeless.alpha"]["text"] == "60" and stored["glance.percentile.alpha.after"]["text"] == "25.1"   # cohort-weighted since fd63773


def test_h8_and_e8_keys(stored):
    assert [stored[f"h8.alpha.gain_on.{k}"]["text"] for k in ("est", "lo", "hi")] == ["+0.268", "+0.190", "+0.346"]
    assert stored["h8.alpha.ratio.reading"]["value"] == "inconclusive" and stored["h8.p"]["text"] == "0.058"
    assert all(f"e8.wiki.newcomer_change_m{m}.est" in stored for m in (1, 2, 4))


# ------------------------------------------------------------------------ partial inputs and the generic flattening
def test_missing_inputs_give_no_keys_and_no_error(scratch):
    assert V.build(scratch) == ({}, {})
    copy(V.E9A, scratch)
    values, sha = V.build(scratch)
    assert "h8.alpha.gain_on.est" in values and list(sha) == [V.E9A]
    assert not any(k.startswith(("dec.", "glance.", "cppr.", "range.gain_u2")) for k in values)


def test_generic_flattening_names_intervals_and_skips_noise(scratch):
    doc = dict(hypotheses={"H1": dict(networks={"wiki-rfa": dict(components=dict(b_cut=dict(
        estimate=0.9, ci95=[0.81, 0.97], p_value=0.004, decision="met", met=True, by_seed=list(range(11)),
        bootstrap_streams=dict(crossed=[1, 2]))))})}, provenance=dict(n=1), status="ok")
    path = os.path.join(scratch, "results", "analysis", "cppr_confirm.json")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(doc, f)
    values, _ = V.build(scratch)
    base = "cppr.hypotheses.h1.networks.wiki.components.b_cut"
    assert {k: e["text"] for k, e in values.items()} == {
        f"{base}.estimate": "0.900", f"{base}.ci95.lo": "0.810", f"{base}.ci95.hi": "0.970",
        f"{base}.p_value": "0.0040", f"{base}.decision": "met", "cppr.status": "ok"}
    assert values[f"{base}.ci95.hi"]["path"] == "/hypotheses/H1/networks/wiki-rfa/components/b_cut/ci95/1"


def test_generic_flattening_of_a_real_module_output(scratch):
    """cppr_confirm.json of the CPPR analysis' synthetic runs (Alpha only): every key resolves."""
    from tests import cppr_confirm_fixtures as F
    from tests import test_cppr_confirm as TC
    runs = os.path.join(scratch, "runs")
    result = json.loads(json.dumps(C._finite(TC.analyse(runs, F.build_results(runs))), default=C._json_default))
    path = os.path.join(scratch, "results", "analysis", "cppr_confirm.json")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(result, f)
    values, _ = V.build(scratch)
    assert values["cppr.hypotheses.h1.networks.alpha.components.b_cut.reading"]["value"] == "met"
    assert values["cppr.h3a.n_runs"]["value"] == result["h3a"]["n_runs"]
    assert all(V.resolve(result, e["path"]) == e["value"] for e in values.values())


# ------------------------------------------------------------------------ --check and LaTeX
def test_check_passes_on_fresh_outputs_and_fails_on_a_stale_one(scratch):
    copy(V.E9A, scratch)
    args = ["--results-root", scratch, "--out-root", scratch]
    assert V.main(args + ["--check"]) == 1                 # nothing written yet
    assert V.main(args) == 0 and V.main(args + ["--check"]) == 0
    with open(os.path.join(scratch, V.OUT_TEX), "a") as f:
        f.write("% edited by hand\n")
    assert V.main(args + ["--check"]) == 1


def test_the_repository_outputs_are_current():
    assert V.main(["--check"]) == 0


@pytest.mark.skipif(not os.path.isfile(PDFLATEX), reason="pdflatex not installed")
def test_macros_compile_and_an_undefined_key_is_visible(scratch, stored):
    """A tiny IEEEtran document: hand-made cases on page 1, then every stored key (numbers in math mode)."""
    fitz = pytest.importorskip("fitz")
    values = {"h8.alpha.gain_on.est": dict(text="-0.268"), "x.count_1": dict(text="12{,}949"),
              "x.p": dict(text=r"1.7\times10^{-11}"), "dec.h8.decision": dict(text="not supported")}
    every = "".join(f"\\val{{{k}}}\\par\n" if e["format"] == "str" else f"$\\val{{{k}}}$\\par\n"
                    for k, e in stored.items())
    with open(os.path.join(scratch, "keeper_values.tex"), "w") as f:
        f.write(V.render_tex({**stored, **values}))
    shutil.copy(os.path.join(C.ROOT, "paper", "IEEEtran.cls"), scratch)
    with open(os.path.join(scratch, "doc.tex"), "w") as f:
        f.write("\\documentclass[lettersize,journal]{IEEEtran}\n\\usepackage{xcolor}\n\\usepackage{hyperref}\n"
                "\\input{keeper_values}\n\\begin{document}\n"
                "Gain $\\val{h8.alpha.gain_on.est}$, count $\\val{x.count_1}$, p $\\val{x.p}$, "
                "decision \\val{dec.h8.decision}. Text \\val{no.such_key} and math $\\val{no.such_key2}$.\n"
                f"\\newpage\n{every}\\end{{document}}\n")
    run = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=scratch,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout[-2000:]
    pdf = fitz.open(os.path.join(scratch, "doc.pdf"))
    assert not any("??" in p.get_text() for p in list(pdf)[1:])                # every stored key is defined
    page = pdf[0]
    text = " ".join(page.get_text().split())
    assert "Gain −0.268, count 12,949, p 1.7 × 10−11" in text and "not supported." in text
    assert "??no.such_key??" in text and "??no.such_key2??" in text
    spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]]
    marks = [s for s in spans if "??" in s["text"]]
    assert marks and all(s["color"] == 0xFF0000 for s in marks)              # red
    assert any(s["flags"] & 16 for s in marks)                                 # bold
    with open(os.path.join(scratch, "doc.log"), errors="replace") as f:
        assert "Undefined control sequence" not in f.read()
