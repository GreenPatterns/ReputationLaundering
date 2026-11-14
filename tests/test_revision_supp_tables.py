import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from keeper.paperkit import make_revision_supp_tables as g

PDFLATEX = Path.home() / "texlive/2024/bin/x86_64-linux/pdflatex"
DOC = (r"\documentclass[lettersize,journal]{IEEEtran}" "\n" r"\usepackage{booktabs,array,multirow,amsmath}" "\n"
       r"\begin{document}" "\n%s\n" r"\end{document}" "\n")


def test_check_passes_on_committed_outputs():
    assert g.main(["--check"]) == 0


def test_perturbed_source_changes_one_table(tmp_path):
    for f in (g.E1, g.LAUND, g.TIERS, g.E9):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(g.ROOT / f, tmp_path / f)
    base, _, _ = g.build(tmp_path)
    p = tmp_path / g.LAUND
    data = json.loads(p.read_text())
    data["bitcoinalpha"]["amplification"]["laplace"] += 0.5
    p.write_text(json.dumps(data))
    new, _, _ = g.build(tmp_path)
    assert [n for n in base if base[n] != new[n]] == ["s_yardsticks"]


def test_missing_source_is_skipped(tmp_path):
    texs, _, skipped = g.build(tmp_path)
    assert not texs and len(skipped) == len(g.TABLES)
    assert g.main(["--root", str(tmp_path)]) == 0


@pytest.mark.skipif(not PDFLATEX.exists(), reason="pdflatex missing")
def test_every_table_compiles_without_overfull_boxes():
    work = Path(tempfile.mkdtemp())
    shutil.copy(g.ROOT / "paper/IEEEtran.cls", work)
    names = sorted(g.TABLES)
    for n in names:
        shutil.copy(g.ROOT / g.TEX_DIR / f"{n}.tex", work)
    body = "\n\\clearpage\n".join(rf"\input{{{n}.tex}}" for n in names)
    (work / "t.tex").write_text(DOC % body)
    run = subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "t.tex"], cwd=work, capture_output=True,
                         text=True, timeout=300)
    log = (work / "t.log").read_text()
    assert run.returncode == 0 and not re.search(r"^!", log, re.M)
    assert "Overfull \\hbox" not in log and "Float too large" not in log
