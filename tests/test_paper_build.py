"""Tests for paper/build_isolated.py and paper/package_overleaf.py."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))
import build_isolated as bi  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "latex_fixture"
needs_tex = pytest.mark.skipif(not (bi.TEXBIN / "pdflatex").exists(), reason="pdflatex not installed")

@needs_tex
def test_fixture_builds_in_isolation_without_question_marks():
    main = bi.build(FIXTURE / "main.tex")
    assert main["problems"] == [] and main["files"] == ["main.tex"] and main["qq"] == 0


@needs_tex
def test_max_pages_enforced():
    res = bi.build(FIXTURE / "main.tex", max_pages=0)
    assert any("max-pages" in p for p in res["problems"])


# ------------------------------------------------------------------ paper/package_overleaf.py
@needs_tex
def test_package_flattens_outside_inputs_and_compiles_from_its_own_root(tmp_path):
    """A document that inputs ../shared/part.tex becomes a folder that holds shared/part.tex and compiles alone."""
    import package_overleaf as po
    (tmp_path / "doc").mkdir()
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "part.tex").write_text("Shared text.\n")
    (tmp_path / "doc" / "local.tex").write_text("Local text.\n")
    (tmp_path / "doc" / "d.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{../shared/part.tex}\n\\input{local}\n\\end{document}\n")
    out = tmp_path / "pkg"
    files = po.package(tmp_path / "doc" / "d.tex", out)
    assert sorted(files) == ["d.tex", "local.tex", "shared/part.tex"]
    assert "\\input{shared/part.tex}" in (out / "d.tex").read_text() and "../" not in (out / "d.tex").read_text()
    res = po.compile_in_place(out, "d.tex")
    assert res["problems"] == [] and res["pages"] == 1
    assert sorted(p.name for p in out.iterdir()) == ["d.tex", "local.tex", "shared"]      # build files removed


def test_package_refuses_two_files_with_one_package_path(tmp_path):
    import package_overleaf as po
    for folder in ("doc/tables", "tables"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "t.tex").write_text("x\n")
    (tmp_path / "doc" / "d.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{tables/t.tex}\n\\input{../tables/t.tex}\n\\end{document}\n")
    with pytest.raises(ValueError, match="tables/t.tex"):
        po.package(tmp_path / "doc" / "d.tex", tmp_path / "pkg")
