#!/usr/bin/env python3
"""Build one LaTeX document alone, in a fresh folder holding only the files it needs.

  python paper/build_isolated.py [DOC.tex] [--pdf-out OUT.pdf] [--max-pages N]

DOC defaults to paper/main_scem.tex. Prints the needed-file list (the Overleaf upload
list), builds with pdflatex (+ bibtex if \\bibliography is used) and exits non-zero if the
log has errors, undefined references/citations or Overfull \\hbox, if the PDF text
contains "??", or if the page count exceeds --max-pages. Temporary folders are kept.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEXBIN = Path(os.environ.get("TEXBIN", "~/texlive/2024/bin/x86_64-linux")).expanduser()
GRAPHIC_EXTS = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def _first_existing(base_dirs, name, exts):
    for d in base_dirs:
        for e in exts:
            p = (d / (name + e)).resolve()
            if p.is_file():
                return p
    return None


def scan(doc):
    """Return (needed files as resolved Paths, missing description list, uses_bibtex)."""
    needed, missing, uses_bib = {doc.resolve()}, [], False
    queue = [doc.resolve()]
    while queue:
        f = queue.pop()
        d = doc.resolve().parent  # TeX resolves every path from the main file's folder
        tex = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
        gpaths = [d / g for g in re.findall(r"\{([^{}]*)\}", " ".join(re.findall(r"\\graphicspath\s*\{((?:\{[^{}]*\})+)\}", tex)))]
        found = []

        def want(kind, name, exts, dirs):
            p = _first_existing(dirs, name, exts)
            if p:
                found.append(p)
            elif kind in ("input", "graphic"):  # classes/styles/bst may come from the TeX tree
                missing.append("%s: %s (from %s)" % (kind, name, f.name))

        for m in re.finditer(r"\\(?:input|include)\s*\{([^}]*)\}", tex):
            want("input", m.group(1).strip(), (".tex", ""), [d])
        for m in re.finditer(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]*)\}", tex):
            want("graphic", m.group(1).strip(), GRAPHIC_EXTS, [d] + gpaths)
        for m in re.finditer(r"\\documentclass(?:\[[^\]]*\])?\{([^}]*)\}", tex):
            want("class", m.group(1).strip(), (".cls",), [d])
        for m in re.finditer(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]*)\}", tex):
            for pkg in m.group(1).split(","):
                want("package", pkg.strip(), (".sty",), [d])
        for m in re.finditer(r"\\bibliographystyle\{([^}]*)\}", tex):
            want("bst", m.group(1).strip(), (".bst",), [d])
        for m in re.finditer(r"\\bibliography\{([^}]*)\}", tex):
            uses_bib = True
            for b in m.group(1).split(","):
                want("bib", b.strip(), (".bib",), [d])
        for p in found:
            if p not in needed:
                needed.add(p)
                if p.suffix == ".tex":
                    queue.append(p)
    return sorted(needed), missing, uses_bib


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)


def check_log(log):
    """Return {category: [offending lines]} for the final-pass log."""
    lines = log.splitlines()
    return {
        "errors": [l for l in lines if l.startswith("!")],
        "undefined": [l for l in lines if re.search(r"(Reference|Citation) `[^']*' on page \d+ undefined", l)],
        "overfull": [l for l in lines if l.startswith("Overfull \\hbox")],
    }


def pdf_stats(pdf):
    import fitz  # PyMuPDF

    with fitz.open(pdf) as d:
        text = "\n".join(p.get_text() for p in d)
        return len(d), text.count("??")


def build(doc, pdf_out=None, max_pages=None, workroot=None):
    """Build doc alone; return a result dict (never raises on LaTeX failure)."""
    doc = Path(doc).resolve()
    files, missing, uses_bib = scan(doc)
    root = Path(os.path.commonpath([str(p.parent) for p in files]))
    tmp = Path(tempfile.mkdtemp(prefix="isolated_" + doc.stem + "_", dir=workroot))
    for p in files:
        dst = tmp / p.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    wd = tmp / doc.parent.relative_to(root)
    pdflatex = [str(TEXBIN / "pdflatex"), "-interaction=nonstopmode", "-halt-on-error", doc.name]
    res = {"tmp": tmp, "files": [str(p.relative_to(root)) for p in files], "missing": missing,
           "problems": [], "pages": None, "qq": None, "counts": {}}
    if missing:
        res["problems"] += ["needed file not found: " + m for m in missing]
    steps = [pdflatex] + ([[str(TEXBIN / "bibtex"), doc.stem]] if uses_bib else []) + [pdflatex, pdflatex]
    for i, cmd in enumerate(steps):
        r = run(cmd, wd)
        if r.returncode != 0:
            res["problems"].append("%s failed (step %d)" % (Path(cmd[0]).name, i + 1))
            if cmd[0].endswith("pdflatex"):
                break
    logf = wd / (doc.stem + ".log")
    log = logf.read_text(encoding="utf-8", errors="replace") if logf.exists() else ""
    found = check_log(log)
    res["counts"] = {k: len(v) for k, v in found.items()}
    for cat, lines in found.items():
        res["problems"] += ["%s: %s" % (cat, l) for l in lines]
    pdf = wd / (doc.stem + ".pdf")
    if pdf.exists():
        res["pages"], res["qq"] = pdf_stats(pdf)
        if res["qq"]:
            res["problems"].append('"??" occurs %d time(s) in the PDF text' % res["qq"])
        if max_pages is not None and res["pages"] > max_pages:
            res["problems"].append("%d pages exceeds --max-pages %d" % (res["pages"], max_pages))
        if pdf_out:
            shutil.copy2(pdf, pdf_out)
    else:
        res["problems"].append("no PDF produced")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", nargs="?", default="paper/main_scem.tex")
    ap.add_argument("--pdf-out")
    ap.add_argument("--max-pages", type=int)
    a = ap.parse_args(argv)
    res = build(a.doc, a.pdf_out, a.max_pages)
    print("Needed files (Overleaf upload list, paths relative to a common root):")
    for f in res["files"]:
        print("  " + f)
    print("build folder:", res["tmp"])
    print("pages=%s  ??=%s  counts=%s" % (res["pages"], res["qq"], res["counts"]))
    for p in res["problems"]:
        print("FAIL " + p, file=sys.stderr)
    print("RESULT:", "FAILED" if res["problems"] else "ok")
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
