#!/usr/bin/env python3
"""Write the self-contained Overleaf folder of the manuscript and compile it there.

  python paper/package_overleaf.py [--out paper/overleaf] [--max-pages 18]

1. Package. Copy paper/main_scem.tex, with every file it needs (class file, figures, inputs), into OUT. A needed
   file outside the document's own folder (../x/y.tex) is placed inside the package (x/y.tex) and the paths in the
   copied sources are rewritten, so that the folder compiles from its own root, as Overleaf does.
2. Check. Compile the package in place and fail on a LaTeX error, an undefined reference, an Overfull \\hbox,
   "??" in the PDF text, or more than --max-pages pages. The PDF is written next to the source and the build
   files are removed from the package.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_isolated as bi  # noqa: E402

MAIN = HERE / "main_scem.tex"
BUILD_SUFFIXES = (".aux", ".log", ".out", ".pdf", ".toc", ".bbl", ".blg")


def package_path(path: Path, doc_dir: Path) -> str:
    """The path of a needed file inside the package: its path from the document's folder without leading ../."""
    parts = Path(os.path.relpath(path, doc_dir)).parts
    return "/".join(p for i, p in enumerate(parts) if p != ".." or any(q != ".." for q in parts[:i]))


def package(doc, out) -> list:
    """Write doc and the files it needs into the fresh folder out; return the package paths (the upload list)."""
    doc, out = Path(doc).resolve(), Path(out)
    files, missing, _ = bi.scan(doc)
    if missing:
        raise ValueError("needed file not found: " + "; ".join(missing))
    placed = {}
    for p in files:
        rel = package_path(p, doc.parent)
        if rel in placed:
            raise ValueError(f"two needed files map to {rel}: {placed[rel]} and {p}")
        placed[rel] = p
    if out.exists():
        shutil.rmtree(out)
    for rel, p in placed.items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".tex":
            dst.write_text(re.sub(r"\{(?:\.\./)+", "{", p.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(p, dst)
    return sorted(placed)


def compile_in_place(folder, name, max_pages=None, pdf_out=None) -> dict:
    """pdflatex three times in folder; the result of build_isolated's checks; build files are removed afterwards."""
    folder, stem = Path(folder), Path(name).stem
    res = {"problems": [], "pages": None, "qq": None}
    cmd = [str(bi.TEXBIN / "pdflatex"), "-interaction=nonstopmode", "-halt-on-error", name]
    for i in range(3):
        if bi.run(cmd, folder).returncode != 0:
            res["problems"].append(f"pdflatex failed (pass {i + 1})")
            break
    log = folder / (stem + ".log")
    for cat, lines in bi.check_log(log.read_text(encoding="utf-8", errors="replace") if log.exists() else "").items():
        res["problems"] += [f"{cat}: {l}" for l in lines]
    pdf = folder / (stem + ".pdf")
    if pdf.exists():
        res["pages"], res["qq"] = bi.pdf_stats(pdf)
        if res["qq"]:
            res["problems"].append(f'"??" occurs {res["qq"]} time(s) in the PDF text')
        if max_pages is not None and res["pages"] > max_pages:
            res["problems"].append(f"{res['pages']} pages exceeds the limit of {max_pages}")
        if pdf_out:
            shutil.copy2(pdf, pdf_out)
    else:
        res["problems"].append("no PDF produced")
    for p in folder.iterdir():
        if p.is_file() and p.suffix in BUILD_SUFFIXES and p.stem == stem:
            p.unlink()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "overleaf"))
    ap.add_argument("--max-pages", type=int, default=18, help="page limit of the manuscript")
    a = ap.parse_args(argv)
    files = package(MAIN, a.out)
    res = compile_in_place(a.out, MAIN.name, a.max_pages, pdf_out=MAIN.with_suffix(".pdf"))
    print(f"{a.out} ({len(files)} files to upload; compile {MAIN.name}): pages={res['pages']} ??={res['qq']}")
    for f in files:
        print("  " + f)
    for p in res["problems"]:
        print("FAIL " + p, file=sys.stderr)
    print("\nRESULT:", "FAILED" if res["problems"] else "ok")
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
