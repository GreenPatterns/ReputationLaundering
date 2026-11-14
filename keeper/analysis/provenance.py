"""Provenance helpers shared by the result writers (keeper.analysis's laundering_extra, defense_extra and tiers_report,
and the CPPR driver keeper.cppr.cp_vpe_gpu).

git runs a git command in the repository root, and git_state gives the code-state fields every result's provenance
starts with: the commit, whether a tracked file was modified, how many paths were untracked, and the full status.
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root


def git(*args) -> str | None:
    """Stdout of `git <args>` run in the repository root, without its final newlines (leading whitespace is kept: in
    `git status --porcelain`, " M" and "M " are different states); None if git is missing or fails."""
    try:
        res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return res.stdout.rstrip("\n") if res.returncode == 0 else None


def git_state() -> dict:
    """The code state when the result was written (Task 4.1b): git_commit (HEAD); git_dirty, True when a tracked file
    differs from HEAD, staged or not (untracked files do not count: `git status --porcelain --untracked-files=no`);
    git_untracked, the number of untracked paths (an untracked folder counts once); git_status, the exact porcelain
    list, untracked paths included. Both status calls name their untracked mode, so a host's
    status.showUntrackedFiles setting changes neither the list nor the count."""
    status = git("status", "--porcelain", "--untracked-files=normal")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    lines = status.splitlines() if status else []
    return dict(git_commit=git("rev-parse", "HEAD"), git_dirty=bool(tracked),
                git_untracked=sum(line.startswith("??") for line in lines), git_status=lines)
