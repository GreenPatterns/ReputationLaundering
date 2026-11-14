"""Provenance: keeper.analysis.provenance, the git helpers that the result writers share (Task 3.1 review, item e),
and the library versions of keeper.laundering.core.provenance (Task 4.1b)."""
import platform
import re
import subprocess
import sys
from importlib import metadata

import numpy as np
import pytest

from keeper.analysis import provenance as P
from keeper.laundering import core as C

LIBRARIES = ("numpy", "scipy", "pandas", "torch_scatter", "torch_geometric", "scikit-learn")


def test_git_state_has_the_code_state_fields():
    state = P.git_state()
    assert list(state) == ["git_commit", "git_dirty", "git_untracked", "git_status"]
    assert state["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", state["git_commit"])
    assert isinstance(state["git_dirty"], bool) and isinstance(state["git_untracked"], int)
    assert state["git_untracked"] == sum(line.startswith("??") for line in state["git_status"])


def _git(repo, *args):
    """git in a throwaway repository, independent of the user's identity, signing and hooks settings."""
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false",
                    "-c", "core.hooksPath=/dev/null", *args], cwd=repo, check=True, capture_output=True)


@pytest.mark.parametrize("show_untracked", [None, "no", "all"])
def test_git_dirty_means_tracked_changes_and_untracked_paths_are_counted(tmp_path, monkeypatch, show_untracked):
    # git_dirty: a tracked file differs from HEAD, staged or not. Untracked paths do not make the tree dirty; they
    # are counted in git_untracked (an untracked folder is one path), and git_status keeps the exact porcelain list.
    # A host's status.showUntrackedFiles setting changes none of it (fix round: --untracked-files=normal).
    _git(tmp_path, "init", "-q")
    if show_untracked:
        _git(tmp_path, "config", "status.showUntrackedFiles", show_untracked)
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.setattr(P, "ROOT", str(tmp_path))
    clean = P.git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", clean["git_commit"])
    assert (clean["git_dirty"], clean["git_untracked"], clean["git_status"]) == (False, 0, [])
    (tmp_path / "notes.txt").write_text("scratch\n")
    (tmp_path / "logs").mkdir()
    for name in ("a.log", "b.log"):
        (tmp_path / "logs" / name).write_text("")
    untracked = P.git_state()
    assert (untracked["git_dirty"], untracked["git_untracked"]) == (False, 2)
    assert untracked["git_status"] == ["?? logs/", "?? notes.txt"]
    (tmp_path / "code.py").write_text("x = 2\n")                  # a tracked modification, not staged
    modified = P.git_state()
    assert (modified["git_dirty"], modified["git_untracked"]) == (True, 2)
    assert modified["git_status"] == [" M code.py", "?? logs/", "?? notes.txt"]   # the leading space is kept
    _git(tmp_path, "checkout", "--", "code.py")
    _git(tmp_path, "add", "notes.txt")                             # staged: now tracked
    staged = P.git_state()
    assert (staged["git_dirty"], staged["git_untracked"]) == (True, 1)
    assert staged["git_status"] == ["A  notes.txt", "?? logs/"]


def test_git_keeps_leading_whitespace_and_drops_only_the_final_newlines(monkeypatch):
    # The porcelain status code is two columns: " M" (modified, not staged) differs from "M " (staged).
    class Done:
        returncode, stdout = 0, " M first.py\nM  second.py\n\n"

    monkeypatch.setattr(P.subprocess, "run", lambda *args, **kwargs: Done)
    assert P.git("status", "--porcelain") == " M first.py\nM  second.py"


def test_git_returns_none_when_the_command_fails():
    assert P.git("no-such-subcommand-of-git") is None


def test_the_analysis_writers_use_the_shared_helper():
    from keeper.analysis import defense_extra, laundering_extra, tiers_report
    for module in (laundering_extra, defense_extra, tiers_report):
        assert not hasattr(module, "_git") and module.git_state is P.git_state
        assert "keeper/analysis/provenance.py" in module.MODULE_FILES


def test_core_provenance_records_python_and_the_library_versions():
    import sklearn
    prov = C.provenance()
    assert prov["python"] == platform.python_version()
    assert prov["numpy"] == np.__version__ and prov["scikit-learn"] == sklearn.__version__
    for name in LIBRARIES:
        assert prov[name] == metadata.version(name), name                     # the installed distribution
    assert prov["torch"] and prov["code_sha256"]                               # the existing fields stay


def test_a_package_that_is_not_installed_is_recorded_as_none(monkeypatch):
    real = metadata.version

    def version(name):
        if name in ("pandas", "torch_geometric"):
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", version)
    prov = C.provenance()
    assert prov["pandas"] is None and prov["torch_geometric"] is None
    assert prov["numpy"] == np.__version__


def test_reading_the_versions_loads_no_package(monkeypatch):
    # The versions come from the installed package metadata, so provenance() loads no package that the run did not
    # import. Only the SignedGCN baseline imports torch_geometric, whose import takes about 2 s and warns on hosts
    # whose GLIBC its compiled extensions do not match.
    monkeypatch.delitem(sys.modules, "torch_geometric", raising=False)
    C.provenance()
    assert "torch_geometric" not in sys.modules
