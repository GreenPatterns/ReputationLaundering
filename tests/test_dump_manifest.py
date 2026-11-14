"""keeper.analysis.dump_manifest: the SHA-256 manifest of a dump directory (Task 4.1a).

The manifest lists every regular file under the directory as `<sha256>  <relative path>`, sorted by path, and
`--check` recomputes it. An existing manifest is replaced only with --force (ruling R27). One test also pins the
.gitignore rule that keeps dump files out of git but tracks their manifests.

Every test works in pytest's tmp_path (stored results are write-protected, ruling R27). CPU only, no GPU.
"""
import errno
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from keeper.analysis import dump_manifest as M

ROOT = Path(__file__).resolve().parents[1]
FILES = {"a.npz": b"alpha", "b.json": b'{"b": 1}', "sub/c.bin": b"\x00\x01\x02", "sub/deeper/d.txt": b"delta",
         "Z.txt": b"capital"}
SORTED_NAMES = ["Z.txt", "a.npz", "b.json", "sub/c.bin", "sub/deeper/d.txt"]     # by path: capitals sort first


@pytest.fixture
def dump(tmp_path) -> Path:
    """A dump directory with nested files (tmp_path/dump)."""
    root = tmp_path / "dump"
    for rel, data in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def manifest_lines(root: Path) -> list:
    return (root / M.MANIFEST).read_text().splitlines()


# ------------------------------------------------------------------ writing
def test_writes_one_sorted_sha256_line_per_file(dump, capsys):
    assert M.main([str(dump)]) == 0
    lines = manifest_lines(dump)
    assert [line.split("  ", 1)[1] for line in lines] == SORTED_NAMES
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert digest == hashlib.sha256((dump / rel).read_bytes()).hexdigest()
    assert (dump / M.MANIFEST).read_text().endswith("\n")
    assert "5 files" in capsys.readouterr().out


def test_render_sorts_by_path_whatever_the_order_it_is_given():
    entries = {"b/x.bin": "1" * 64, "B.txt": "2" * 64, "a.bin": "3" * 64}
    assert M.render(entries) == f"{'2' * 64}  B.txt\n{'3' * 64}  a.bin\n{'1' * 64}  b/x.bin\n"


def test_manifest_does_not_list_itself(dump):
    M.main([str(dump)])
    M.main([str(dump), "--force"])
    assert M.MANIFEST not in [line.split("  ", 1)[1] for line in manifest_lines(dump)]


def test_manifest_is_in_the_format_sha256sum_reads(dump):
    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is not installed")
    M.main([str(dump)])
    res = subprocess.run(["sha256sum", "--check", "--strict", M.MANIFEST], cwd=dump, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_files_larger_than_one_read_chunk_are_hashed_whole(dump, monkeypatch):
    big = bytes(range(256)) * 41                                  # 10,496 bytes
    (dump / "big.bin").write_bytes(big)
    monkeypatch.setattr(M, "CHUNK", 1000)                         # eleven reads instead of one
    M.main([str(dump)])
    assert f"{hashlib.sha256(big).hexdigest()}  big.bin" in manifest_lines(dump)


def test_an_empty_directory_gets_an_empty_manifest_that_checks(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert M.main([str(empty)]) == 0
    assert (empty / M.MANIFEST).read_text() == ""
    assert M.main([str(empty), "--check"]) == 0


def test_only_regular_files_are_listed(dump):
    (dump / "link.txt").symlink_to(dump / "a.npz")
    (dump / "linkdir").symlink_to(dump / "sub", target_is_directory=True)
    M.main([str(dump)])
    assert [line.split("  ", 1)[1] for line in manifest_lines(dump)] == SORTED_NAMES


def test_a_nested_manifest_is_an_ordinary_file(dump):
    (dump / "sub" / M.MANIFEST).write_text("inner\n")
    M.main([str(dump)])
    assert "sub/" + M.MANIFEST in [line.split("  ", 1)[1] for line in manifest_lines(dump)]


# ------------------------------------------------------------------ the overwrite guard (ruling R27)
def test_refuses_to_replace_a_manifest_without_force(dump, capsys):
    M.main([str(dump)])
    before = (dump / M.MANIFEST).read_bytes()
    (dump / "a.npz").write_bytes(b"omega")
    capsys.readouterr()
    assert M.main([str(dump)]) == 1
    assert "--force" in capsys.readouterr().err
    assert (dump / M.MANIFEST).read_bytes() == before


def test_force_replaces_the_manifest(dump):
    M.main([str(dump)])
    (dump / "a.npz").write_bytes(b"omega")
    assert M.main([str(dump), "--force"]) == 0
    assert f"{hashlib.sha256(b'omega').hexdigest()}  a.npz" in manifest_lines(dump)


def test_force_on_an_unchanged_directory_rewrites_identical_bytes(dump):
    M.main([str(dump)])
    before = (dump / M.MANIFEST).read_bytes()
    assert M.main([str(dump), "--force"]) == 0
    assert (dump / M.MANIFEST).read_bytes() == before


def test_the_guard_fires_before_any_file_is_hashed(dump, monkeypatch):
    M.main([str(dump)])
    monkeypatch.setattr(M, "sha256_file", lambda path: pytest.fail("hashed a file before the overwrite guard"))
    assert M.main([str(dump)]) == 1


def test_a_manifest_that_appears_after_the_guard_is_not_replaced_either(dump, monkeypatch):
    M.main([str(dump)])
    before = (dump / M.MANIFEST).read_bytes()
    (dump / "a.npz").write_bytes(b"omega")
    monkeypatch.setattr(M.os.path, "exists", lambda path: False)       # the guard sees no manifest, as if it came later
    with pytest.raises(FileExistsError, match="--force"):                # the same refusal, with its hint
        M.write_manifest(str(dump))
    assert (dump / M.MANIFEST).read_bytes() == before
    assert sorted(p.name for p in dump.iterdir()) == sorted(["Z.txt", "a.npz", "b.json", "sub", M.MANIFEST])


# ------------------------------------------------------------------ the write is atomic
def test_the_manifest_is_written_to_a_temporary_file_and_linked_or_replaced_into_place(dump, monkeypatch):
    calls = []

    def spying(name):
        real = getattr(os, name)
        return lambda src, dst: (calls.append((name, src, dst)), real(src, dst))[1]

    monkeypatch.setattr(M.os, "link", spying("link"))
    monkeypatch.setattr(M.os, "replace", spying("replace"))
    assert M.main([str(dump)]) == 0 and M.main([str(dump), "--force"]) == 0
    assert [name for name, _, _ in calls] == ["link", "replace"]       # no force: link; --force: replace
    for _, src, dst in calls:
        assert os.path.dirname(src) == str(dump) and dst == str(dump / M.MANIFEST) and src != dst
    assert sorted(p.name for p in dump.iterdir()) == sorted(["Z.txt", "a.npz", "b.json", "sub", M.MANIFEST])


@pytest.mark.parametrize("force, failing", [(False, "link"), (True, "replace")])
def test_a_failed_write_leaves_the_old_manifest_whole_and_no_temporary_file(dump, monkeypatch, capsys, force, failing):
    before = None
    if force:
        M.main([str(dump)])
        before = (dump / M.MANIFEST).read_bytes()
    (dump / "a.npz").write_bytes(b"omega")

    def refuse(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(M.os, failing, refuse)
    capsys.readouterr()
    assert M.main([str(dump), *(["--force"] if force else [])]) == 1
    assert "No space" in capsys.readouterr().err
    names = sorted(p.name for p in dump.iterdir())
    assert names == sorted(["Z.txt", "a.npz", "b.json", "sub"] + ([M.MANIFEST] if force else []))   # no temporary file
    assert not force or (dump / M.MANIFEST).read_bytes() == before


def test_the_temporary_file_of_the_write_is_not_listed(dump):
    (dump / (M.MANIFEST + ".tmp-123")).write_text("half written")
    M.main([str(dump)])
    assert [line.split("  ", 1)[1] for line in manifest_lines(dump)] == SORTED_NAMES


def make_undecodable_name(dump) -> None:
    try:
        os.close(os.open(os.fsencode(str(dump)) + b"/bad\xff.bin", os.O_CREAT | os.O_WRONLY))
    except OSError:
        pytest.skip("this file system does not take a file name that is not UTF-8")


def test_a_file_name_that_cannot_be_encoded_is_reported_and_nothing_is_written(dump, capsys):
    make_undecodable_name(dump)
    assert M.main([str(dump)]) == 1
    assert "codec" in capsys.readouterr().err                          # a message, not a traceback
    assert sorted(p.name for p in dump.iterdir() if p.name.startswith("MANIFEST")) == []


def test_check_reports_a_file_name_that_cannot_be_printed_instead_of_raising(dump, capsys):
    M.main([str(dump)])
    make_undecodable_name(dump)
    assert M.main([str(dump), "--check"]) == 1
    assert "codec" in capsys.readouterr().err


# ------------------------------------------------------------------ --check
def test_check_passes_on_an_untouched_dump(dump, capsys):
    M.main([str(dump)])
    capsys.readouterr()
    assert M.main([str(dump), "--check"]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_lists_changed_missing_and_extra_files(dump, capsys):
    M.main([str(dump)])
    before = (dump / M.MANIFEST).read_bytes()
    (dump / "a.npz").write_bytes(b"omega")                        # same length as "alpha": content, not size
    (dump / "b.json").unlink()
    (dump / "sub" / "new.bin").write_bytes(b"new")
    capsys.readouterr()
    assert M.main([str(dump), "--check"]) == 1
    out = capsys.readouterr().out
    assert "changed: a.npz" in out
    assert "missing: b.json" in out
    assert "extra: sub/new.bin" in out
    for untouched in ("Z.txt", "sub/c.bin", "sub/deeper/d.txt"):
        assert untouched not in out
    assert (dump / M.MANIFEST).read_bytes() == before            # a check never writes


@pytest.mark.parametrize("damage, heading", [
    (lambda d: (d / "Z.txt").write_bytes(b"CAPITAL"), "changed: Z.txt"),
    (lambda d: (d / "Z.txt").unlink(), "missing: Z.txt"),
    (lambda d: (d / "extra.txt").write_text("x"), "extra: extra.txt"),
])
def test_check_flags_each_kind_of_difference_alone(dump, capsys, damage, heading):
    M.main([str(dump)])
    damage(dump)
    capsys.readouterr()
    assert M.main([str(dump), "--check"]) == 1
    assert heading in capsys.readouterr().out


def test_check_without_a_manifest_fails(dump, capsys):
    assert M.main([str(dump), "--check"]) == 1
    assert M.MANIFEST in capsys.readouterr().err


@pytest.mark.parametrize("text", ["not a manifest\n", f"{'0' * 63}  short.txt\n", f"{'g' * 64}  nothex.txt\n",
                                  f"{'0' * 64} one-space.txt\n", f"{'0' * 64}  a.txt\n{'1' * 64}  a.txt\n"])
def test_check_rejects_a_malformed_manifest(dump, capsys, text):
    (dump / M.MANIFEST).write_text(text)
    assert M.main([str(dump), "--check"]) == 1
    assert "line" in capsys.readouterr().err


def test_check_rejects_a_manifest_that_is_not_text(dump, capsys):
    (dump / M.MANIFEST).write_bytes(b"\xff\xfe\x00\x80")
    assert M.main([str(dump), "--check"]) == 1
    assert "UTF-8" in capsys.readouterr().err


# ------------------------------------------------------------------ bad input and the module entry
def test_a_missing_directory_is_reported_not_raised(tmp_path, capsys):
    assert M.main([str(tmp_path / "nowhere")]) == 1
    assert "nowhere" in capsys.readouterr().err


def test_file_names_with_a_line_break_are_refused_and_nothing_is_written(dump, capsys):
    (dump / "bad\nname.txt").write_text("x")
    assert M.main([str(dump)]) == 1
    assert "line break" in capsys.readouterr().err
    assert not (dump / M.MANIFEST).exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any directory")
def test_an_unreadable_directory_fails_the_run_instead_of_shortening_the_manifest(dump, capsys):
    locked = dump / "locked"
    locked.mkdir()
    (locked / "hidden.bin").write_bytes(b"hidden")
    locked.chmod(0)
    try:
        assert M.main([str(dump)]) == 1
        assert "locked" in capsys.readouterr().err
        assert not (dump / M.MANIFEST).exists()
    finally:
        locked.chmod(0o700)


def test_runs_as_a_module(dump):
    res = subprocess.run([sys.executable, "-m", "keeper.analysis.dump_manifest", str(dump)], cwd=ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert [line.split("  ", 1)[1] for line in manifest_lines(dump)] == SORTED_NAMES


# ------------------------------------------------------------------ .gitignore
def test_gitignore_keeps_dump_files_out_of_git_but_tracks_their_manifests(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    visible_to_git = {                            # path -> shows up as untracked (False: ignored)
        "results/dumps/e3/verdicts_s5.json": False,
        "results/dumps/e3/deep/er/targets_s5.json": False,
        "results/dumps/loose.json": False,
        "results/dumps/e3/MANIFEST_SHA256.txt": True,
        "results/dumps/e3/deep/er/MANIFEST_SHA256.txt": True,
        "results/dumps/MANIFEST_SHA256.txt": True,
        "results/laundering/stored.json": True,   # the rest of results/ stays tracked
    }
    for rel in visible_to_git:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x")
    env = dict(os.environ, HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "xdg"),
               GIT_CONFIG_NOSYSTEM="1")           # no personal ignore rules
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True, capture_output=True)
    res = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, env=env, check=True,
                         capture_output=True, text=True)
    assert set(res.stdout.split()) == {rel for rel, visible in visible_to_git.items() if visible} | {".gitignore"}
