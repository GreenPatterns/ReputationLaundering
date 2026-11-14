"""SHA-256 manifest of a dump directory (Task 4.1a, ruling C3): the dump files are git-ignored, the manifest is tracked.

Usage (repository root; CPU only, standard library only):
  python -m keeper.analysis.dump_manifest DIR [--check] [--force]

Without --check it writes DIR/MANIFEST_SHA256.txt: one line `<sha256>  <path relative to DIR>` per regular file
under DIR (recursive; symbolic links, other special files and the manifest itself are left out), sorted by path.
That is the format `sha256sum --check` reads when run inside DIR. An existing manifest is replaced only with
--force (ruling R27: stored results are evidence); the guard fires before any file is hashed.

The write is atomic. The text goes to a temporary file in DIR (MANIFEST_SHA256.txt.tmp-<pid>, which the listing
ignores), is flushed to disk, and is then hard-linked into place (it fails if a manifest has appeared meanwhile) or,
with --force, renamed over the old manifest. A reader therefore sees the old manifest or the whole new one, and a
write that fails leaves the old manifest as it was and no temporary file behind.

With --check it recomputes the digests and compares them with the manifest. It lists the files whose content
changed, the files the manifest names that are gone, and the files under DIR that the manifest does not name.
It exits 1 if there is any such file and 0 if DIR matches its manifest. It never writes.

Every failure exits 1 with a message on stderr; a check that cannot run, such as one with no manifest or a
malformed one, counts as a failed check. A file name with a line break cannot be listed and is refused, so that
a manifest line always holds exactly one path; a name that cannot be written as UTF-8 is reported, not raised.
"""
import argparse
import contextlib
import hashlib
import os
import stat
import sys

MANIFEST = "MANIFEST_SHA256.txt"
TEMP_INFIX = ".tmp-"             # MANIFEST_SHA256.txt.tmp-<pid>: the file a write goes through
CHUNK = 1 << 22                  # bytes per read when hashing
DIGEST_LENGTH = 64               # hex characters of a SHA-256
HEX_DIGITS = frozenset("0123456789abcdef")


class ManifestError(Exception):
    """The manifest cannot be written, read or trusted; the message says why."""


def sha256_file(path: str) -> str:
    """Hex SHA-256 of the file's content, read in CHUNK-byte pieces."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _raise(error: OSError):
    raise error


def list_files(root: str) -> list:
    """Paths of the regular files under root, relative to it, written with '/', in no particular order; without
    the manifest and its temporary file. A directory that cannot be read raises instead of being skipped, so a
    manifest never silently lacks files."""
    found = []
    for folder, _dirs, names in os.walk(root, onerror=_raise):
        for name in names:
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if rel == MANIFEST or rel.startswith(MANIFEST + TEMP_INFIX) or not stat.S_ISREG(os.lstat(path).st_mode):
                continue
            if "\n" in rel or "\r" in rel:
                raise ManifestError(f"{rel!r}: a file name with a line break cannot be listed")
            found.append(rel)
    return found


def digests(root: str) -> dict:
    """Path (as list_files gives it) -> sha256 of every regular file under root."""
    return {rel: sha256_file(os.path.join(root, rel)) for rel in list_files(root)}


def render(entries: dict) -> str:
    """The manifest text of a path -> digest mapping: one `<sha256>  <path>` line each, sorted by path."""
    return "".join(f"{digest}  {rel}\n" for rel, digest in sorted(entries.items()))


def parse(text: str, name: str = MANIFEST) -> dict:
    """Path -> digest of a manifest's text; ManifestError names the first line that is not `<sha256>  <path>`."""
    entries = {}
    for number, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        if len(digest) != DIGEST_LENGTH or not set(digest) <= HEX_DIGITS or not rel:
            raise ManifestError(f"{name}: line {number} is not '<sha256>  <path>'")
        if rel in entries:
            raise ManifestError(f"{name}: line {number} lists {rel} a second time")
        entries[rel] = digest
    return entries


def write_manifest(root: str, force: bool = False) -> tuple:
    """Write root/MANIFEST_SHA256.txt atomically; returns (its path, the number of files listed).

    FileExistsError if the manifest exists and force is false: raised before any file is hashed, and again if a
    manifest appears while the temporary file is written (the hard link into place fails, so it is not replaced)."""
    path = os.path.join(root, MANIFEST)
    refusal = FileExistsError(f"{path} exists; pass --force to replace it (stored results are evidence, ruling R27)")
    if os.path.exists(path) and not force:
        raise refusal
    text = render(digests(root))
    temporary = f"{path}{TEMP_INFIX}{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise refusal from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    return path, text.count("\n")


def read_manifest(path: str) -> dict:
    """Path -> digest of the manifest file at path."""
    try:
        with open(path, encoding="utf-8") as f:
            return parse(f.read(), path)
    except FileNotFoundError:
        raise ManifestError(f"{path} does not exist; write it first (without --check)") from None
    except UnicodeDecodeError:
        raise ManifestError(f"{path} is not a UTF-8 text file") from None


def compare(expected: dict, actual: dict) -> dict:
    """The differences of two path -> digest mappings: changed (digest differs), missing (only in expected) and
    extra (only in actual), each a sorted list of paths."""
    return dict(changed=sorted(rel for rel in expected.keys() & actual.keys() if expected[rel] != actual[rel]),
                missing=sorted(expected.keys() - actual.keys()),
                extra=sorted(actual.keys() - expected.keys()))


def check_manifest(root: str) -> tuple:
    """(differences, number of files the manifest lists) of root against its manifest; see compare."""
    expected = read_manifest(os.path.join(root, MANIFEST))
    return compare(expected, digests(root)), len(expected)


def run_check(root: str) -> int:
    """Print the outcome of the check of root; 0 if it matches its manifest, else 1."""
    diff, n_files = check_manifest(root)
    if not any(diff.values()):
        print(f"OK: {n_files} files under {root} match {MANIFEST}")
        return 0
    print(f"FAILED: {root} differs from {MANIFEST} ({', '.join(f'{len(v)} {k}' for k, v in diff.items())})")
    for kind, names in diff.items():
        for rel in names:
            print(f"  {kind}: {rel}")
    return 1


def run_write(root: str, force: bool) -> int:
    path, n_files = write_manifest(root, force)
    print(f"wrote {path} ({n_files} files)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write or check the SHA-256 manifest of a dump directory.")
    ap.add_argument("dir", metavar="DIR", help="the dump directory")
    ap.add_argument("--check", action="store_true",
                    help="recompute the digests and compare them with the manifest; writes nothing, exits 1 on "
                         "any changed, missing or extra file")
    ap.add_argument("--force", action="store_true", help="replace an existing manifest (ruling R27)")
    args = ap.parse_args(argv)
    try:
        if not os.path.isdir(args.dir):
            raise ManifestError(f"{args.dir} is not a directory")
        return run_check(args.dir) if args.check else run_write(args.dir, args.force)
    except (ManifestError, OSError, UnicodeError) as error:        # FileExistsError is an OSError
        print(f"dump_manifest: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
