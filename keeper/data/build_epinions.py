#!/usr/bin/env python3
"""Build the dated Epinions dataset for the reputation-laundering study.

PRIMARY SOURCE (single self-contained file, sign AND timestamp)
----------------------------------------------------------------
  KONECT  http://konect.cc/networks/epinions
          http://konect.cc/files/download.tsv.epinions.tar.bz2  (3,205,575 B)

  ``out.epinions``: whitespace-separated, one edge per line, columns
      from_node  to_node  weight(+1 trust / -1 distrust)  unix_timestamp
  841,372 edges, 131,828 nodes, 939 distinct timestamps. Node ids are 1-based.
  Cite: @MISC{konect:2017:epinions} (Koblenz Network Collection).

  NOTE: konect.cc times out on port 443 from this host but serves the same
  file on port 80. Fetch the download path from the network page rather than
  guessing the filename.

FALLBACK SOURCE (two files, joined; for when KONECT is unreachable)
-------------------------------------------------------------------
  SNAP   https://snap.stanford.edu/data/soc-sign-epinions.txt.gz   sign, no time
  4AlexMin/dynamic-networks  human/Recommendation Networks/
         soc-epinion-combined.txt                                    time, no sign
  Their node ids differ by exactly one; after subtracting one the edge key
  sets are identical (841,372/841,372, 0 unmatched either side).

  This fallback was verified to produce output IDENTICAL to KONECT on all
  841,372 edges -- same key set, same sign, same timestamp. It is not a guess
  about provenance but a second route to the same graph, and
  ``--verify-konect`` re-proves it on demand.

REJECTED CANDIDATES (checked, with reasons; do not re-try)
---------------------------------------------------------
  MSU Tang epinions_with_timestamps_11.rar  300,548 TRUST-ONLY rows (no sign
      column) and a different user-id space: only 592 of its 300,091 pairs
      appear in the KONECT graph. Timestamps are 11 coarse buckets, not real
      times. No distrust, so no attack surface.
  MSU Tang epinions_with_rating_timestamp*  timestamps PRODUCT REVIEWS, not
      trust edges; the trust file has 2 columns and no sign.
  Network Repository epinions               archive is a 2-column,
      14,385,987-row file despite the page advertising timestamps.
  NDS-VU signed-network-datasets             README says "Temporal: Yes" but
      epinions.csv is src,dst,sign with no time column.
  RED/CNRS Rich Epinions                     trust only by design ("only
      positive trust values appear in the dataset"); no block list.

TIMESTAMP CAVEAT (state this in the paper)
-------------------------------------------
68.8% of edges (578,996 of 841,372) carry the single crawl timestamp
t0 = 979081200 (2001-01-09). Only the remaining 31.2% carry a true formation
time. This matches the published analysis of the same data (Patil et al.,
"Quantifying Social Influence in Epinions", SocialCom 2013): "about 69% of
trust/distrust edges are observed on the first day of the crawl". Snapshotting
therefore cannot split that block; see epinions_temporal.md.

OUTPUT (same two-file shape as the Bitcoin datasets)
----------------------------------------------------
  data/cyberdata/epinions-dated-rating.txt   TAB: source target trust distrust
  data/cyberdata/epinions-dated.csv          header: source,target,rating,time

Edges are written sorted by (time, source, target), which is what
laundering/core.py:load()'s stable-sort sign-alignment check requires.

Usage
-----
  # primary path
  curl -sSLo /tmp/epinions.tar.bz2 http://konect.cc/files/download.tsv.epinions.tar.bz2
  tar xjf /tmp/epinions.tar.bz2 -C /tmp            # -> /tmp/epinions/out.epinions
  python laundering/build_epinions.py --konect /tmp/epinions/out.epinions

  # fallback path
  python laundering/build_epinions.py \\
      --snap /tmp/soc-sign-epinions.txt \\
      --times /tmp/soc-epinion-combined.txt

  # re-prove the two paths agree
  python laundering/build_epinions.py --verify-konect /tmp/epinions/out.epinions \\
      --snap /tmp/soc-sign-epinions.txt --times /tmp/soc-epinion-combined.txt
"""
import argparse
import collections
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "cyberdata")
RATING_OUT = os.path.join(OUT_DIR, "epinions-dated-rating.txt")
CSV_OUT = os.path.join(OUT_DIR, "epinions-dated.csv")

EXPECT_EDGES = 841372
EXPECT_NODES = 131828
EXPECT_TIMESTAMPS = 939
T0 = 979081200              # crawl timestamp shared by 68.8% of edges
ID_OFFSET = -1             # 1-based (KONECT / dynamic-networks) -> 0-based


def _utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def read_konect(path):
    """KONECT out.epinions -> [(src, dst, sign, time)] sorted by (time, src, dst).

    Node ids are 1-based there, so they are shifted to 0-based to match SNAP
    numbering and the Bitcoin datasets.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("%"):          # '% asym signed' header
                continue
            p = line.split()
            if len(p) < 4:
                continue
            rows.append((int(p[0]) + ID_OFFSET, int(p[1]) + ID_OFFSET,
                         int(p[2]), int(p[3])))
    rows.sort(key=lambda r: (r[3], r[0], r[1]))
    return rows


def _read_snap_signs(path):
    """(src, dst) -> +1 trust / -1 distrust. Skips '#' header lines."""
    signs = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) != 3:
                continue
            signs[(int(p[0]), int(p[1]))] = int(p[2])
    return signs


def _read_times(path):
    """(src, dst) -> unix timestamp."""
    times = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) != 3:
                continue
            times[(int(p[0]), int(p[1]))] = int(p[2])
    return times


def read_join(snap_path, times_path, strict=True):
    """Fallback two-source join.

    ``strict`` re-checks totality: a partial join would silently change the
    edge set, so it raises rather than warns.
    """
    signs = _read_snap_signs(snap_path)
    times = _read_times(times_path)
    shifted = {(a + ID_OFFSET, b + ID_OFFSET): t for (a, b), t in times.items()}
    only_sign = set(signs) - set(shifted)
    only_time = set(shifted) - set(signs)
    if strict and (only_sign or only_time):
        raise AssertionError(
            f"join is not total: {len(only_sign)} sign-only, "
            f"{len(only_time)} time-only edges")
    rows = [(a, b, signs[(a, b)], t) for (a, b), t in shifted.items()]
    rows.sort(key=lambda r: (r[3], r[0], r[1]))
    return rows


def verify_against_konect(rows, konect_path):
    """Assert the given rows are IDENTICAL to KONECT, edge for edge."""
    ref = {(a, b): (s, t) for a, b, s, t in read_konect(konect_path)}
    mine = {(a, b): (s, t) for a, b, s, t in rows}
    if set(ref) != set(mine):
        raise AssertionError(
            f"key sets differ: konect-only {len(set(ref) - set(mine))}, "
            f"join-only {len(set(mine) - set(ref))}")
    bad_sign = sum(1 for k in ref if ref[k][0] != mine[k][0])
    bad_time = sum(1 for k in ref if ref[k][1] != mine[k][1])
    if bad_sign or bad_time:
        raise AssertionError(f"disagreement: {bad_sign} signs, {bad_time} times")
    return dict(edges=len(ref), sign_mismatches=bad_sign,
                time_mismatches=bad_time, identical=True)


def census(rows):
    """Diagnostics that must match the published dataset before we ship it."""
    n = len(rows)
    nodes = set()
    for a, b, _, _ in rows:
        nodes.add(a)
        nodes.add(b)
    pairs = {(a, b) for a, b, _, _ in rows}
    ts = collections.Counter(t for _, _, _, t in rows)
    signs = collections.Counter(s for _, _, s, _ in rows)
    rep = dict(
        edges=n,
        nodes=len(nodes),
        max_id=max(nodes),
        duplicate_pairs=n - len(pairs),
        distinct_timestamps=len(ts),
        t0_edges=ts[T0],
        t0_share=round(ts[T0] / n, 4),
        distinct_signs={k: v for k, v in sorted(signs.items())},
        distrust_share=round(signs[-1] / n, 4),
    )
    if n != EXPECT_EDGES:
        raise AssertionError(f"edge count {n} != {EXPECT_EDGES}")
    if rep["nodes"] != EXPECT_NODES:
        raise AssertionError(f"node count {rep['nodes']} != {EXPECT_NODES}")
    if rep["max_id"] != EXPECT_NODES - 1:
        raise AssertionError(f"max_id {rep['max_id']} != {EXPECT_NODES - 1}")
    if rep["duplicate_pairs"]:
        raise AssertionError(f"{rep['duplicate_pairs']} duplicate (u,v) pairs")
    if set(signs) != {1, -1}:
        raise AssertionError(f"unexpected sign set {sorted(signs)}")
    if rep["distinct_timestamps"] != EXPECT_TIMESTAMPS:
        raise AssertionError(f"distinct timestamps "
                             f"{rep['distinct_timestamps']} != {EXPECT_TIMESTAMPS}")
    return rep


def write(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RATING_OUT, "w") as rt, open(CSV_OUT, "w") as ct:
        ct.write("source,target,rating,time\n")
        for a, b, s, t in rows:
            trust, distrust = (1, 0) if s == 1 else (0, 1)
            rt.write(f"{a}\t{b}\t{trust}\t{distrust}\n")
            ct.write(f"{a},{b},{s},{t}\n")
    return RATING_OUT, CSV_OUT


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--konect", help="KONECT out.epinions (preferred)")
    src.add_argument("--snap", help="SNAP soc-sign-epinions.txt (fallback signs)")
    ap.add_argument("--times", help="soc-epinion-combined.txt (fallback times)")
    ap.add_argument("--verify-konect",
                    help="also check the chosen rows against this KONECT file")
    ap.add_argument("--verify-only", action="store_true",
                    help="print the census, write nothing")
    a = ap.parse_args()
    if a.snap and not a.times:
        ap.error("--snap requires --times")

    if a.konect:
        rows = read_konect(a.konect)
        print(f"[konect] {len(rows)} edges from {a.konect}")
    else:
        rows = read_join(a.snap, a.times)
        print(f"[join   ] {len(rows)} edges from SNAP + dynamic-networks")

    rep = census(rows)
    print("\n--- census ---")
    for k, v in rep.items():
        print(f"  {k:22s} {v}")
    lo = _utc(min(r[3] for r in rows))
    hi = _utc(max(r[3] for r in rows))
    print(f"  {'time range':22s} {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}")

    if a.verify_konect:
        chk = verify_against_konect(rows, a.verify_konect)
        print(f"\n--- cross-check vs KONECT --- {chk}")

    if a.verify_only:
        print("\nverify-only: nothing written")
        return 0
    r1, r2 = write(rows)
    print(f"\nwrote {r1}\n      {r2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())