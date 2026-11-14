"""Build the Wiki-RfA signed, timestamped vote network (proposal §6 dataset).

Source: SNAP, "Wikipedia Requests for Adminship (with text)",
https://snap.stanford.edu/data/wiki-RfA.html (West, Paskov, Leskovec, Potts,
TACL 2014). Each record is one vote:

    SRC:<voter>  TGT:<candidate>  VOT:<1 | 0 | -1>  RES:<1 | -1>  YEA:<year>
    DAT:<hh:mm, d Month yyyy>  TXT:<comment>

Output, in the repository's signed format:
    data/cyberdata/wiki-rfa.csv     source,target,rating,time   (int ids, unix UTC)

The rating is the vote (+1 support, -1 oppose, 0 neutral). Neutral votes are KEPT
in the CSV; the sign-aware loader drops and counts them under the dataset's
zero policy (amendment A4), so the policy stays visible in every result. Records
without a parseable date cannot be placed in time and are dropped here, with the
count printed and written to the provenance file. Usernames are mapped to
integer ids in sorted order (deterministic). Repeated votes by the same voter on
the same candidate (re-nominations) are separate events and are all kept.

Usage:
  <ps python> laundering/build_wikirfa.py [--src wiki-RfA.txt.gz] [--download]
"""
import argparse
import calendar
import collections
import gzip
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URL = "https://snap.stanford.edu/data/wiki-RfA.txt.gz"
OUT_CSV = os.path.join(ROOT, "data", "cyberdata", "wiki-rfa.csv")
OUT_META = os.path.join(ROOT, "data", "cyberdata", "wiki-rfa.provenance.json")
MONTHS = {m: i for i, m in enumerate(calendar.month_name) if m}
DAT_RE = re.compile(r"(\d{1,2}):(\d{2}),\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})")


def parse_date(s: str):
    m = DAT_RE.search(s or "")
    if not m:
        return None
    hh, mm, d, mon, y = m.groups()
    mon = MONTHS.get(mon.capitalize())
    if mon is None:
        return None
    try:
        return calendar.timegm((int(y), mon, int(d), int(hh), int(mm), 0))
    except ValueError:
        return None


def records(path):
    rec = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if rec:
                    yield rec
                    rec = {}
                continue
            k, _, v = line.partition(":")
            if k in ("SRC", "TGT", "VOT", "RES", "YEA", "DAT"):
                rec[k] = v.strip()
    if rec:
        yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("/tmp", "wiki-RfA.txt.gz"))
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download or not os.path.exists(a.src):
        print(f"downloading {URL} -> {a.src}")
        urllib.request.urlretrieve(URL, a.src)
    rows, dropped = [], collections.Counter()
    for r in records(a.src):
        if not r.get("SRC") or not r.get("TGT"):
            dropped["missing_endpoint"] += 1
            continue
        try:
            vot = int(r["VOT"])
        except (KeyError, ValueError):
            dropped["bad_vote"] += 1
            continue
        t = parse_date(r.get("DAT", ""))
        if t is None:
            dropped["no_parseable_date"] += 1
            continue
        rows.append((r["SRC"], r["TGT"], vot, t))
    names = sorted({x for s, d, _, _ in rows for x in (s, d)})
    idx = {n: i for i, n in enumerate(names)}
    rows.sort(key=lambda x: (x[3], idx[x[0]], idx[x[1]]))
    with open(OUT_CSV, "w") as f:
        f.write("source,target,rating,time\n")
        for s, d, v, t in rows:
            f.write(f"{idx[s]},{idx[d]},{v},{t}\n")
    votes = collections.Counter(v for _, _, v, _ in rows)
    meta = dict(source=URL, citation="West, Paskov, Leskovec, Potts. Exploiting Social "
                "Network Structure for Person-to-Person Sentiment Analysis. TACL 2014.",
                built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                n_rows=len(rows), n_nodes=len(names), dropped=dict(dropped),
                votes={str(k): int(v) for k, v in sorted(votes.items())},
                time_min=min(r[3] for r in rows), time_max=max(r[3] for r in rows),
                distinct_times=len({r[3] for r in rows}),
                self_votes=sum(1 for s, d, _, _ in rows if s == d))
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
