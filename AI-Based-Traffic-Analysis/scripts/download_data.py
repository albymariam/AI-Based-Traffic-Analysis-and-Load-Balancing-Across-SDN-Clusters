"""Discover and download the REAL Abilene TM files from the official index page; write the manifest.
   python scripts/download_data.py            # all weeks + metadata
   python scripts/download_data.py --weeks 1-4  # quick subset
Nothing is hard-coded: file names come from the page's own links."""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
WANT = re.compile(r"^(X\d\d\.gz|A|demands|links|readme\.txt|topo-.*\.txt)$")
MANUAL = """Automated download failed. Manual steps:
  1. Open {base} in a browser.
  2. Save X01.gz..X24.gz, A, demands, links, readme.txt, topo-2003-04-10.txt (or AbileneTM-all.tar and extract)
  3. Put them in {dest}
  4. python scripts/download_data.py --manifest-only"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "abilene.yaml"))["data"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="1-99")
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.weeks.split("-"))
    dest = ROOT / cfg["raw_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    base, entries = cfg["base_url"], []
    try:
        if not a.manifest_only:
            html = requests.get(base, timeout=30)
            html.raise_for_status()
            names = sorted({n for n in re.findall(r'href="([^"?/][^"]*)"', html.text) if WANT.match(n)})
            if not names:
                raise RuntimeError("no expected files found on index page")
            for n in names:
                m = re.match(r"X(\d\d)\.gz", n)
                if m and not lo <= int(m.group(1)) <= hi:
                    continue
                url, out = base + n, dest / n
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                want = int(r.headers.get("Content-Length", -1))
                if out.exists() and out.stat().st_size == want:
                    print("have", n)
                    continue
                with open(out.with_suffix(out.suffix + ".part"), "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
                got = out.with_suffix(out.suffix + ".part").stat().st_size
                if want >= 0 and got != want:
                    raise RuntimeError("%s incomplete: %d of %d bytes" % (n, got, want))
                out.with_suffix(out.suffix + ".part").rename(out)
                print("got ", n, got)
    except Exception as e:
        print("ERROR:", repr(e), file=sys.stderr)
        print(MANUAL.format(base=base, dest=dest), file=sys.stderr)
        sys.exit(2)

    for p in sorted(dest.iterdir()):
        if WANT.match(p.name):
            entries.append({"file": p.name, "url": base + p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
                            "retrieved_utc": dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat()})
    terms = []
    rd = dest / "readme.txt"
    if rd.exists():
        terms = [l.strip() for l in rd.read_text(errors="ignore").splitlines()
                 if re.search(r"cite|citation|reference|copyright|licen[sc]e|permission|redistribut", l, re.I)]
    man = {"dataset": "Abilene/Internet2 traffic matrices", "source_page": base, "files": entries,
           "licence": "NOT STATED by the downloader. Lines in readme.txt that may govern reuse are listed in "
                      "'readme_reuse_lines'; read readme.txt and record the terms here. Do not redistribute raw files.",
           "readme_reuse_lines": terms}
    out = ROOT / "data" / "manifests" / "dataset_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2))
    print("manifest:", out, "|", len(entries), "files")
    if terms:
        print("readme lines about citation/reuse:\n  " + "\n  ".join(terms))


if __name__ == "__main__":
    main()
