"""Check every reference with a DOI in the paper against Crossref (year, volume, issue, pages, first author, title).

Usage:  python3 verify_refs.py Paper.docx            # needs internet; python-docx only for reading
Prints one line per reference: OK, or the fields that differ. References without a DOI are listed at the end.
"""
import re, sys, json, difflib, urllib.request, urllib.parse
import docx

def refs(path):
    d = docx.Document(path)
    out = {}
    for p in d.paragraphs:
        m = re.match(r"\[(\d+)\]\s+(.*)", p.text.strip())
        if m and len(m.group(2)) > 60:
            out[int(m.group(1))] = m.group(2)
    return out

def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check (mailto:albymariam15@gmail.com)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["message"]

def norm(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def check(n, text):
    m = re.search(r"https://doi.org/(\S+)", text)
    if not m:
        return None
    doi = m.group(1).rstrip(".")
    try:
        c = crossref(doi)
    except Exception as e:
        return f"[{n}] DOI {doi}: lookup failed ({e})"
    bad = []
    year = (c.get("issued", {}).get("date-parts") or [[None]])[0][0]
    ym = re.search(r"\((\d{4})\)", text)
    if ym and year and int(ym.group(1)) != year:
        bad.append(f"year {ym.group(1)} vs {year}")
    au = (c.get("author") or [{}])[0].get("family", "")
    if au and norm(au).split()[-1] not in norm(text.split("(")[0]):
        bad.append(f"first author {au}")
    title = (c.get("title") or [""])[0]
    # title is the part after the year, before the venue
    body = text.split(").", 1)[-1]
    sim = difflib.SequenceMatcher(None, norm(title), norm(body[:len(title) + 20])).ratio()
    if sim < 0.8:
        bad.append(f"title differs: Crossref says '{title[:70]}'")
    vol, iss, pg = c.get("volume"), c.get("issue"), c.get("page") or c.get("article-number")
    if vol and not re.search(r"\b%s\b" % re.escape(str(vol)), body):
        bad.append(f"volume {vol}")
    if pg:
        pgn = str(pg).replace("–", "-")
        a = pgn.split("-")[0]
        if a not in body.replace("–", "-"):
            bad.append(f"pages {pg}")
    return f"[{n}] OK" if not bad else f"[{n}] CHECK: " + "; ".join(bad)

if __name__ == "__main__":
    R = refs(sys.argv[1])
    nodoi = []
    for n in sorted(R):
        r = check(n, R[n])
        if r is None:
            nodoi.append(n)
        else:
            print(r)
    print("\nNo DOI in the entry (check by hand):", nodoi)
