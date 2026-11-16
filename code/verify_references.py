#!/usr/bin/env python3
"""Check every DOI of the reference list against Crossref: the DOI resolves,
and title, year, volume and first page or article number agree with the entry."""
import difflib, json, re, sys, time, urllib.request
from pathlib import Path

here = Path(__file__).resolve().parent
cands = [Path(a) for a in sys.argv[1:]] + [here / "P4_Machines.tex", here.parent / "P4_Machines.tex"]
tex = next(p for p in cands if p.exists()).read_text()
bib = tex[tex.find("\\begin{thebibliography}"):]
items = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", bib, re.S)


def clean(s):
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"[{}\\~'^`\"]", "", s).lower())


bad = 0
for n, (key, entry) in enumerate(items, 1):
    e = " ".join(entry.split())
    m = re.search(r"doi\.org/(\S+?)\.?$", e)
    if not m:
        print(f"[{n}] {key}: no DOI (standard or book) - checked manually")
        continue
    doi = m.group(1).replace("\\_", "_")
    try:
        msg = json.load(urllib.request.urlopen(f"https://api.crossref.org/works/{doi}", timeout=30))["message"]
    except Exception as ex:
        print(f"[{n}] {key}: DOI FAILED {doi} ({ex})"); bad += 1; continue
    title = (msg.get("title") or [""])[0]
    year = str(((msg.get("published-print") or msg.get("issued") or {}).get("date-parts") or [[None]])[0][0])
    years = {str(((msg.get(k) or {}).get("date-parts") or [[None]])[0][0]) for k in ("issued", "published-print", "published-online")}
    vol, page = msg.get("volume"), msg.get("page") or msg.get("article-number")
    t_bib = re.search(r"\.\s(.*?)\.\s\\textit", entry, re.S)
    ratio = difflib.SequenceMatcher(None, clean(title), clean(e), autojunk=False).find_longest_match(0, len(clean(title)), 0, len(clean(e))).size / max(1, len(clean(title)))
    ybib = re.search(r"\\textbf\{(\d{4})\}", entry)
    issues = []
    if ratio < 0.85: issues.append(f"title? '{title[:70]}'")
    if ybib and ybib.group(1) not in years: issues.append(f"year {ybib.group(1)} vs {sorted(years)}")
    if vol and f"\\textit{{{vol}}}" not in entry: issues.append(f"volume {vol}")
    if page:
        first = str(page).split("-")[0]
        if first not in entry: issues.append(f"page {page}")
    print(f"[{n}] {key}: {'OK' if not issues else 'CHECK ' + '; '.join(issues)}")
    bad += bool(issues)
    time.sleep(1.2)
print(f"\n{len(items)} references, {bad} to check")
