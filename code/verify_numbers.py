#!/usr/bin/env python3
"""Recompute every economic number of P4 from p4_economics.py and check that
it appears, as printed, in P4_Systems.tex. Also runs style checks."""
import re, sys
from pathlib import Path
import numpy as np
import p4_economics as E

_here = Path(__file__).resolve().parent
_cands = [Path(a) for a in sys.argv[1:]] + [_here / "P4_Systems.tex", _here.parent / "P4_Systems.tex"]
_tex = next((p for p in _cands if p.exists()), None)
if _tex is None:
    sys.exit("Manuscript source not found; pass its path as an argument.")
tex = _tex.read_text()
flat = re.sub(r"\s+", " ", tex)
fails = []


def has(s, why):
    if s not in flat:
        fails.append(f"NOT FOUND [{why}]: {s}")


r = E.results()
for i in range(5):
    row = (f"$N = {i+1}$ & {E.CAPEX[i]:.0f} & {r['netcf'][i]:.0f} & {r['npv'][i]/1000:.2f} & "
           f"{r['irr'][i]:.0f} & {r['payback'][i]:.1f}")
    has(row, "Table npv")
    has(f"{E.UNITS_SHIFT[i]} & {'Single' if E.BASE_SCHED[i]=='single' else 'Dual'} & {r['units_year'][i]:.0f} & "
        f"{r['revenue_m'][i]:.2f} & {r['gross_k'][i]:.0f} & {r['manual_gross'][i]:.0f} & {r['delta_gross'][i]:.0f}", "Table revenue")
has(" & ".join(f"\\textbf{{{v:.1f}}}" for v in E.OPEX_SINGLE), "opex single total")
has(" & ".join(f"\\textbf{{{v:.1f}}}" for v in E.OPEX_BASE), "opex base total")
has(" & ".join(f"\\textbf{{{v:.0f}}}" for v in E.CAPEX), "capex total")
for name, (vals, _) in E.OPEX_ITEMS.items():
    has(" & ".join(f"{v:.1f}" for v in vals), f"opex item {name}")
for name, vals in E.CAPEX_ITEMS.items():
    has(" & ".join(f"{v:.0f}" for v in vals), f"capex item {name}")
assert np.allclose(np.array(E.OPEX_ITEMS["Preventive maintenance"][0]), np.round(0.08 * E.CAPEX, 1), atol=0.05)

ms = E.multishift()
for i in range(3):
    has(f"$N = {i+1}$ & " + " & ".join(f"{a:.2f} & {b:.1f}" for a, b, _ in ms[i]), "Table multishift")
ratio = [ms[i][1][0] / ms[i][0][0] for i in range(3)]
red = [100 * (1 - ms[i][1][1] / ms[i][0][1]) for i in range(3)]
has(f"{min(ratio):.1f}--{max(ratio):.1f}", "dual/single NPV ratio")
has(f"{min(red):.0f}--{max(red):.0f}\\%", "payback reduction")

for i, (v, p, _) in enumerate(E.breakeven()):
    has(f"$N = {i+1}$ & {'Single' if E.BASE_SCHED[i]=='single' else 'Dual'} & {v:.0f} & {p:.0f}", "Table breakeven")
has(f"about {E.breakeven()[1][1]/100*192*8:.0f} operating hours", "break-even hours")

base, sens = E.sensitivity()
lab = {"Labor reallocation": "Redeployment allowance"}
for k, (lo, hi) in sens.items():
    has(f"{lab.get(k,k)} & {lo:.2f} & {hi:.2f} & {abs(hi-lo):.2f}", "Table sensitivity")
has(f"removes {100*(1-sens['Production volume'][0]/base):.0f}\\% of NPV", "volume -20%")

npv, X = E.monte_carlo(); s = E.mc_summary(npv)
has(f"\\${s['p5']:.2f}M, \\${s['p25']:.2f}M, \\${s['p50']:.2f}M, \\${s['p75']:.2f}M and \\${s['p95']:.2f}M", "MC percentiles")
has(f"[\\${s['p5']:.2f}M, \\${s['p95']:.2f}M]", "MC interval")
has(f"{s['p_neg']:.1f}\\%", "P(neg)")
has(f"{s['p_gt05']:.1f}\\% and {s['p_gt1']:.1f}\\%", "P(>0.5,>1)")
sh, r2 = E.variance_shares(npv, X)
for v in sh:
    has(f"{100*v:.1f}\\%", "variance share")
has(f"{100*r2:.1f}\\% of its variance", "R2")
nc, _ = E.monte_carlo(rho_vp=-0.5); sc = E.mc_summary(nc)
has(f"[\\${sc['p5']:.2f}M, \\${sc['p95']:.2f}M]", "correlated interval"); has(f"falls to {sc['p_neg']:.1f}\\%", "correlated P(neg)")
for cv in (0.10, 0.25):
    n3, _ = E.monte_carlo(cv_volume=cv); has(f"{E.mc_summary(n3)['p_neg']:.1f}\\% at", f"cv {cv}")
has(f"standard deviation {E.MC_OPEX_SD_FRAC*E.OPEX_BASE[1]:.1f}", "opex sd")

scn, ev = E.scenarios()
for k, lab2 in (("Pessimistic", "Downturn"), ("Base", "Base case"), ("Optimistic", "Growth")):
    v = scn[k]
    has(f"& {v['npv']:.2f} & {v['irr']:.0f} & {v['pb']:.1f}", f"scenario {k}")
has(f"expected NPV of \\${ev:.2f}M", "scenario EV")
dn, dp, di = E.disruption_adjusted()
has(f"{dn/1000:.2f} & {di:.0f} & {dp:.1f}", "disrupted row")
for w, (n, p) in E.labor_release().items():
    has(f"{n:.2f} &", f"labor release {w}")
qs1, qs2 = E.quality_savings("single"), E.quality_savings("dual")
has(f"\\${qs1:.1f}$k" if False else f"{qs1:.1f}", "quality single"); has(f"{qs2:.1f}k", "quality dual")


# derived quantities quoted in the text
hw = np.sum([E.CAPEX_ITEMS[k] for k in E.HARDWARE], axis=0) / E.CAPEX * 100
rob = np.array(E.CAPEX_ITEMS["Robots (UR5e)"]) / E.CAPEX * 100
has(f"Hardware accounts for {hw[0]:.0f}\\% of capital at $N = 1$ and {hw[-1]:.0f}\\% at $N = 5$", "hardware share")
has(f"the robots alone account for {rob[0]:.0f}--{rob[-1]:.0f}\\%", "robot share")
has(f"hardware accounts for {hw[0]:.0f}--{hw[-1]:.0f}\\%", "Fig cost caption")
marg = np.diff(E.CAPEX); has(f"Each added robot costs \\${marg.min():.0f}--{marg.max():.0f}k", "marginal capex")
pr = E.CAPEX / E.N_RANGE
has(f"from \\${pr[0]:.0f}k at $N = 1$ to \\${pr[-1]:.0f}k at $N = 5$", "capex per robot")
has(f"(\\${pr[-1]:.0f}--{pr[2]:.0f}k per robot)", "capex per robot N>=3")
cont = np.array(E.CAPEX_ITEMS["Contingency"]) / (E.CAPEX - np.array(E.CAPEX_ITEMS["Contingency"])) * 100
has(f"{cont.min():.0f}--{cont.max():.0f}\\%", "contingency share")
op = E.OPEX_ITEMS
lab = (np.array(op["Technician time"][0]) + op["Supervision time"][0]) / E.OPEX_SINGLE * 100
mt = (np.array(op["Preventive maintenance"][0]) + op["Spare parts"][0] + np.array(op["Calibration"][0])) / E.OPEX_SINGLE * 100
en = (np.array(op["Electricity"][0]) + op["HVAC"][0]) / E.OPEX_SINGLE * 100
has(f"make up {lab.min():.0f}--{lab.max():.0f}\\% of operating cost and maintenance {mt.min():.0f}--{mt.max():.0f}\\%; energy is {en.min():.0f}--{en.max():.0f}\\%", "opex shares")
tf = np.array(op["Technician time"][0]) / 75; sf = np.array(op["Supervision time"][0]) / 80
has(f"{tf.min():.2f}--{tf.max():.2f} full-time", "technician FTE"); has(f"{sf.min():.2f}--{sf.max():.2f} FTE", "supervision FTE")
inc = np.diff(E.UNITS_SHIFT); has(f"{inc[0]} and {inc[1]}", "increments 2-3"); has(f"{inc[2]} and {inc[3]}", "increments 4-5")
has(f"its {E.UNITS_SHIFT[0]-E.MANUAL_UNITS} added units", "N1 increment")
has(f"by {E.UNITS_SHIFT[1]-E.MANUAL_UNITS};", "N2 increment"); has(f"by {0.2*E.UNITS_SHIFT[1]:.0f} units", "20% of N2 output")
has(f"$S_q = \\${qs1:.1f}$k per year on one shift and \\${qs2:.1f}k on two", "quality savings")
for k, lab2 in (("Pessimistic", "\\$0.05M"),):
    has(f"barely positive (\\${scn[k]['npv']:.2f}M) with a {scn[k]['irr']:.0f}\\% IRR and a payback of {scn[k]['pb']:.1f} months", "downturn text")
has(f"from \\$0.69M to \\${dn/1000:.2f}M and the payback lengthens from {r['payback'][1]:.1f} to {dp:.1f} months", "disruption text")
lr = E.labor_release(); has(f"\\${lr[28.0][0]:.2f}M and \\${lr[45.0][0]:.2f}M", "labor release text")
cx = sens["Capital expenditure"][1]; has(f"by only \\${base-cx:.2f}M", "capex overrun")
if not (base - cx) / base < 0.05: fails.append("capex overrun claim (<5%) false")
# ------------------------------------------------- Systems version: new claims
import p4_systems as S
tests, mc = S.verification_tests()
if not all(d <= t for _, d, t in tests): fails.append("a verification check fails")
an = max(d for n, d, t in tests[:6]); has("to within $10^{-11}$\\,k\\$", "analytic tol")
if an >= 1e-11: fails.append("analytic deviation >= 1e-11")
if tests[7][1] >= 1e-12: fails.append("re-implementation deviation >= 1e-12")
has(f"change by {tests[8][1]:.2f} percentage points", "MC conv pp")
if tests[9][1] >= 1.0: fails.append("MC p5 change >= 1k")
has(f"the loss probability is {mc['p_neg_100k']:.2f}\\% and the 5th and 95th percentiles are \\${mc['p5_100k']:.2f}M and \\${mc['p95_100k']:.2f}M", "MC 100k")
cap = E.mc_summary(S.monte_carlo_capped())
has(f"the mean falls to \\${cap['mean']:.2f}M, the 90\\% interval becomes [\\${cap['p5']:.2f}M, \\${cap['p95']:.2f}M]", "capped MC")
has(f"negligible ({cap['p_gt1']:.2f}\\%)", "capped >1M"); has(f"unchanged at {cap['p_neg']:.1f}\\%", "capped loss")
cases = S.published_cases()
for c in cases:
    if c["key"] != "polonara2024" and c["max_dev"] > 0.005: fails.append(f"case not reproduced: {c['key']}")
pi = [c["pi"] for c in cases]; pb = [c["pb"] for c in cases]
has(f"indices of {pi[0]:.2f}, {pi[1]:.2f} and {pi[2]:.2f} and paybacks of {pb[0]:.1f}, {pb[1]:.1f} and {pb[2]:.1f} months", "case PI")
own = S.own_designs_common()
has(f"indices of {min(d['pi'] for d in own):.2f}--{max(d['pi'] for d in own):.2f} and paybacks of {min(d['pb'] for d in own):.1f}--{max(d['pb'] for d in own):.1f} months", "own PI")
has(f"(index {own[1]['pi']:.2f}, {own[1]['pb']:.1f} months)", "N2 PI")
has(f"implies {cases[0]['simple_pb']:.1f} months", "martin simple pb")
has(f"break-even of {cases[1]['recomputed'][0]:,.0f} parts".replace(",", "{,}"), "polonara be")
cpr = [c["capex_per_robot"] / 1000 for c in cases]
lo, hi = E.CAPEX[-1] / 5 / max(cpr), E.CAPEX[0] / min(cpr)
has(f"{lo:.1f}--{hi:.1f} times", "capex per robot ratio")
has("about eleven times the USD 4407.58", "opex ratio text")
if round(E.OPEX_SINGLE[0] * 1000 / cases[0]["opex"]) != 11: fails.append("opex ratio not eleven")
if round(100 * (E.DEFECT_MANUAL - E.DEFECT_COBOT) / E.DEFECT_MANUAL) != 81: fails.append("defect reduction 81%")
sv, ex = S.switching_values()
has(f"operating cost can rise by {100 * sv['Operating cost'][3]:.0f}\\% and capital by {100 * sv['Capital expenditure'][3]:.0f}\\%", "switch cost")
has(f"internal rate of return of {sv['Discount rate'][0]:.1f}\\%", "switch IRR")
has(f"the NPV is \\${ex['npv_no_qs'] / 1000:.2f}M and \\${ex['npv_no_ls'] / 1000:.2f}M, and without both it is \\${ex['npv_no_qs_ls'] / 1000:.2f}M", "switch qs ls")
has(f"cell output can fall by {-100 * sv['Cell output (digital-twin throughput)'][3]:.0f}\\% and the unit margin by {-100 * sv['Unit margin (part value x gross margin)'][3]:.0f}\\%", "switch rev")
has(f"produce {100 * sv['Manual comparator output'][3]:.0f}\\% more", "switch manual")
has(f"at least {sv['Cell output (digital-twin throughput)'][0]:.0f} units per shift against a manual line of no more than {sv['Manual comparator output'][0]:.0f}", "pilot criteria")
if round(1 + sv["Capital expenditure"][3]) != 5: fails.append("capital fivefold claim")
tt = S.twin_tolerance()
has(f"overstated by {tt[0]['cell_overstatement']:.0f}\\% for one robot, {tt[1]['cell_overstatement']:.0f}\\% for two and "
    f"{min(t['cell_overstatement'] for t in tt[2:]):.0f}--{max(t['cell_overstatement'] for t in tt[2:]):.0f}\\%", "twin tol")
has("understated by " + ", ".join(f"{t['manual_understatement']:.0f}\\%" for t in tt[:4]) + f" and {tt[4]['manual_understatement']:.0f}\\%", "manual tol")
if not (tt[0]["cell_overstatement"] < 20 and all(t["cell_overstatement"] >= 20 for t in tt[1:])): fails.append("only N=1 below 20%")
d, grid, best, bv, bands, mind = S.demand_map()
b = {(None if k is None else (k[0] + 1, k[1])): (lo_, hi_) for lo_, hi_, k, *_ in bands}
if b[None][1] + 1000 != 50000: fails.append("no-invest threshold")
if b[(2, "single")] != (73000, 120000): fails.append("N2 band")
if b[(2, "dual")][0] != 210000 or b[(5, "single")][1] != 209000: fails.append("dual threshold")
if b[(3, "triple")][0] != 417000: fails.append("triple threshold")
if b[(3, "dual")] != (233000, 327000): fails.append("N3 dual band")
has(f"three robots on one shift earn \\${S.npv_demand(2, 'single', 154000) / 1000:.2f}M and two robots on two shifts \\${S.npv_demand(1, 'dual', 154000) / 1000:.2f}M", "154k comparison")
has(f"two robots on one shift make {S.capacity(1, 'single'):,.0f} units".replace(",", "{,}"), "cap N2"); has(f"make {S.capacity(2, 'dual'):,.0f} units".replace(",", "{,}"), "cap N3 dual")
lr = E.labor_release(); has(f"by \\${lr[28.0][0] - r['npv'][1] / 1000:.2f}--{lr[45.0][0] - r['npv'][1] / 1000:.2f}M", "labor release delta")

# abstract
ab_txt = re.search(r"\\abstract\{(.*?)\}\n", tex, re.S).group(1)
for s_ in (f"\\${r['npv'][0]/1000:.2f}M, {r['payback'][0]:.1f}-month payback", f"\\${r['npv'][1]/1000:.2f}M with a {s['p_neg']:.1f}\\% probability",
           f"explains {100*sh[0]:.0f}\\%", f"throughput is {-100 * sv['Cell output (digital-twin throughput)'][3]:.0f}\\% lower",
           "below 50,000", "between 73,000 and 120,000", "only above 210,000", f"{len(tests)} verification checks".replace("10", "ten")):
    if s_.replace(",", "{,}") not in ab_txt and s_ not in ab_txt: fails.append(f"abstract missing: {s_}")

# style checks
body = tex[tex.find("\\begin{document}"):tex.find("\\begin{thebibliography}")]
for w in re.findall(r"\b(we|our|us|We|Our|Us)\b", re.sub(r"US dollars", "", body)):
    fails.append(f"first person: {w}")
for w in ("delve", "pivotal", "crucial", "landscape", "realm", "underscore", "leverage", "seamless",
          "comprehensive", "intricate", "multifaceted", "notably", "Moreover", "Furthermore",
          "It is worth noting", "validated", "modelled", "labour", "favour", "behaviour", "analyse"):
    for m in re.finditer(r"\b" + w + r"\b", body):
        fails.append(f"style word '{w}' at: ...{body[max(0,m.start()-40):m.end()+40]}...")
ab = re.search(r"\\abstract\{(.*?)\}\n", tex, re.S).group(1)
nwords = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", ab).split())
if nwords > 200:
    fails.append(f"abstract has {nwords} words")
print(f"abstract words: {nwords}")
print("\n".join(fails) if fails else "ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
