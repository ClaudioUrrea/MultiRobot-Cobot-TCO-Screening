#!/usr/bin/env python3
"""
p4_systems.py -- credibility and system-design analyses added for the Systems
version of the article (C. Urrea, 2026). Builds on p4_economics.py.

1. verification_tests(): computerized-model verification (closed forms,
   extreme conditions, homogeneity, monotonicity, an independent
   re-implementation and Monte Carlo convergence).
2. published_cases(): replication of the arithmetic of three independently
   reported industrial cobot cases and their restatement on the appraisal
   basis of this article.
3. switching_values(): the value of each input at which the N = 2 NPV
   reaches zero, and the tolerance of each configuration to an overstated
   digital-twin throughput or an understated manual comparator.
4. demand_map(): NPV of every (N, schedule) design when sales are capped by
   annual demand, and the design that maximizes NPV at each demand level.
"""
import csv
from pathlib import Path

import numpy as np

import p4_economics as E

SCHEDS = ("single", "dual", "triple")


# ------------------------------------------------------------------ helpers
def _bisect(f, lo, hi, n=200):
    flo = f(lo)
    for _ in range(n):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


def npv_common(capex, benefit, opex, ramp=True):
    """NPV, IRR and payback of a generic cobot project on this article's basis.
    capex, benefit and opex in thousands of currency units."""
    cf = E.cash_flows(1, dg=benefit, capex=capex, opex=opex, qs=0.0, ls=0.0)
    if not ramp:
        cf = cf.copy()
    return E.npv_of(cf, capex), E.irr_of(cf, capex), E.payback_of(cf, capex)


# ------------------------------------------------------ 1. verification tests
def verification_tests():
    tests = []
    i, s = 1, "single"
    cx, ox = E.CAPEX[i], E.opex_for(i, s)
    p = E.delta_gross(i, s) + E.quality_savings(s) + E.LABOR_REALLOC - ox
    r, T = E.DISCOUNT, E.HORIZON
    ann = (1 - (1 + r) ** -T) / r

    # T1 closed form: no tax, no escalation, no ramp, no salvage
    sv = (E.TAX_RATE, E.OPEX_ESC, E.YEAR1_RAMP, E.SALVAGE_FRAC)
    E.TAX_RATE, E.OPEX_ESC, E.YEAR1_RAMP, E.SALVAGE_FRAC = 0.0, 0.0, 1.0, 0.0
    eng = E.npv_of(E.cash_flows(i, tax=0.0), cx)
    E.TAX_RATE, E.OPEX_ESC, E.YEAR1_RAMP, E.SALVAGE_FRAC = sv
    cf_closed = -cx + p * ann
    tests.append(("Closed-form annuity (no tax, escalation, ramp or salvage)", abs(eng - cf_closed), 1e-6))

    # T2 closed form with tax and depreciation shield (profitable every year)
    sv = (E.OPEX_ESC, E.YEAR1_RAMP)
    E.OPEX_ESC, E.YEAR1_RAMP = 0.0, 1.0
    eng = E.npv_of(E.cash_flows(i), cx)
    E.OPEX_ESC, E.YEAR1_RAMP = sv
    after = p - E.TAX_RATE * (p - cx / T)
    closed = -cx + after * ann + E.SALVAGE_FRAC * cx / (1 + r) ** T
    tests.append(("Closed form with tax and straight-line depreciation", abs(eng - closed), 1e-6))

    # T3 IRR consistency
    cf = E.cash_flows(i)
    tests.append(("NPV evaluated at the computed IRR", abs(E.npv_of(cf, cx, E.irr_of(cf, cx) / 100)), 1e-3))

    # T4 payback consistency: cumulative monthly flow at payback equals zero
    pb = E.payback_of(cf, cx)
    cum = E.cumulative_monthly(i, months=int(np.ceil(pb)) + 1)
    m = int(np.floor(pb))
    at_pb = cum[m] + (cum[m + 1] - cum[m]) * (pb - m)
    tests.append(("Cumulative cash flow at the computed payback month", abs(at_pb), 1e-6))

    # T5 extreme condition: no benefit -> NPV equals discounted after-tax cost
    cf0 = E.cash_flows(i, dg=0.0, qs=0.0, ls=0.0)
    pv = sum(-ox * (1 + E.OPEX_ESC) ** (y - 1) / (1 + r) ** y for y in range(1, T + 1))
    closed0 = -cx + pv + E.SALVAGE_FRAC * cx / (1 + r) ** T
    tests.append(("Zero-benefit case equals discounted operating cost", abs(E.npv_of(cf0, cx) - closed0), 1e-6))

    # T6 homogeneity of degree one in all monetary inputs
    k = 3.7
    a = E.npv_of(E.cash_flows(i), cx)
    b = E.npv_of(E.cash_flows(i, dg=k * E.delta_gross(i, s), capex=k * cx, opex=k * ox,
                             qs=k * E.quality_savings(s), ls=k * E.LABOR_REALLOC), k * cx)
    tests.append(("Homogeneity: scaling every monetary input by 3.7", abs(b - k * a), 1e-6))

    # T7 monotonicity over a grid
    g = np.linspace(0.5, 1.5, 21)
    mono = [
        np.all(np.diff([E.npv_of(E.cash_flows(i, dg=E.delta_gross(i, s, units_shift=E.UNITS_SHIFT[i] * f)), cx) for f in g]) >= 0),
        np.all(np.diff([E.npv_of(E.cash_flows(i, dg=E.delta_gross(i, s, value=E.PART_VALUE * f)), cx) for f in g]) >= 0),
        np.all(np.diff([E.npv_of(E.cash_flows(i, capex=cx * f), cx * f) for f in g]) <= 0),
        np.all(np.diff([E.npv_of(E.cash_flows(i, opex=ox * f), cx) for f in g]) <= 0),
        np.all(np.diff([E.npv_of(E.cash_flows(i), cx, discount=r * f) for f in g]) <= 0),
    ]
    tests.append(("Monotonicity in volume, price, capital, operating cost, discount rate", float(not all(mono)), 0.5))

    # T8 independent re-implementation (vectorized, from the raw item tables)
    dev = 0.0
    for j, sj in enumerate(E.BASE_SCHED):
        dev = max(dev, abs(_independent_npv(j, sj) - E.npv_of(E.cash_flows(j), E.CAPEX[j])))
    tests.append(("Independent vectorized re-implementation, all five designs", dev, 1e-6))

    # T9 Monte Carlo convergence: 10,000 vs 100,000 draws with another seed
    n1, _ = E.monte_carlo()
    n2, _ = E.monte_carlo(n_sim=100_000, seed=2026)
    s1, s2 = E.mc_summary(n1), E.mc_summary(n2)
    tests.append(("Monte Carlo: loss probability, 10k (seed 42) vs 100k (seed 2026), points",
                  abs(s1["p_neg"] - s2["p_neg"]), 0.5))
    tests.append(("Monte Carlo: 5th percentile, 10k vs 100k draws, k$", abs(s1["p5"] - s2["p5"]) * 1000, 15.0))
    return tests, dict(p_neg_10k=s1["p_neg"], p_neg_100k=s2["p_neg"], p5_10k=s1["p5"], p5_100k=s2["p5"],
                       p95_10k=s1["p95"], p95_100k=s2["p95"])


def _independent_npv(j, sched):
    """Second implementation: matrix form, items read again from the CSV files."""
    rows = list(csv.DictReader(open(E.DATA / "capex_items_kUSD.csv")))
    capex = sum(float(rw[f"N{j + 1}"]) for rw in rows)
    rows = list(csv.DictReader(open(E.DATA / "opex_items_single_shift_kUSD_per_year.csv")))
    factor = E.SHIFTS[sched] / E.SHIFTS["single"]
    opex = sum(float(rw[f"N{j + 1}"]) * (factor if rw["scales_with_operating_hours"] == "1" else 1.0) for rw in rows)
    sh = E.SHIFTS[sched]
    benefit = ((E.UNITS_SHIFT[j] - E.MANUAL_UNITS) * sh * E.PART_VALUE * E.GROSS_MARGIN
               + (E.DEFECT_MANUAL - E.DEFECT_COBOT) * E.WARRANTY * E.MANUAL_UNITS * sh) / 1000 + E.LABOR_REALLOC
    y = np.arange(1, E.HORIZON + 1)
    ramp = np.where(y == 1, (sum(E.RAMP) + 12 - len(E.RAMP)) / 12, 1.0)
    pre = benefit * ramp - opex * (1 + E.OPEX_ESC) ** (y - 1)
    post = pre - E.TAX_RATE * np.clip(pre - capex / E.HORIZON, 0, None)
    disc = (1 + E.DISCOUNT) ** -y
    return -capex + post @ disc + E.SALVAGE_FRAC * capex * disc[-1]


# --------------------------------------------------------- 2. published cases
def published_cases():
    rows = list(csv.DictReader(open(E.DATA / "published_cases.csv")))
    out = []
    for rw in rows:
        cx, ben, ox = float(rw["capex"]), float(rw["annual_benefit"]), float(rw["annual_opex"])
        key = rw["source_key"]
        if key == "martin2025":
            rep = [float(v) for v in rw["reported_value"].split(";")]
            rec = [-cx + (ben - ox), -cx + 2 * (ben - ox)]
            simple_pb = 12 * cx / (ben - ox)
        elif key == "polonara2024":
            rep = [float(rw["reported_value"])]
            per_unit = 3656.0 / 10000.0            # EUR of operator time per part, Table 3 of the source
            rec = [cx / per_unit]
            simple_pb = 12 * cx / (ben - ox)
        else:
            rep = [float(v) for v in rw["reported_value"].split(";")]
            rec = [2 * ben, 85594.38 + 2 * ox]     # profitability test as written in the source
            simple_pb = 12 * cx / (ben - ox)
        npv, irr, pb = npv_common(cx / 1000, ben / 1000, ox / 1000)
        out.append(dict(case=rw["case"], key=key, robots=int(rw["robots"]), currency=rw["currency"],
                        capex=cx, capex_per_robot=cx / int(rw["robots"]), benefit=ben, opex=ox,
                        reported=rep, recomputed=rec,
                        max_dev=max(abs(a - b) for a, b in zip(rep, rec)) if key != "polonara2024" else abs(rep[0] - rec[0]),
                        simple_pb=simple_pb, npv=npv, irr=irr, pb=pb, pi=npv / (cx / 1000)))
    return out


def own_designs_common():
    r = E.results()
    return [dict(case=f"N = {i + 1}", capex=E.CAPEX[i], capex_per_robot=E.CAPEX[i] / (i + 1),
                 npv=r["npv"][i], irr=r["irr"][i], pb=r["payback"][i], pi=r["npv"][i] / E.CAPEX[i])
            for i in range(5)]


# ------------------------------------------------------- 3. switching values
def switching_values(i=1):
    s = E.BASE_SCHED[i]
    cx, ox = E.CAPEX[i], E.opex_for(i, s)

    def npv(**kw):
        c = kw.pop("capex", cx)
        d = kw.pop("discount", None)
        return E.npv_of(E.cash_flows(i, capex=c, **kw), c, discount=d)

    def dg_with(units=None, manual=None, value=E.PART_VALUE):
        u = E.UNITS_SHIFT[i] if units is None else units
        m = E.MANUAL_UNITS if manual is None else manual
        return max(u - m, 0) * E.SHIFTS[s] * value * E.GROSS_MARGIN / 1000

    out = {}
    f = _bisect(lambda k: npv(dg=dg_with(units=E.UNITS_SHIFT[i] * k)), 0.0, 1.0)
    out["Cell output (digital-twin throughput)"] = (E.UNITS_SHIFT[i] * f, "units/shift", E.UNITS_SHIFT[i], f - 1)
    mm = _bisect(lambda m: npv(dg=dg_with(manual=m)), E.MANUAL_UNITS, E.UNITS_SHIFT[i])
    out["Manual comparator output"] = (mm, "units/shift", E.MANUAL_UNITS, mm / E.MANUAL_UNITS - 1)
    f = _bisect(lambda k: npv(dg=dg_with(value=E.PART_VALUE * k)), 0.0, 1.0)
    out["Unit margin (part value x gross margin)"] = (E.PART_VALUE * E.GROSS_MARGIN * f, "$/unit",
                                                      E.PART_VALUE * E.GROSS_MARGIN, f - 1)
    f = _bisect(lambda k: npv(opex=ox * k), 1.0, 10.0)
    out["Operating cost"] = (ox * f, "k$/yr", ox, f - 1)
    f = _bisect(lambda k: npv(capex=cx * k), 1.0, 20.0)
    out["Capital expenditure"] = (cx * f, "k$", cx, f - 1)
    irr = E.irr_of(E.cash_flows(i), cx) / 100
    out["Discount rate"] = (100 * irr, "%", 100 * E.DISCOUNT, irr / E.DISCOUNT - 1)
    out["Quality savings"] = (None, "k$/yr", E.quality_savings(s), None) if npv(qs=0.0) > 0 else None
    out["Redeployment allowance"] = (None, "k$/yr", E.LABOR_REALLOC, None) if npv(ls=0.0) > 0 else None
    both = npv(qs=0.0, ls=0.0)
    return out, dict(npv_no_qs=npv(qs=0.0), npv_no_ls=npv(ls=0.0), npv_no_qs_ls=both)


def twin_tolerance():
    """For each design on its base schedule: largest overstatement of the cell's
    simulated throughput, and largest understatement of the manual comparator,
    that leave NPV >= 0."""
    out = []
    for i, s in enumerate(E.BASE_SCHED):
        cx = E.CAPEX[i]

        def v(u, m):
            dg = max(u - m, 0) * E.SHIFTS[s] * E.PART_VALUE * E.GROSS_MARGIN / 1000
            return E.npv_of(E.cash_flows(i, dg=dg), cx)
        uf = _bisect(lambda k: v(E.UNITS_SHIFT[i] * k, E.MANUAL_UNITS), 0.0, 1.0)
        mf = _bisect(lambda k: v(E.UNITS_SHIFT[i], E.MANUAL_UNITS * k), 1.0, E.UNITS_SHIFT[i] / E.MANUAL_UNITS)
        out.append(dict(N=i + 1, sched=s, cell_overstatement=100 * (1 - uf), manual_understatement=100 * (mf - 1)))
    return out


# ------------------------------------------------------------ 4. demand map
def npv_demand(i, sched, demand):
    """NPV (k$) of design (N = i+1, schedule) when annual sales are capped at `demand` units."""
    sh = E.SHIFTS[sched]
    cell = min(E.UNITS_SHIFT[i] * sh, demand)
    manual = min(E.MANUAL_UNITS * sh, demand)
    dg = max(cell - manual, 0) * E.PART_VALUE * E.GROSS_MARGIN / 1000
    qs = (E.DEFECT_MANUAL - E.DEFECT_COBOT) * E.WARRANTY * manual / 1000
    return E.npv_of(E.cash_flows(i, sched=sched, dg=dg, qs=qs), E.CAPEX[i])


def demand_map(demands=None):
    if demands is None:
        demands = np.arange(20_000, 600_001, 1_000)
    grid = {(i, s): np.array([npv_demand(i, s, d) for d in demands]) for i in range(5) for s in SCHEDS}
    keys = list(grid)
    M = np.vstack([np.zeros(len(demands))] + [grid[k] for k in keys])   # row 0: do not invest
    opts = [None] + keys
    best = [opts[j] for j in M.argmax(0)]
    best_val = M.max(0)
    # bands of constant optimal design
    bands, start = [], 0
    for k in range(1, len(demands) + 1):
        if k == len(demands) or best[k] != best[start]:
            bands.append((demands[start], demands[k - 1], best[start], best_val[start], best_val[k - 1]))
            start = k
    # minimum demand for NPV >= 0 per design
    min_dem = {}
    for k in keys:
        pos = np.where(grid[k] >= 0)[0]
        min_dem[k] = demands[pos[0]] if len(pos) else None
    return demands, grid, best, best_val, bands, min_dem


def capacity(i, sched):
    return E.UNITS_SHIFT[i] * E.SHIFTS[sched]


def report():
    tests, mc = verification_tests()
    print("Verification tests (deviation, tolerance, pass):")
    for name, dev, tol in tests:
        print(f"  {name:80s} {dev:.3g}  {tol}  {dev <= tol}")
    print("  MC convergence:", {k: round(float(v), 3) for k, v in mc.items()})
    print("Published cases:")
    for c in published_cases():
        print(f"  {c['case']:16s} rep={c['reported']} rec={[round(x, 2) for x in c['recomputed']]} "
              f"simplePB={c['simple_pb']:.1f} NPV={c['npv']:.1f}k IRR={c['irr']:.0f} PB={c['pb']:.1f} PI={c['pi']:.2f} "
              f"capex/robot={c['capex_per_robot']:.0f}")
    for d in own_designs_common():
        print(f"  {d['case']}: PI={d['pi']:.2f} capex/robot={d['capex_per_robot']:.0f}")
    sv, extra = switching_values()
    print("Switching values N=2:")
    for k, v in sv.items():
        print(" ", k, v if v is None else tuple(None if x is None else (round(x, 3) if isinstance(x, float) else x) for x in v))
    print(" ", {k: round(v, 1) for k, v in extra.items()})
    print("Twin tolerance:", [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()} for t in twin_tolerance()])
    d, grid, best, bv, bands, mind = demand_map()
    print("Demand bands:")
    for a, b, k, v0, v1 in bands:
        print(f"  {a:>7}-{b:>7}: " + ("no investment" if k is None else f"N={k[0] + 1} {k[1]:6s}") + f" NPV {v0 / 1000:.2f}->{v1 / 1000:.2f} M")
    print("Min demand for NPV>=0:", {f"N{k[0] + 1}-{k[1]}": v for k, v in mind.items()})


if __name__ == "__main__":
    report()


# ------------------------------------------- 5. capacity-capped Monte Carlo
def monte_carlo_capped(i=1):
    """Base Monte Carlo draws, with annual output capped at the simulated
    capacity of the design (upside volume cannot exceed what the cell makes)."""
    s = E.BASE_SCHED[i]
    volume, value, capex, opex = E.mc_inputs(i=i)
    cap = E.UNITS_SHIFT[i] * E.SHIFTS[s]
    vol = np.minimum(volume, cap)
    manual = E.MANUAL_UNITS * E.SHIFTS[s]
    dg = np.maximum(vol - manual, 0) * value * E.GROSS_MARGIN / 1000.0
    return np.array([E.npv_of(E.cash_flows(i, dg=dg[k], capex=capex[k], opex=opex[k]), capex[k])
                     for k in range(len(vol))])


if __name__ == "__main__":
    print("Capped MC:", {k: round(float(v), 3) for k, v in E.mc_summary(monte_carlo_capped()).items()})
