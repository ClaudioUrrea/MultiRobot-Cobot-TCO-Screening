#!/usr/bin/env python3
"""
p4_economics.py -- single source of truth for every economic number in the article
"Digital-Twin-Based Total Cost of Ownership and Risk Screening for Multi-Robot
Collaborative Assembly Cells" (C. Urrea, submitted to Systems, 2026).

All inputs are read from the data/ folder: digital-twin throughput series,
operational inputs, itemized capital and operating cost (with the flag that
marks items scaling with operating hours) and the financial, scenario and
Monte Carlo parameters.

Appraisal basis: ten-year cash flows, INCREMENTAL against a manual line that
runs the same shift schedule, AFTER TAX with a straight-line depreciation
shield and no tax credit for losses; 15% salvage at year ten.

Nature of the data: every input is a simulation output of the companion
digital twin (doi:10.1109/ACCESS.2026.3754761) or a stated scenario parameter.
No installation was costed, no accounting record was consulted and no person
supplied data. The results are a pre-investment screening.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np


def _data_dir():
    here = Path(__file__).resolve().parent
    for d in (os.environ.get("P4_DATA"), here / "data", here.parent / "data"):
        if d and (Path(d) / "parameters.json").exists():
            return Path(d)
    raise FileNotFoundError("data/ folder with parameters.json not found")


DATA = _data_dir()
_P = json.loads((DATA / "parameters.json").read_text())


def _rows(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


_dt = _rows("digital_twin_inputs.csv")
N_RANGE = np.array([int(r["N"]) for r in _dt])
EFFICIENCY = np.array([float(r["coordination_efficiency"]) for r in _dt])
UNITS_SHIFT = np.array([int(r["units_per_shift"]) for r in _dt])
SINGLE_RATE = _P["single_robot_rate_units_per_shift"]
MANUAL_UNITS = _P["manual_units_per_shift"]

# ---------------------------------------------------------------- capital (k$)
CAPEX_ITEMS = {r["item"]: [float(r[f"N{n}"]) for n in N_RANGE] for r in _rows("capex_items_kUSD.csv")}
CAPEX = np.sum(list(CAPEX_ITEMS.values()), axis=0)
HARDWARE = [r["item"] for r in _rows("capex_items_kUSD.csv") if r["category"] == "Hardware"]

# ------------------------------------------------- operating cost (k$/yr, 1 shift)
OPEX_ITEMS = {r["item"]: ([float(r[f"N{n}"]) for n in N_RANGE], r["scales_with_operating_hours"] == "1")
              for r in _rows("opex_items_single_shift_kUSD_per_year.csv")}
OPEX_FIXED = np.sum([np.array(v) for v, h in OPEX_ITEMS.values() if not h], axis=0)
OPEX_HOURLY = np.sum([np.array(v) for v, h in OPEX_ITEMS.values() if h], axis=0)
OPEX_SINGLE = OPEX_FIXED + OPEX_HOURLY

# ------------------------------------------------------------------- schedule
SHIFTS = {k: float(v) for k, v in _P["shifts_per_year"].items()}
BASE_SCHED = _P["base_schedule"]


def opex_for(i, sched):
    return OPEX_FIXED[i] + OPEX_HOURLY[i] * SHIFTS[sched] / SHIFTS["single"]


OPEX_BASE = np.array([opex_for(i, s) for i, s in enumerate(BASE_SCHED)])

# ------------------------------------------------------------ revenue/benefits
PART_VALUE, GROSS_MARGIN = _P["part_value_usd"], _P["gross_margin"]
DEFECT_MANUAL, DEFECT_COBOT = _P["defect_rate_manual"], _P["defect_rate_cobot_nominal"]
DEFECT_DISRUPTED, UNITS_DISRUPTED = _P["defect_rate_cobot_disrupted"], _P["disrupted_units_per_shift_N2"]
WARRANTY = _P["warranty_cost_usd_per_defect"]          # $ per defective unit
LABOR_REALLOC = _P["redeployment_allowance_kUSD"]       # k$/yr

DISCOUNT, HORIZON, SALVAGE_FRAC = _P["discount_rate"], int(_P["horizon_years"]), _P["salvage_fraction"]
TAX_RATE, OPEX_ESC = _P["tax_rate"], _P["operating_cost_escalation"]
RAMP = _P["ramp_first_three_months"]
YEAR1_RAMP = (sum(RAMP) + (12 - len(RAMP))) / 12.0


def quality_savings(sched, d_cobot=DEFECT_COBOT):
    """Defect reduction on the units the manual line made on the same schedule."""
    return (DEFECT_MANUAL - d_cobot) * WARRANTY * MANUAL_UNITS * SHIFTS[sched] / 1000.0


def delta_gross(i, sched, units_shift=None, value=PART_VALUE, margin=GROSS_MARGIN):
    u = UNITS_SHIFT[i] if units_shift is None else units_shift
    return max(u - MANUAL_UNITS, 0) * SHIFTS[sched] * value * margin / 1000.0


def cash_flows(i, sched=None, dg=None, capex=None, opex=None, tax=None,
               qs=None, ls=None):
    s = BASE_SCHED[i] if sched is None else sched
    dg = delta_gross(i, s) if dg is None else dg
    cx = CAPEX[i] if capex is None else capex
    ox = opex_for(i, s) if opex is None else opex
    tx = TAX_RATE if tax is None else tax
    qs = quality_savings(s) if qs is None else qs
    ls = LABOR_REALLOC if ls is None else ls
    dep = cx / HORIZON
    out = []
    for y in range(1, HORIZON + 1):
        ramp = YEAR1_RAMP if y == 1 else 1.0
        pre = (dg + qs + ls) * ramp - ox * (1 + OPEX_ESC) ** (y - 1)
        out.append(pre - max(0.0, pre - dep) * tx)
    return np.array(out)


def npv_of(cf, cx, discount=None):
    d = DISCOUNT if discount is None else discount
    pv = sum(c / (1 + d) ** y for y, c in enumerate(cf, start=1))
    return -cx + pv + SALVAGE_FRAC * cx / (1 + d) ** HORIZON


def irr_of(cf, cx):
    if npv_of(cf, cx, 1e-6) <= 0:
        return float("nan")
    lo, hi = 1e-6, 10.0
    for _ in range(300):
        r = (lo + hi) / 2
        if npv_of(cf, cx, discount=r) > 0: lo = r
        else: hi = r
    return lo * 100


def payback_of(cf, cx):
    cum = 0.0
    for m in range(1, 12 * HORIZON + 1):
        y = (m - 1) // 12
        if m <= 3:    step = cf[0] / 12.0 * RAMP[m - 1] / YEAR1_RAMP
        elif m <= 12: step = cf[0] / 12.0 / YEAR1_RAMP
        else:         step = cf[y] / 12.0
        if step > 0 and cum + step >= cx:
            return m - 1 + (cx - cum) / step
        cum += step
    return float("nan")


def cumulative_monthly(i, months=36):
    """Cumulative after-tax cash flow by month (k$), starting at -capex."""
    cf, cx = cash_flows(i), CAPEX[i]
    out, cum = [-cx], -cx
    for m in range(1, months + 1):
        y = (m - 1) // 12
        if m <= 3:    step = cf[0] / 12.0 * RAMP[m - 1] / YEAR1_RAMP
        elif m <= 12: step = cf[0] / 12.0 / YEAR1_RAMP
        else:         step = cf[y] / 12.0
        cum += step; out.append(cum)
    return np.array(out)


def results():
    r = {k: [] for k in ("units_year", "revenue_m", "gross_k", "manual_gross",
                         "delta_gross", "quality", "opex", "netcf", "npv", "irr",
                         "payback")}
    for i, s in enumerate(BASE_SCHED):
        uy = UNITS_SHIFT[i] * SHIFTS[s]
        r["units_year"].append(uy)
        r["revenue_m"].append(uy * PART_VALUE / 1e6)
        r["gross_k"].append(uy * PART_VALUE * GROSS_MARGIN / 1000)
        r["manual_gross"].append(MANUAL_UNITS * SHIFTS[s] * PART_VALUE * GROSS_MARGIN / 1000)
        r["delta_gross"].append(delta_gross(i, s))
        r["quality"].append(quality_savings(s))
        r["opex"].append(opex_for(i, s))
        cf = cash_flows(i)
        r["netcf"].append(cf.mean()); r["npv"].append(npv_of(cf, CAPEX[i]))
        r["irr"].append(irr_of(cf, CAPEX[i])); r["payback"].append(payback_of(cf, CAPEX[i]))
    return {k: np.array(v) for k, v in r.items()}


def multishift():
    rows = {}
    for i in (0, 1, 2):
        rows[i] = []
        for s in ("single", "dual", "triple"):
            cf = cash_flows(i, sched=s)
            rows[i].append((npv_of(cf, CAPEX[i]) / 1000, payback_of(cf, CAPEX[i]),
                            irr_of(cf, CAPEX[i])))
    return rows


def breakeven():
    """Annual cell output (k units) at which NPV = 0, and its share of capacity."""
    out = []
    for i, s in enumerate(BASE_SCHED):
        lo, hi = 0.0, 3 * UNITS_SHIFT[i]
        for _ in range(200):
            mid = (lo + hi) / 2
            v = npv_of(cash_flows(i, dg=delta_gross(i, s, units_shift=mid)), CAPEX[i])
            if v < 0: lo = mid
            else: hi = mid
        out.append((hi * SHIFTS[s] / 1000.0, 100 * hi / UNITS_SHIFT[i], hi))
    return out


def disruption_adjusted():
    """N = 2 with the disrupted campaign rate (547 u/shift) and simulated 0.8 % defects."""
    i, s = 1, "single"
    cf = cash_flows(i, dg=delta_gross(i, s, units_shift=UNITS_DISRUPTED),
                    qs=quality_savings(s, d_cobot=DEFECT_DISRUPTED))
    return npv_of(cf, CAPEX[i]), payback_of(cf, CAPEX[i]), irr_of(cf, CAPEX[i])


def labor_release(i=1, wages=tuple(_P["labor_release"]["wages_per_hour"]),
                  hours_per_shift=_P["labor_release"]["freed_hours_per_shift_N2"]):
    """NPV if the operator-hours freed by the cell (P3: 16 -> 8 op-h/shift at N = 2)
    were released at a fully loaded wage w instead of redeployed (not in base case)."""
    s, out = BASE_SCHED[i], {}
    for w in wages:
        ls = LABOR_REALLOC + hours_per_shift * SHIFTS[s] * w / 1000.0
        cf = cash_flows(i, ls=ls)
        out[w] = (npv_of(cf, CAPEX[i]) / 1000, payback_of(cf, CAPEX[i]))
    return out


def sensitivity(i=1, f=_P["one_way_change"]):
    s = BASE_SCHED[i]
    base = npv_of(cash_flows(i), CAPEX[i]) / 1000
    def v(**kw):
        cx = kw.pop("capex", CAPEX[i]); d = kw.pop("discount", None)
        return npv_of(cash_flows(i, capex=cx, **kw), cx, discount=d) / 1000
    res = {
        "Production volume": [v(dg=delta_gross(i, s, units_shift=UNITS_SHIFT[i] * k)) for k in (1 - f, 1 + f)],
        "Part value": [v(dg=delta_gross(i, s, value=PART_VALUE * k)) for k in (1 - f, 1 + f)],
        "Operating cost": [v(opex=opex_for(i, s) * k) for k in (1 - f, 1 + f)],
        "Capital expenditure": [v(capex=CAPEX[i] * k) for k in (1 - f, 1 + f)],
        "Discount rate": [v(discount=DISCOUNT * k) for k in (1 - f, 1 + f)],
        "Tax rate": [v(tax=TAX_RATE * k) for k in (1 - f, 1 + f)],
        "Quality savings": [v(qs=quality_savings(s) * k) for k in (1 - f, 1 + f)],
        "Labor reallocation": [v(ls=LABOR_REALLOC * k) for k in (1 - f, 1 + f)],
    }
    return base, dict(sorted(res.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])))


_MC = _P["monte_carlo"]
MC_CV_VOLUME, MC_CAPEX_SD_FRAC, MC_OPEX_SD_FRAC = _MC["volume_cv"], _MC["capex_sd_fraction"], _MC["opex_sd_fraction"]
MC_DRAWS, MC_SEED = int(_MC["draws"]), int(_MC["seed"])
PV_LO, PV_HI = _MC["part_value_uniform"]


def mc_inputs(n_sim=MC_DRAWS, seed=MC_SEED, rho_vp=0.0, cv_volume=MC_CV_VOLUME, i=1):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sim, 2))
    z2 = rho_vp * z[:, 0] + np.sqrt(1 - rho_vp ** 2) * z[:, 1]
    vol_mu = UNITS_SHIFT[i] * SHIFTS[BASE_SCHED[i]]
    volume = vol_mu * (1 + cv_volume * z[:, 0])
    from math import erf
    u = 0.5 * (1 + np.vectorize(erf)(z2 / np.sqrt(2)))       # Gaussian copula
    value = PV_LO + (PV_HI - PV_LO) * u
    capex = rng.normal(CAPEX[i], MC_CAPEX_SD_FRAC * CAPEX[i], n_sim)
    opex = rng.normal(OPEX_BASE[i], MC_OPEX_SD_FRAC * OPEX_BASE[i], n_sim)
    return volume, value, capex, opex


def monte_carlo(n_sim=MC_DRAWS, seed=MC_SEED, rho_vp=0.0, cv_volume=MC_CV_VOLUME, i=1):
    s = BASE_SCHED[i]
    volume, value, capex, opex = mc_inputs(n_sim, seed, rho_vp, cv_volume, i)
    manual = MANUAL_UNITS * SHIFTS[s]
    dg = np.maximum(volume - manual, 0) * value * GROSS_MARGIN / 1000.0
    out = np.array([npv_of(cash_flows(i, dg=dg[k], capex=capex[k], opex=opex[k]), capex[k])
                    for k in range(n_sim)])
    return out, np.column_stack([volume, value, capex, opex])


def variance_shares(npv, X):
    Z = (X - X.mean(0)) / X.std(0); y = (npv - npv.mean()) / npv.std()
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), y, rcond=None)
    src2 = beta[1:] ** 2
    r2 = 1 - np.sum((y - np.column_stack([np.ones(len(y)), Z]) @ beta) ** 2) / np.sum(y ** 2)
    return src2 / src2.sum(), r2


def mc_summary(npv):
    q = np.percentile(npv, [5, 25, 50, 75, 95]) / 1000
    m, sd = npv.mean(), npv.std()
    skew = np.mean(((npv - m) / sd) ** 3)
    return dict(p5=q[0], p25=q[1], p50=q[2], p75=q[3], p95=q[4],
                mean=m / 1000, p_neg=100 * np.mean(npv < 0),
                p_gt05=100 * np.mean(npv > 500), p_gt1=100 * np.mean(npv > 1000),
                skew=skew)


SCENARIOS = _P["scenarios"]  # multipliers on volume, part value, capex, opex
SCEN_WEIGHTS = _P["scenario_weights"]


def scenarios(i=1):
    s, out = BASE_SCHED[i], {}
    for k, m in SCENARIOS.items():
        cx = CAPEX[i] * m["capex"]
        cf = cash_flows(i, dg=delta_gross(i, s, units_shift=UNITS_SHIFT[i] * m["vol"],
                                          value=PART_VALUE * m["value"]),
                        capex=cx, opex=opex_for(i, s) * m["opex"])
        out[k] = dict(npv=npv_of(cf, cx) / 1000, irr=irr_of(cf, cx), pb=payback_of(cf, cx),
                      volume=UNITS_SHIFT[i] * m["vol"] * SHIFTS[s] / 1000,
                      value=PART_VALUE * m["value"], capex=cx, opex=opex_for(i, s) * m["opex"])
    ev = sum(SCEN_WEIGHTS[k] * out[k]["npv"] for k in out)
    return out, ev


def report():
    r = results()
    print("Config  sched  u/yr    dGross  Qual  Opex   Capex NetCF  NPV$M  IRR  PB")
    for i in range(5):
        print(f"N={i+1}  {BASE_SCHED[i]:6s} {r['units_year'][i]:7.0f} {r['delta_gross'][i]:7.1f} "
              f"{r['quality'][i]:5.1f} {r['opex'][i]:6.1f} {CAPEX[i]:5.0f} {r['netcf'][i]:5.0f} "
              f"{r['npv'][i]/1000:6.2f} {r['irr'][i]:4.0f} {r['payback'][i]:5.1f}")
    print("opex single:", OPEX_SINGLE, " fixed:", OPEX_FIXED, " hourly:", OPEX_HOURLY)
    print("Multi-shift (NPV $M / PB mo / IRR %):")
    for i, rows in multishift().items():
        print(f"  N={i+1}: " + " | ".join(f"{a:.2f}/{b:.1f}/{c:.0f}" for a, b, c in rows))
    print("Break-even:", [(round(a), round(b), round(c)) for a, b, c in breakeven()])
    print("Disruption-adjusted N=2:", disruption_adjusted())
    b, sens = sensitivity(); print("Sensitivity N=2, base", round(b, 3))
    for k, v in sens.items(): print(f"  {k:22s} {v[0]:.2f} {v[1]:.2f}")
    npv, X = monte_carlo(); print("MC:", {k: round(v, 3) for k, v in mc_summary(npv).items()})
    sh, r2 = variance_shares(npv, X); print("  shares vol,val,capex,opex:", np.round(sh * 100, 1), "R2", round(r2, 3))
    for rho in (-0.5,):
        n2, _ = monte_carlo(rho_vp=rho); print(f"  rho={rho}:", {k: round(v, 3) for k, v in mc_summary(n2).items()})
    for cv in (0.10, 0.25):
        n3, _ = monte_carlo(cv_volume=cv); print(f"  cv={cv}:", {k: round(v, 3) for k, v in mc_summary(n3).items()})
    print("Labor release N=2:", labor_release())
    sc, ev = scenarios(); print("Scenarios:", {k: {a: round(b, 2) for a, b in v.items()} for k, v in sc.items()}, "EV", round(ev, 3))


if __name__ == "__main__":
    report()
