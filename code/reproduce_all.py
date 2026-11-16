#!/usr/bin/env python3
"""Regenerate every result of the article from the deposited data.

    python code/reproduce_all.py                     # results/ and figures/
    python code/reproduce_all.py path/to/manuscript.tex   # also verify the text

Writes CSV and JSON files to results/, the eight figures to figures/, and,
when a manuscript source is given, runs verify_numbers.py against it."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "code" else HERE
sys.path.insert(0, str(HERE))
import p4_economics as E  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def write_csv(name, header, rows):
    with open(RES / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


r = E.results()
write_csv("appraisal_by_configuration.csv",
          ["N", "schedule", "units_per_shift", "units_per_year", "revenue_MUSD", "gross_kUSD",
           "manual_gross_kUSD", "delta_gross_kUSD", "quality_savings_kUSD", "opex_year1_kUSD",
           "capex_kUSD", "mean_after_tax_cf_kUSD", "npv_MUSD", "irr_pct", "payback_months"],
          [[int(E.N_RANGE[i]), E.BASE_SCHED[i], int(E.UNITS_SHIFT[i]), round(r["units_year"][i]),
            round(r["revenue_m"][i], 4), round(r["gross_k"][i], 2), round(r["manual_gross"][i], 2),
            round(r["delta_gross"][i], 2), round(r["quality"][i], 2), round(r["opex"][i], 2),
            E.CAPEX[i], round(r["netcf"][i], 2), round(r["npv"][i] / 1000, 4), round(r["irr"][i], 2),
            round(r["payback"][i], 2)] for i in range(5)])

ms = E.multishift()
write_csv("shift_schedule.csv", ["N", "schedule", "npv_MUSD", "payback_months", "irr_pct"],
          [[i + 1, s, round(ms[i][j][0], 4), round(ms[i][j][1], 2), round(ms[i][j][2], 2)]
           for i in range(3) for j, s in enumerate(("single", "dual", "triple"))])

write_csv("break_even.csv", ["N", "schedule", "break_even_kunits_per_year", "share_of_capacity_pct"],
          [[i + 1, E.BASE_SCHED[i], round(v, 2), round(p, 2)] for i, (v, p, _) in enumerate(E.breakeven())])

base, sens = E.sensitivity()
write_csv("one_way_sensitivity_N2.csv", ["input", "npv_minus20_MUSD", "npv_plus20_MUSD", "range_MUSD"],
          [[k, round(a, 4), round(b, 4), round(abs(b - a), 4)] for k, (a, b) in sens.items()])

npv, X = E.monte_carlo()
write_csv("monte_carlo_draws_N2.csv", ["cell_output_units", "part_value_usd", "capex_kUSD", "opex_kUSD", "npv_kUSD"],
          [[round(x, 3) for x in row] + [round(v, 3)] for row, v in zip(X, npv)])
shares, r2 = E.variance_shares(npv, X)
def _r(d):
    return {k: round(float(v), 4) for k, v in d.items()}


summary = {"base": _r(E.mc_summary(npv)),
           "variance_shares_pct": dict(zip(["cell_output", "part_value", "capex", "opex"], (100 * shares).round(2).tolist())),
           "regression_r2": round(float(r2), 4),
           "correlated_rho_-0.5": _r(E.mc_summary(E.monte_carlo(rho_vp=-0.5)[0])),
           "volume_cv_0.10": _r(E.mc_summary(E.monte_carlo(cv_volume=0.10)[0])),
           "volume_cv_0.25": _r(E.mc_summary(E.monte_carlo(cv_volume=0.25)[0]))}
json.dump(summary, open(RES / "monte_carlo_summary_N2.json", "w"), indent=2)

sc, ev = E.scenarios()
dn, dp, di = E.disruption_adjusted()
rows = [[k, round(v["npv"], 4), round(v["irr"], 2), round(v["pb"], 2)] for k, v in sc.items()]
rows += [["Weighted expected value", round(ev, 4), "", ""],
         ["Disrupted operating rate", round(dn / 1000, 4), round(di, 2), round(dp, 2)]]
for w, (n, p) in E.labor_release().items():
    cf = E.cash_flows(1, ls=E.LABOR_REALLOC + 8 * E.SHIFTS["single"] * w / 1000)
    rows.append([f"Freed hours released at {w:.0f}/h", round(n, 4), round(E.irr_of(cf, E.CAPEX[1]), 2), round(p, 2)])
write_csv("scenarios_and_robustness_N2.csv", ["case", "npv_MUSD", "irr_pct", "payback_months"], rows)
print(f"results written to {RES}")

subprocess.run([sys.executable, str(HERE / "p4_figures.py")], check=True)

if len(sys.argv) > 1:
    sys.exit(subprocess.run([sys.executable, str(HERE / "verify_numbers.py"), sys.argv[1]]).returncode)
