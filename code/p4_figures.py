#!/usr/bin/env python3
"""Figures for P4 (Machines). Every plotted value comes from p4_economics.py
(single source of truth) or from the digital-twin inputs archived with the
companion study (P3). Style: estilo_figuras.aplicar()."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figure_style as EF
import p4_economics as E

EF.aplicar()
C = EF.PALETTE
_here = Path(__file__).resolve().parent
OUT = (_here.parent / "figures") if (_here.parent / "data").exists() and _here.name == "code" else (_here / "figures")
OUT.mkdir(exist_ok=True)
W_FULL, W_085 = 5.6, 4.7
CFG = [f"$N = {n}$" for n in E.N_RANGE]


def sched_label(s):
    return "1 shift" if s == "single" else "2 shifts"


def panel(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def fig1():
    groups = {
        "Robots": ["Robots (UR5e)"],
        "Grippers and tooling": ["Grippers and tooling"],
        "Cameras and network": ["Depth cameras", "Network"],
        "Compute": ["Compute platform"],
        "Software": ["Control licenses", "Digital twin platform", "Simulation tools"],
        "Installation": ["Mechanical mounting", "Electrical integration", "Safety enclosure", "Cabling"],
        "Training and commissioning": ["Operator training", "Technician training", "Commissioning"],
        "Contingency": ["Contingency"],
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.7))
    x, bot = np.arange(5), np.zeros(5)
    cols = C + ["#b0b0b0"]
    for (lbl, keys), col in zip(groups.items(), cols):
        v = np.sum([E.CAPEX_ITEMS[k] for k in keys], axis=0)
        a.bar(x, v, 0.6, bottom=bot, color=col, label=lbl, edgecolor="white", linewidth=0.4)
        bot += v
    for i, t in enumerate(bot):
        a.text(i, t + 8, f"{t:.0f}", ha="center", fontsize=7)
    a.set_xticks(x, CFG); a.set_ylabel("Capital expenditure (k\\$)"); a.set_ylim(0, 760)
    a.legend(ncol=2, fontsize=6, loc="upper left", handlelength=1.0, columnspacing=0.8)
    panel(a, "(a)")
    ogroups = {
        "Maintenance": ["Preventive maintenance", "Spare parts", "Calibration"],
        "Energy": ["Electricity", "HVAC"],
        "Software": ["License renewals", "Updates and patches"],
        "Technician and supervision": ["Technician time", "Supervision time"],
        "Insurance and audits": ["Liability insurance", "Safety audits"],
    }
    bot = np.zeros(5)
    for (lbl, keys), col in zip(ogroups.items(), C):
        v = np.zeros(5)
        for k in keys:
            vals, hourly = E.OPEX_ITEMS[k]
            mult = np.array([E.SHIFTS[s] / E.SHIFTS["single"] if hourly else 1.0 for s in E.BASE_SCHED])
            v += np.array(vals) * mult
        b.bar(x, v, 0.6, bottom=bot, color=col, label=lbl, edgecolor="white", linewidth=0.4)
        bot += v
    for i, t in enumerate(bot):
        b.text(i, t + 4, f"{t:.1f}", ha="center", fontsize=7)
    b.set_xticks(x, [f"{c}\n{sched_label(s)}" for c, s in zip(CFG, E.BASE_SCHED)], fontsize=6.5)
    b.set_ylabel("First-year operating cost (k\\$/yr)"); b.set_ylim(0, 330)
    b.legend(fontsize=6, loc="upper left", handlelength=1.0)
    panel(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig1_Cost_Composition")


def fig2():
    r = E.results()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    x = np.arange(5)
    a.bar(x, r["npv"] / 1000, 0.6, color=C[0])
    for i in range(5):
        a.text(i, r["npv"][i] / 1000 + 0.08, f"{r['npv'][i]/1000:.2f}\nIRR {r['irr'][i]:.0f}%",
               ha="center", fontsize=6.5)
    a.set_xticks(x, CFG); a.set_ylabel("Ten-year NPV (M\\$)"); a.set_ylim(0, 4.9)
    panel(a, "(a)")
    b.barh(x, r["payback"], 0.6, color=C[1])
    for i in range(5):
        b.text(r["payback"][i] + 0.8, i, f"{r['payback'][i]:.1f}", va="center", fontsize=7)
    b.set_yticks(x, CFG); b.invert_yaxis(); b.set_xlabel("Payback period (months)"); b.set_xlim(0, 46)
    panel(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig4_NPV_Payback")


def fig3():
    r = E.results()
    fig, ax = plt.subplots(figsize=(W_085, 2.7))
    m = np.arange(0, 43)
    for i in range(5):
        cum = E.cumulative_monthly(i, months=42)
        ax.plot(m, cum, color=C[i], label=f"{CFG[i]} ({sched_label(E.BASE_SCHED[i])})")
        ax.plot([r["payback"][i]], [0], marker="o", ms=4, color=C[i], zorder=5)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Months from start of production")
    ax.set_ylabel("Cumulative after-tax cash flow (k\\$)")
    ax.set_xlim(0, 42); ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "Fig5_Cumulative_Cash_Flow")


def fig4():
    ms = E.multishift()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    x, w = np.arange(3), 0.26
    for j, (s, col) in enumerate(zip(("One shift", "Two shifts", "Three shifts"), C)):
        a.bar(x + (j - 1) * w, [ms[i][j][0] for i in range(3)], w, color=col, label=s)
        b.bar(x + (j - 1) * w, [ms[i][j][1] for i in range(3)], w, color=col, label=s)
    a.set_xticks(x, CFG[:3]); a.set_ylabel("Ten-year NPV (M\\$)"); a.legend(loc="upper left")
    b.set_xticks(x, CFG[:3]); b.set_ylabel("Payback period (months)"); b.legend(loc="upper right")
    panel(a, "(a)"); panel(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig6_Shift_Schedule")


def fig5():
    names = ["Throughput\n(units/shift)", "Defect rate\n(%)", "Energy\n(kWh/shift)", "Uptime\n(%)"]
    nom = np.array([554, 0.6, 31.2, 98.5]); dis = np.array([547, 0.8, 32.8, 95.8])
    fig = plt.figure(figsize=(W_FULL, 3.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.75, wspace=0.6)
    for k in range(4):
        ax = fig.add_subplot(gs[0, k])
        ax.bar([0, 1], [nom[k], dis[k]], 0.6, color=[C[0], C[1]])
        ax.set_xticks([0, 1], ["Nominal", "Disrupted"], fontsize=6.5)
        ax.set_title(names[k], fontsize=7)
        top = max(nom[k], dis[k]) * 1.22
        ax.set_ylim(0, top)
        for i, v in enumerate((nom[k], dis[k])):
            ax.text(i, v + top * 0.02, f"{v:g}", ha="center", fontsize=6.5)
        if k == 0:
            panel(ax, "(a)")
    ax = fig.add_subplot(gs[1, :])
    pct = dis / nom * 100
    y = np.arange(4)
    ax.axvspan(90, 110, color="#e6e6e6", label="$\\pm$10% band")
    ax.barh(y, pct, 0.55, color=[C[2] if 90 <= p <= 110 else C[3] for p in pct])
    ax.axvline(100, color="black", lw=0.8, ls="--")
    for i, p in enumerate(pct):
        ax.text(max(p, 110) + 1, i, f"{p:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y, [n.replace("\n", " ") for n in names]); ax.invert_yaxis()
    ax.set_xlim(70, 145); ax.set_xlabel("Disrupted value as a percentage of the nominal value")
    ax.legend(loc="lower right")
    panel(ax, "(b)")
    save(fig, "Fig3_Nominal_vs_Disrupted")


def fig6():
    base, sens = E.sensitivity()
    fig, ax = plt.subplots(figsize=(W_085, 2.6))
    names = list(sens.keys())[::-1]
    for i, k in enumerate(names):
        lo, hi = sens[k]
        ax.barh(i, lo - base, left=base, color=C[3], height=0.6, label="$-$20%" if i == 0 else None)
        ax.barh(i, hi - base, left=base, color=C[0], height=0.6, label="$+$20%" if i == 0 else None)
        ax.text(min(lo, hi) - 0.015, i, f"{min(lo, hi):.2f}", ha="right", va="center", fontsize=6.5)
        ax.text(max(lo, hi) + 0.015, i, f"{max(lo, hi):.2f}", ha="left", va="center", fontsize=6.5)
    ax.axvline(base, color="black", lw=0.8)
    ax.set_yticks(range(len(names)), [n.replace("Labor reallocation", "Redeployment allowance") for n in names])
    ax.set_xlabel("NPV for $N = 2$ (M\\$)"); ax.set_xlim(0.2, 1.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "Fig7_Sensitivity_Tornado")


def fig7():
    npv, _ = E.monte_carlo()
    v = npv / 1000
    s = E.mc_summary(npv)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    a.hist(v, bins=60, color=C[0], alpha=0.85)
    ytop = a.get_ylim()[1]
    for q, lab, ls in ((s["p5"], "5th", ":"), (s["p50"], "Median", "--"), (s["p95"], "95th", ":")):
        a.axvline(q, color="black", lw=0.8, ls=ls)
        a.text(q, ytop * 0.98, f" {lab}\n {q:.2f}", fontsize=6.5, va="top")
    a.axvline(0, color=C[3], lw=1.0)
    a.set_xlabel("NPV (M\\$)"); a.set_ylabel("Number of draws")
    panel(a, "(a)")
    xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
    b.plot(xs, ys, color=C[0])
    b.axvline(0, color=C[3], lw=1.0)
    b.text(0.05, 0.08, f"P(NPV < 0) = {s['p_neg']:.1f}%", fontsize=7)
    b.set_xlabel("NPV (M\\$)"); b.set_ylabel("Cumulative probability"); b.set_ylim(0, 1.02)
    panel(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig8_MonteCarlo_NPV")


def fig8():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.4))
    x = np.arange(5)
    a.bar(x, E.UNITS_SHIFT, 0.6, color=C[0], label="Simulated")
    a.plot(x, E.SINGLE_RATE * E.N_RANGE, color=C[1], marker="o", ms=3, label="Linear scaling (307$N$)")
    for i, u in enumerate(E.UNITS_SHIFT):
        a.text(i, u - 30, str(u), ha="center", va="top", fontsize=6.5, color="white")
    a.set_xticks(x, CFG); a.set_ylabel("Throughput (units/shift)"); a.set_ylim(0, 1700)
    a.legend(loc="upper left")
    panel(a, "(a)")
    b.plot(x, E.EFFICIENCY * 100, color=C[0], marker="o", ms=4)
    for i, e in enumerate(E.EFFICIENCY * 100):
        b.text(i, e + 1.5, f"{e:.1f}", ha="center", fontsize=6.5)
    b.set_xticks(x, CFG); b.set_ylabel("Coordination efficiency (%)"); b.set_ylim(60, 106)
    panel(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig2_Throughput_Scaling")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
