"""
Electro-thermal modeling of Joule-heated metallized SiSiC foams
---------------------------------------------------------------

Author: Pietro Colucci
Affiliation: ENEA – Italian National Agency for New Technologies, Energy and Sustainable Economic Development

This script accompanies the publication:

"Electro-thermal modeling of Joule-heated metallized SiSiC foams"

The code implements the electro–thermal model and the Arrhenius-based
thermal degradation model used to generate the figures reported in the paper.

License: MIT
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ============================================================
# GLOBAL JOURNAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
    "savefig.dpi": 600,
    "figure.dpi": 150
})


def style_axes(ax, xlabel=None, ylabel=None, xlim=None, ylim=None, grid=False):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(direction="in", top=True, right=True, width=1.0)
    ax.grid(grid)


# ============================================================
# MODEL PARAMETERS
# ============================================================

alpha = 0.4760
lambda_ = 4.0e-11
T0 = 25.0


def solve_temperature_from_power(P, alpha=alpha, lambda_=lambda_, T0=T0, n_iter=100):
    """
    Solve the implicit electro-thermal balance:

    lambda * (T - T0)^4 + (T - T0) = alpha * P
    """
    P = np.asarray(P, dtype=float)
    T = np.full_like(P, 700.0, dtype=float)

    for _ in range(n_iter):
        dT = T - T0
        T = T0 + alpha * P / (1.0 + lambda_ * dT**3)

    return T


# ============================================================
# FIGURE 1 — MODEL VALIDATION
# ============================================================

V_exp = np.array([23.0, 23.4, 24.0, 25.0, 25.5])
I_exp = np.array([51.0, 68.5, 80.0, 105.0, 113.5])
T_exp = np.array([625.0, 800.0, 935.0, 1200.0, 1300.0])

P_exp = V_exp * I_exp
T_model = solve_temperature_from_power(P_exp)

abs_error = np.abs(T_exp - T_model)
rel_error = abs_error / T_exp * 100.0

rmse = np.sqrt(np.mean((T_exp - T_model) ** 2))
mae = np.mean(abs_error)
r2 = 1.0 - np.sum((T_exp - T_model) ** 2) / np.sum((T_exp - np.mean(T_exp)) ** 2)
mape = np.mean(rel_error)

validation_table = pd.DataFrame({
    "Voltage_V": V_exp,
    "Current_A": I_exp,
    "T_exp_C": T_exp,
    "T_model_C": np.round(T_model, 1),
    "Abs_error_C": np.round(abs_error, 1),
    "Rel_error_percent": np.round(rel_error, 2)
})

validation_table.to_csv(DATA_DIR / "validation_table.csv", index=False)

print("Validation metrics")
print("------------------")
print(f"RMSE = {rmse:.2f} °C")
print(f"MAE  = {mae:.2f} °C")
print(f"R²   = {r2:.3f}")
print(f"MAPE = {mape:.2f} %")
print()
print(validation_table)

fig, ax = plt.subplots(figsize=(6.6, 5.0))

ax.plot(I_exp, T_model, color="black", label="Model prediction")
ax.plot(
    I_exp,
    T_exp,
    linestyle="none",
    marker="o",
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.0,
    label="Experimental data"
)

style_axes(
    ax,
    xlabel="Current / A",
    ylabel="Temperature / $^\\circ$C",
    xlim=(45, 120),
    ylim=(550, 1350),
    grid=False
)

ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "Figure1_model_fit.png", bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 2 — ISOTHERMAL OPERATING MAP
# ============================================================

V_bias = 24.0

R0 = 0.30
t_ref = 0.20
eps_ref = 0.80
m = 2.0
n = 1.0

eps_range = np.linspace(0.70, 0.90, 300)
t_range = np.linspace(0.10, 0.35, 300)
E, TT = np.meshgrid(eps_range, t_range)

R_eff = R0 * (t_ref / TT) ** m * ((1.0 - eps_ref) / (1.0 - E)) ** n
P_map = V_bias**2 / R_eff
T_map = solve_temperature_from_power(P_map)

fig, ax = plt.subplots(figsize=(6.8, 5.4))

levels_general = np.arange(400, 1301, 50)

cs = ax.contour(
    E,
    TT,
    T_map,
    levels=levels_general,
    colors="black",
    linewidths=0.8
)
ax.clabel(cs, inline=True, fontsize=8, fmt="%d $^\\circ$C")

cs_ref = ax.contour(
    E,
    TT,
    T_map,
    levels=[950, 1050, 1150],
    colors="black",
    linewidths=[2.0, 1.5, 1.5],
    linestyles=["-", "--", ":"]
)

ax.clabel(
    cs_ref,
    inline=True,
    fontsize=8,
    fmt={
        950: "Safe 950 $^\\circ$C",
        1050: "Warning 1050 $^\\circ$C",
        1150: "Critical 1150 $^\\circ$C"
    }
)

style_axes(
    ax,
    xlabel=r"Porosity, $\varepsilon$",
    ylabel=r"Ligament thickness, $t$ / mm",
    xlim=(0.70, 0.90),
    ylim=(0.10, 0.35),
    grid=False
)

plt.tight_layout()
plt.savefig(FIG_DIR / "Figure2_isotherms.png", bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 4 — DAMAGE PROBABILITY VS TIME
# ============================================================

R_gas = 8.3145
Ea = 210.0e3
k0 = 2.0e6

eps_ref_damage = 0.80
t_ref_damage = 0.20
a_eps = 2.0
b_t = 1.5


def microstructure_factor(t_mm, eps):
    return (eps / eps_ref_damage) ** a_eps * (t_ref_damage / max(t_mm, 1.0e-6)) ** b_t


def hazard_rate(T_C, t_mm, eps):
    T_K = T_C + 273.15
    return microstructure_factor(t_mm, eps) * k0 * np.exp(-Ea / (R_gas * T_K))


def damage_probability(T_C, hours, t_mm, eps):
    h = hazard_rate(T_C, t_mm, eps)
    return 1.0 - np.exp(-h * np.asarray(hours))


t_mm = 0.10
eps = 0.82

temps_C = [25, 700, 800, 900, 950, 1000, 1100, 1200]
hours = np.linspace(0, 200, 500)

fig, ax = plt.subplots(figsize=(6.8, 5.2))

for T_C in temps_C:
    P_damage = damage_probability(T_C, hours, t_mm, eps)
    ax.plot(hours, P_damage, label=f"{T_C} $^\\circ$C")

style_axes(
    ax,
    xlabel="Exposure time / h",
    ylabel="Damage probability",
    xlim=(0, 200),
    ylim=(0, 1.0),
    grid=False
)

ax.legend(frameon=False, loc="upper left", ncol=1)
plt.tight_layout()
plt.savefig(FIG_DIR / "Figure4_damage_probability.png", bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 5 — TIME TO TARGET DAMAGE PROBABILITY
# ============================================================

T_grid = np.linspace(25, 1200, 300)
targets = [0.01, 0.10, 0.50, 0.90]

fig, ax = plt.subplots(figsize=(6.8, 5.2))

for p in targets:
    time_to_target = []

    for T_C in T_grid:
        h = hazard_rate(T_C, t_mm, eps)
        time_value = -np.log(1.0 - p) / h if h > 0 else np.inf
        time_to_target.append(time_value)

    ax.plot(T_grid, time_to_target, label=f"Time to $P={int(p * 100)}\\%$")

ax.set_yscale("log")

style_axes(
    ax,
    xlabel="Temperature / $^\\circ$C",
    ylabel="Time to target probability / h",
    xlim=(25, 1200),
    grid=False
)

ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "Figure5_time_to_probability.png", bbox_inches="tight")
plt.close()


print()
print("Files generated successfully:")
print(f"- {DATA_DIR / 'validation_table.csv'}")
print(f"- {FIG_DIR / 'Figure1_model_fit.png'}")
print(f"- {FIG_DIR / 'Figure2_isotherms.png'}")
print(f"- {FIG_DIR / 'Figure4_damage_probability.png'}")
print(f"- {FIG_DIR / 'Figure5_time_to_probability.png'}")