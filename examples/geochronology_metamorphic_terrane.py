#!/usr/bin/env python3
"""
Case study: radiogenic heat and multi-system geochronology

Worked example for a metamorphosed volcanic terrane crosscut by two dikes.
Combines U-235 decay heat budgeting, U-Pb zircon ages, and Rb-Sr isochrons
to build an integrated timeline of crystallization, metamorphism, and
post-orogenic intrusion.

Run from the examples directory:
    uv run python geochronology_metamorphic_terrane.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import clean_plot_style

from earthsciences.geochronology import (
    DECAY_CONSTANTS,
    calculate_age,
    isochron_dating,
)
from earthsciences.utils.logging_config import log_block, log_section, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
ISOCHRON_FIG = OUTPUT_DIR / "geochronology_metamorphic_terrane_isochron.png"

WR_RB_SR = np.array([0.25, 0.30, 0.50, 1.00])
WR_SR_RATIO = np.array([0.710202, 0.711642, 0.717404, 0.731807])

B_RB_SR = np.array([0.05, 0.60, 5.00])
B_SR_RATIO = np.array([0.710931, 0.712495, 0.725009])

G_RB_SR = np.array([0.07, 1.30, 15.0])
G_SR_RATIO = np.array([0.729162, 0.732660, 0.771624])

DIKE_V_PB_U = (0.03151, 0.0001)
DIKE_W_PB_U = (0.003107, 0.000015)


def u_pb_age_ma(
    pb_u_ratio: float, pb_u_uncertainty: float, decay_constant: float
) -> tuple[float, float]:
    """Age in Ma from 206Pb/238U with first-order error propagation."""
    age_years = calculate_age(1.0, pb_u_ratio, decay_constant)
    age_err_years = (1 / decay_constant) * (1 / (1 + pb_u_ratio)) * pb_u_uncertainty
    return age_years / 1e6, age_err_years / 1e6


def rb_sr_isochron(rb_sr: np.ndarray, sr_ratio: np.ndarray, label: str) -> dict:
    """Fit an Rb-Sr isochron and log summary statistics."""
    result = isochron_dating(
        rb_sr,
        sr_ratio,
        np.ones(len(rb_sr)),
        DECAY_CONSTANTS["Rb87"],
    )
    age_ma = result["age"] / 1e6
    slope = result["slope"]
    intercept = result["initial_ratio"]
    r_squared = 1 - (
        np.sum((sr_ratio - (intercept + slope * rb_sr)) ** 2)
        / np.sum((sr_ratio - np.mean(sr_ratio)) ** 2)
    )

    logger.info("%s", label)
    logger.info("  87Sr/86Sr = %.6f + %.6f × 87Rb/86Sr", intercept, slope)
    logger.info("  R² = %.6f, MSWD = %.4f", r_squared, result["mswd"])
    logger.info("  Age = %.1f Ma, initial 87Sr/86Sr = %.6f", age_ma, intercept)

    result["age_ma"] = age_ma
    result["r_squared"] = r_squared
    return result


def radiogenic_heat_budget() -> None:
    """Quantify heat from 235U decay and its role in Earth's thermal budget."""
    log_section("Radiogenic heat from 235U decay")

    lambda_235 = DECAY_CONSTANTS["U235"]
    t_half_235 = np.log(2) / lambda_235
    earth_age = 4.56e9

    mass_u235 = 235.04392525
    mass_pb207 = 206.975885
    mass_he4 = 4.0026035
    amu_to_mev = 931.5
    ev_to_joules = 1.60207e-19
    mev_to_ev = 1e6
    cal_to_joules = 4.1855
    mass_ocean = 1.40e24

    n_alpha = (235 - 207) / 4
    mass_defect = mass_u235 - (mass_pb207 + n_alpha * mass_he4)
    energy_mev = mass_defect * amu_to_mev

    atoms_per_ma = 8.18e37
    energy_j = atoms_per_ma * energy_mev * mev_to_ev * ev_to_joules
    energy_cal = energy_j / cal_to_joules
    delta_t_ocean = energy_cal / mass_ocean

    logger.info("  Half-life (235U):           %.3f Ga", t_half_235 / 1e9)
    logger.info("  Half-lives since 4.56 Ga: %.2f", earth_age / t_half_235)
    logger.info("  Alpha particles / atom:     %d", int(n_alpha))
    logger.info("  Mass defect:                %.5f amu", mass_defect)
    logger.info("  Energy per decay:           %.1f MeV", energy_mev)
    logger.info("  Heat at 4 Ga:               %.3e cal/Ma", energy_cal)
    logger.info("  Equivalent ocean ΔT*:       %.2f °C/Ma", delta_t_ocean)
    log_block(
        "*Hypothetical if all radiogenic heat entered the oceans instantly;\n"
        " in reality U is crust-hosted and heat is released gradually."
    )


def upb_and_rbsr_ages() -> dict:
    """Compute U-Pb and Rb-Sr ages for dikes and isochron suites."""
    log_section("U-Pb zircon ages (crosscutting dikes)")

    age_v, err_v = u_pb_age_ma(*DIKE_V_PB_U, DECAY_CONSTANTS["U238"])
    age_w, err_w = u_pb_age_ma(*DIKE_W_PB_U, DECAY_CONSTANTS["U238"])

    logger.info("  Dike V: %.1f ± %.1f Ma  (206Pb/238U = %s)", age_v, err_v, DIKE_V_PB_U[0])
    logger.info("  Dike W: %.1f ± %.1f Ma  (206Pb/238U = %s)", age_w, err_w, DIKE_W_PB_U[0])

    log_section("Rb-Sr isochron ages")

    wr = rb_sr_isochron(WR_RB_SR, WR_SR_RATIO, "Whole-rock suite (layers A, B, D, G)")
    b = rb_sr_isochron(B_RB_SR, B_SR_RATIO, "Rock B mineral separates")
    g = rb_sr_isochron(G_RB_SR, G_SR_RATIO, "Rock G mineral separates")

    return {
        "dike_v": (age_v, err_v),
        "dike_w": (age_w, err_w),
        "whole_rock": wr,
        "rock_b": b,
        "rock_g": g,
    }


def plot_isochrons(results: dict) -> None:
    """Publication-style Rb-Sr isochron diagram for all sample suites."""
    wr, b, g = results["whole_rock"], results["rock_b"], results["rock_g"]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        WR_RB_SR,
        WR_SR_RATIO,
        s=120,
        c="steelblue",
        marker="o",
        label="Whole rock (A, B, D, G)",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        B_RB_SR,
        B_SR_RATIO,
        s=120,
        c="indianred",
        marker="s",
        label="Rock B minerals",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        G_RB_SR,
        G_SR_RATIO,
        s=120,
        c="seagreen",
        marker="^",
        label="Rock G minerals",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )

    for x, slope, intercept, age, color, style in [
        (
            np.linspace(0, 1.2, 100),
            wr["slope"],
            wr["initial_ratio"],
            wr["age_ma"],
            "steelblue",
            "-",
        ),
        (
            np.linspace(0, 5.5, 100),
            b["slope"],
            b["initial_ratio"],
            b["age_ma"],
            "indianred",
            "--",
        ),
        (
            np.linspace(0, 16, 100),
            g["slope"],
            g["initial_ratio"],
            g["age_ma"],
            "seagreen",
            "-.",
        ),
    ]:
        ax.plot(
            x,
            intercept + slope * x,
            color=color,
            linestyle=style,
            linewidth=2.5,
            alpha=0.85,
            label=f"{age:.0f} Ma",
        )

    ax.set_xlabel("⁸⁷Rb/⁸⁶Sr", fontsize=14, fontweight="bold")
    ax.set_ylabel("⁸⁷Sr/⁸⁶Sr", fontsize=14, fontweight="bold")
    ax.set_title(
        "Rb–Sr isochron diagram — metamorphosed volcanic terrane",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95)
    ax.set_xlim(0, 16)
    ax.set_ylim(0.70, None)
    clean_plot_style(ax)

    plt.tight_layout()
    fig.savefig(ISOCHRON_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved isochron figure: %s", ISOCHRON_FIG.name)


def log_summary_table(results: dict) -> None:
    """Tabular summary of all calculated ages."""
    log_section("Age summary")

    v, ev = results["dike_v"]
    w, ew = results["dike_w"]
    rows = [
        ("Dike V (zircon)", "U–Pb", f"{v:.1f} ± {ev:.1f}", "—"),
        ("Dike W (zircon)", "U–Pb", f"{w:.1f} ± {ew:.1f}", "—"),
        (
            "Whole rock (A,B,D,G)",
            "Rb–Sr",
            f"{results['whole_rock']['age_ma']:.1f}",
            f"{results['whole_rock']['initial_ratio']:.6f}",
        ),
        (
            "Rock B minerals",
            "Rb–Sr",
            f"{results['rock_b']['age_ma']:.1f}",
            f"{results['rock_b']['initial_ratio']:.6f}",
        ),
        (
            "Rock G minerals",
            "Rb–Sr",
            f"{results['rock_g']['age_ma']:.1f}",
            f"{results['rock_g']['initial_ratio']:.6f}",
        ),
    ]

    header = f"{'Sample':<28} {'Method':<12} {'Age (Ma)':<22} {'Initial ⁸⁷Sr/⁸⁶Sr'}"
    table_lines = [header, "-" * 70]
    table_lines.extend(
        f"{sample:<28} {method:<12} {age:<22} {initial}" for sample, method, age, initial in rows
    )
    log_block("\n".join(table_lines))


def log_geological_synthesis(results: dict) -> None:
    """Integrate isotopic ages into a coherent tectonic narrative."""
    wr_age = results["whole_rock"]["age_ma"]
    g_age = results["rock_g"]["age_ma"]
    age_v = results["dike_v"][0]
    age_w = results["dike_w"][0]
    initial_sr = results["whole_rock"]["initial_ratio"]

    log_section("Integrated geological timeline")
    log_block(
        f"""
  ~{wr_age:.0f} Ma  — Protolith crystallization (whole-rock Rb–Sr isochron).
                    Volcanic layers A–G record mantle-derived magmatism with
                    initial ⁸⁷Sr/⁸⁶Sr ≈ {initial_sr:.3f}.

  ~{g_age:.0f} Ma   — Peak metamorphism and Dike V emplacement.
                    Concordant U–Pb zircon age ({age_v:.0f} Ma) and mineral
                    isochrons indicate Rb–Sr resetting during cooling through
                    mineral closure temperatures.

  ~{age_w:.0f} Ma   — Post-metamorphic Dike W intrusion, crosscutting all
                    earlier structures; likely extension-related magmatism.

  Whole-rock systems remained closed at hand-specimen scale despite
  high-grade metamorphism, while mineral separates record younger
  thermal events — a classic pattern in long-lived continental crust.
""".strip()
    )


def main() -> None:
    radiogenic_heat_budget()
    results = upb_and_rbsr_ages()
    log_summary_table(results)
    plot_isochrons(results)
    log_geological_synthesis(results)
    logger.info("Case study complete.")


if __name__ == "__main__":
    main()
