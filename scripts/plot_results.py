"""Generates the plots included in the review bundle, from
results/calibration/calibration_summary.csv and results/sweep/sweep_summary.csv.

Usage: python scripts/plot_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "plots"
CONFIG_ORDER = ["W1T8", "W2T4", "W4T2", "W8T1"]
COLORS = {"W1T8": "#4C72B0", "W2T4": "#DD8452", "W4T2": "#55A868", "W8T1": "#C44E52"}


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_calibration_throughput(rows: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for config in CONFIG_ORDER:
        sub = sorted((r for r in rows if r["config"] == config), key=lambda r: float(r["rate_hz_nominal"]))
        rates = [float(r["rate_hz_nominal"]) for r in sub]
        throughput = [float(r["throughput_completed_per_s"]) for r in sub]
        p99 = [float(r["e2e_p99_ms"]) for r in sub]
        ax1.plot(rates, throughput, marker="o", label=config, color=COLORS[config])
        ax2.plot(rates, p99, marker="o", label=config, color=COLORS[config])
    ax1.plot([0, 220], [0, 220], "k--", linewidth=0.8, label="offered = completed (no saturation)")
    ax1.set_xlabel("offered rate (req/s)")
    ax1.set_ylabel("completed throughput (req/s)")
    ax1.set_title("Calibration: throughput vs. offered rate (tuning split)")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("offered rate (req/s)")
    ax2.set_ylabel("e2e p99 latency (ms)")
    ax2.set_yscale("log")
    ax2.set_title("Calibration: p99 latency vs. offered rate (log scale)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_throughput_and_latency.png", dpi=130)
    plt.close(fig)


def plot_sweep_latency_by_rate(rows: list[dict]) -> None:
    rate_labels = ["low", "medium", "stress"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for ax, rate_label in zip(axes, rate_labels):
        sub = [r for r in rows if r["rate_label"] == rate_label]
        for i, config in enumerate(CONFIG_ORDER):
            vals = [float(r["e2e_p50_ms"]) for r in sub if r["config"] == config]
            vals99 = [float(r["e2e_p99_ms"]) for r in sub if r["config"] == config]
            ax.scatter([i - 0.1] * len(vals), vals, color=COLORS[config], marker="o", label="p50" if i == 0 else None)
            ax.scatter([i + 0.1] * len(vals99), vals99, color=COLORS[config], marker="^", label="p99" if i == 0 else None)
        ax.set_xticks(range(len(CONFIG_ORDER)))
        ax.set_xticklabels(CONFIG_ORDER)
        ax.set_title(f"rate={rate_label}")
        ax.set_ylabel("e2e latency (ms)")
        if rate_label == "stress":
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched sweep: e2e p50 (circle) / p99 (triangle) by config and rate, 3 repeats each (eval split)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep_latency_by_config_and_rate.png", dpi=130)
    plt.close(fig)


def plot_recheck_p50_p99(rows: list[dict]) -> None:
    rate_order = ["low", "medium", "stress"]
    rate_hz = {"low": 40, "medium": 90, "stress": 130}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for config in ["W2T4", "W4T2"]:
        p50_by_rate = []
        p99_by_rate = []
        for rate_label in rate_order:
            vals50 = [float(r["e2e_p50_ms"]) for r in rows if r["config"] == config and r["rate_label"] == rate_label]
            vals99 = [float(r["e2e_p99_ms"]) for r in rows if r["config"] == config and r["rate_label"] == rate_label]
            p50_by_rate.append(vals50)
            p99_by_rate.append(vals99)
        x = [rate_hz[r] for r in rate_order]
        p50_mean = [sum(v) / len(v) for v in p50_by_rate]
        p99_mean = [sum(v) / len(v) for v in p99_by_rate]
        color = COLORS[config]
        ax1.plot(x, p50_mean, marker="o", color=color, label=config)
        for xi, vals in zip(x, p50_by_rate):
            ax1.scatter([xi] * len(vals), vals, color=color, alpha=0.4, s=15)
        ax2.plot(x, p99_mean, marker="o", color=color, label=config)
        for xi, vals in zip(x, p99_by_rate):
            ax2.scatter([xi] * len(vals), vals, color=color, alpha=0.4, s=15)
    ax1.set_xlabel("offered rate (req/s)")
    ax1.set_ylabel("e2e p50 latency (ms)")
    ax1.set_title("Recheck: p50 (mean of 3 seeds, dots = individual runs)")
    ax1.legend()
    ax2.set_xlabel("offered rate (req/s)")
    ax2.set_ylabel("e2e p99 latency (ms)")
    ax2.set_title("Recheck: p99 -- the actual crossover")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recheck_w2t4_vs_w4t2.png", dpi=130)
    plt.close(fig)


def plot_sustained_phases(analysis: dict) -> None:
    phase_order = ["low1", "high", "low2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, phase_id in zip(axes, phase_order):
        for config in ["W2T4", "W4T2"]:
            p50s, p99s = [], []
            for run_id, d in analysis["sustained"].items():
                if d["config"] != config:
                    continue
                # per-phase percentiles are computed by smoke_bench.py itself
                # (latency_ms_by_phase); re-derive here from the run's own
                # summary.json rather than the deadline-analysis file, which
                # only carries deadline-miss data.
                summary = json.loads((REPO_ROOT / "results" / "time_varying" / run_id / "summary.json").read_text())
                lat = summary["latency_ms_by_phase"][phase_id]["e2e"]
                p50s.append(lat["p50"])
                p99s.append(lat["p99"])
            color = COLORS[config]
            ax.scatter([0] * len(p50s), p50s, color=color, marker="o", label=f"{config} p50" if phase_id == "low1" else None)
            ax.scatter([1] * len(p99s), p99s, color=color, marker="^", label=f"{config} p99" if phase_id == "low1" else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["p50", "p99"])
        ax.set_ylabel("e2e latency (ms)")
        ax.set_title(f"phase: {phase_id}")
    axes[0].legend(fontsize=7)
    fig.suptitle("Sustained low-high-low (40↚130↚40 req/s): per-phase e2e latency, 2 seeds each")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sustained_by_phase.png", dpi=130)
    plt.close(fig)


def plot_burst_p99(analysis: dict) -> None:
    durations = [0.1, 0.5, 2.0, 10.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for config in ["W2T4", "W4T2"]:
        p99s = []
        ns = []
        for d in durations:
            run_id = f"burst_{d}s_{config}"
            entry = analysis["burst"][run_id]["pooled_by_group"]["burst"]["e2e_latency_ms_pooled"]
            p99s.append(entry["p99"])
            ns.append(entry["n"])
        ax.plot(durations, p99s, marker="o", color=COLORS[config], label=config)
        for x, y, n in zip(durations, p99s, ns):
            ax.annotate(f"n={n}", (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log")
    ax.set_xlabel("burst duration (s)")
    ax.set_ylabel("pooled burst-window e2e p99 (ms)")
    ax.set_title("Burst response: pooled p99 across repeated bursts (baseline 40, burst 130 req/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "burst_p99_by_duration.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calib_rows = read_csv(REPO_ROOT / "results" / "calibration" / "calibration_summary.csv")
    sweep_rows = read_csv(REPO_ROOT / "results" / "sweep" / "sweep_summary.csv")
    plot_calibration_throughput(calib_rows)
    plot_sweep_latency_by_rate(sweep_rows)
    recheck_path = REPO_ROOT / "results" / "recheck" / "recheck_summary.csv"
    if recheck_path.exists():
        plot_recheck_p50_p99(read_csv(recheck_path))
    analysis_path = REPO_ROOT / "results" / "time_varying" / "analysis_summary.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        plot_sustained_phases(analysis)
        plot_burst_p99(analysis)
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
