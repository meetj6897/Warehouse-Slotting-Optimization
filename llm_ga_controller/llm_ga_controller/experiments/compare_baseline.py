"""
experiments/compare_baseline.py
================================
Runs two GA experiments back-to-back:
  1. Plain GA  (no LLM, fixed hyperparameters)
  2. LLM-guided GA (adaptive hyperparameters)

Then plots a side-by-side convergence comparison and prints a summary table.

Usage
-----
    python experiments/compare_baseline.py --generations 200 --seed 30
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import GAParams, LLM_INTERVAL, LLM_HISTORY_WINDOW
from ga.population import generate_reorder_rates, generate_orders
from ga.runner import initialise_ga, run_ga_interval
from ga.warehouse import build_location_dicts
from llm.agent import LLMAgent
from llm.safety_validator import validate_and_apply
from utils.logger import setup_logging
from utils.state_monitor import StateMonitor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs LLM-GA comparison")
    p.add_argument("--generations",   type=int,   default=1500)
    p.add_argument("--pop-size",      type=int,   default=100)
    p.add_argument("--mutation-rate", type=float, default=0.7)
    p.add_argument("--llm-interval",  type=int,   default=LLM_INTERVAL)
    p.add_argument("--policy",        choices=["GZA", "W"], default="GZA")
    p.add_argument("--seed",          type=int,   default=30)
    p.add_argument("--output",        type=str,   default="logs/comparison.png")
    return p.parse_args()


def run_plain_ga(
    generations:     int,
    pop_size:        int,
    mutation_rate:   float,
    orders:          list,
    sku_location_2d: dict,
    sku_location_3d: dict,
) -> list:
    """Run the GA with fixed parameters; return fitness history."""
    params   = GAParams(pop_size=pop_size, generations=generations,
                        mutation_rate=mutation_rate)
    ga_state = initialise_ga(params, sku_location_3d)

    ga_state, results = run_ga_interval(
        state           = ga_state,
        params          = params,
        orders          = orders,
        sku_location_2d = sku_location_2d,
        sku_location_3d = sku_location_3d,
        start_gen       = 0,
        n_generations   = generations,
    )
    return ga_state.history, ga_state.best_distance


def run_llm_ga(
    generations:     int,
    pop_size:        int,
    mutation_rate:   float,
    llm_interval:    int,
    orders:          list,
    sku_location_2d: dict,
    sku_location_3d: dict,
) -> list:
    """Run the LLM-controlled GA; return fitness history."""
    params   = GAParams(pop_size=pop_size, generations=generations,
                        mutation_rate=mutation_rate)
    ga_state = initialise_ga(params, sku_location_3d)
    monitor  = StateMonitor(window_size=LLM_HISTORY_WINDOW,
                            total_generations=generations)
    agent    = LLMAgent()

    current_gen = 0
    while current_gen < generations:
        interval = min(llm_interval, generations - current_gen)

        ga_state, results = run_ga_interval(
            state           = ga_state,
            params          = params,
            orders          = orders,
            sku_location_2d = sku_location_2d,
            sku_location_3d = sku_location_3d,
            start_gen       = current_gen,
            n_generations   = interval,
        )
        current_gen += interval
        monitor.update(results)

        if current_gen >= generations:
            break

        opt_state = monitor.compute()
        if opt_state:
            logger.info("LLM consultation at gen %d …", current_gen)
            rec    = agent.query(opt_state)
            report = validate_and_apply(rec, params)
            logger.info("Changes: %s", report.summary())

    return ga_state.history, ga_state.best_distance


def plot_comparison(
    plain_history: list,
    llm_history:   list,
    output_path:   str,
    generations:   int,
) -> None:
    """Save a side-by-side convergence plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = range(generations)

    # ── Left: both curves on one axis ────────────────────────────────────────
    axes[0].plot(plain_history, label="Plain GA (fixed params)", color="steelblue", lw=2)
    axes[0].plot(llm_history,   label="LLM-Guided GA",           color="darkorange", lw=2)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best Fitness (lower = better)")
    axes[0].set_title("Convergence Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # ── Right: improvement gap ────────────────────────────────────────────────
    n = min(len(plain_history), len(llm_history))
    gap = [plain_history[i] - llm_history[i] for i in range(n)]
    colour = ["green" if g >= 0 else "red" for g in gap]
    axes[1].bar(range(n), gap, color=colour, alpha=0.7, width=1.0)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Plain GA Fitness − LLM GA Fitness")
    axes[1].set_title("LLM Advantage per Generation\n(positive = LLM is better)")
    axes[1].grid(True, alpha=0.4, axis="y")

    plt.suptitle("LLM-Guided GA vs Plain GA  ·  Warehouse Order-Picking",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Comparison plot saved → %s", output_path)
    plt.show()


def main() -> None:
    args = parse_args()
    setup_logging()

    # ── Shared setup ──────────────────────────────────────────────────────────
    logger.info("Building warehouse (policy=%s) …", args.policy)
    sku_location_3d, sku_location_2d = build_location_dicts(policy=args.policy)

    reorder_rates    = generate_reorder_rates(seed=args.seed)
    sku_reorder_rate = {i: r for i, r in enumerate(reorder_rates)}
    sku_ids          = list(range(len(reorder_rates)))
    orders           = generate_orders(sku_ids, sku_reorder_rate, seed=args.seed)

    # ── Run 1: Plain GA ───────────────────────────────────────────────────────
    logger.info("═══ RUN 1: Plain GA (baseline) ═══")
    plain_history, plain_best = run_plain_ga(
        generations     = args.generations,
        pop_size        = args.pop_size,
        mutation_rate   = args.mutation_rate,
        orders          = orders,
        sku_location_2d = sku_location_2d,
        sku_location_3d = sku_location_3d,
    )
    logger.info("Plain GA best fitness: %.2f", plain_best)

    # ── Run 2: LLM-Guided GA ─────────────────────────────────────────────────
    logger.info("═══ RUN 2: LLM-Guided GA ═══")
    llm_history, llm_best = run_llm_ga(
        generations     = args.generations,
        pop_size        = args.pop_size,
        mutation_rate   = args.mutation_rate,
        llm_interval    = args.llm_interval,
        orders          = orders,
        sku_location_2d = sku_location_2d,
        sku_location_3d = sku_location_3d,
    )
    logger.info("LLM-Guided GA best fitness: %.2f", llm_best)

    # ── Summary table ─────────────────────────────────────────────────────────
    improvement = (plain_best - llm_best) / plain_best * 100
    print("\n" + "═" * 60)
    print("  COMPARISON SUMMARY")
    print("═" * 60)
    print(f"  Plain GA best fitness   : {plain_best:>14,.2f}")
    print(f"  LLM-Guided best fitness : {llm_best:>14,.2f}")
    print(f"  Improvement             : {improvement:>+13.2f}%")
    print("═" * 60 + "\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(plain_history, llm_history, args.output, args.generations)


if __name__ == "__main__":
    main()
