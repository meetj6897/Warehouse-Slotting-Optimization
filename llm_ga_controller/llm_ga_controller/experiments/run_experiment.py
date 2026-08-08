"""
experiments/run_experiment.py
==============================
Entry point for the LLM-Guided Adaptive GA experiment.

Pipeline per interval
---------------------
  1. Run GA for LLM_INTERVAL generations  (ga/runner.py)
  2. StateMonitor derives trend metrics    (utils/state_monitor.py)
  3. PromptBuilder assembles the prompt   (llm/prompt_builder.py)
  4. LLMAgent queries Claude              (llm/agent.py)
  5. SafetyValidator clamps & applies     (llm/safety_validator.py)
  6. ExperienceLog records the event      (memory/experience_log.py)
  7. After next interval: record outcome  (memory/experience_log.py)

Usage
-----
    # Basic run (uses defaults from config/settings.py)
    python experiments/run_experiment.py

    # Custom settings
    python experiments/run_experiment.py \
        --generations 300 \
        --pop-size 200 \
        --llm-interval 10 \
        --policy GZA \
        --seed 30
"""

from __future__ import annotations

import argparse
import sys
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Make project root importable when running as a script ─────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import GAParams, LLM_INTERVAL, LLM_HISTORY_WINDOW
from ga.population import generate_reorder_rates, generate_orders
from ga.runner import initialise_ga, run_ga_interval
from ga.warehouse import build_location_dicts
from llm.agent import LLMAgent
from llm.safety_validator import validate_and_apply
from memory.experience_log import ExperienceLog
from utils.logger import (
    setup_logging,
    print_section,
    print_llm_decision,
    print_generation_summary,
)
from utils.state_monitor import StateMonitor

logger = logging.getLogger(__name__)


# ── CLI argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-Guided Adaptive Meta-Controller for Warehouse GA"
    )
    p.add_argument("--generations",   type=int,   default=200,    help="Total GA generations")
    p.add_argument("--pop-size",      type=int,   default=200,    help="Initial population size")
    p.add_argument("--mutation-rate", type=float, default=0.5,    help="Initial mutation rate")
    p.add_argument("--elitism",       type=int,   default=20,     help="Elitism count")
    p.add_argument("--llm-interval",  type=int,   default=LLM_INTERVAL,
                   help="Consult LLM every N generations")
    p.add_argument("--history-window",type=int,   default=LLM_HISTORY_WINDOW,
                   help="Number of past generations to include in LLM prompt")
    p.add_argument("--policy",        choices=["GZA", "W"], default="GZA",
                   help="Storage policy: GZA (gold-zone) or W (random)")
    p.add_argument("--seed",          type=int,   default=30,     help="Random seed")
    p.add_argument("--no-llm",        action="store_true",
                   help="Disable LLM controller (plain GA baseline)")
    p.add_argument("--verbose",       action="store_true",
                   help="Enable DEBUG logging to console")
    p.add_argument("--outcome-lag",   type=int,   default=10,
                   help="Generations after intervention before measuring outcome")
    return p.parse_args()


# ── Main experiment ───────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    console_level = logging.DEBUG if args.verbose else logging.INFO
    log_path = setup_logging(console_level=console_level, run_id=run_id)

    print_section(f"LLM-Guided Adaptive GA  ·  run_id={run_id}")
    logger.info("Arguments: %s", vars(args))

    # ── 1. Build warehouse geometry ───────────────────────────────────────────
    logger.info("Building warehouse geometry (policy=%s) …", args.policy)
    sku_location_3d, sku_location_2d = build_location_dicts(policy=args.policy)
    logger.info(
        "Warehouse ready: %d 3-D slots, %d 2-D positions",
        len(sku_location_3d), len(sku_location_2d),
    )

    # ── 2. Simulate demand & orders ───────────────────────────────────────────
    logger.info("Generating reorder rates and orders (seed=%d) …", args.seed)
    reorder_rates    = generate_reorder_rates(seed=args.seed)
    sku_reorder_rate = {i: r for i, r in enumerate(reorder_rates)}
    sku_ids          = list(range(len(reorder_rates)))
    orders           = generate_orders(sku_ids, sku_reorder_rate, seed=args.seed)
    logger.info("Orders ready: %d orders, %d unique SKUs", len(orders), len(sku_ids))

    # ── 3. Initialise GA parameters & state ───────────────────────────────────
    params = GAParams(
        pop_size      = args.pop_size,
        generations   = args.generations,
        mutation_rate = args.mutation_rate,
        elitism_count = args.elitism,
    )
    ga_state = initialise_ga(params, sku_location_3d)
    logger.info("GA initialised | pop=%d | μ=%.3f | elitism=%d",
                params.pop_size, params.mutation_rate, params.elitism_count)

    # ── 4. Initialise supporting components ───────────────────────────────────
    monitor  = StateMonitor(
        window_size       = args.history_window,
        total_generations = args.generations,
    )
    exp_log  = ExperienceLog(run_id=run_id)
    llm_agent = None if args.no_llm else LLMAgent()

    if args.no_llm:
        logger.info("LLM controller DISABLED — running plain GA baseline.")
    else:
        logger.info("LLM controller ENABLED — model=%s | interval=%d gens",
                    "claude-sonnet-4-6", args.llm_interval)

    # Tracks pending interventions waiting for outcome measurement
    pending_interventions = []   # list of (InterventionRecord, trigger_gen)

    # ── 5. Main generation loop ───────────────────────────────────────────────
    current_gen = 0
    print_section("Starting Evolution")

    while current_gen < args.generations:
        interval = min(args.llm_interval, args.generations - current_gen)

        # Run one interval of the GA
        ga_state, results = run_ga_interval(
            state           = ga_state,
            params          = params,
            orders          = orders,
            sku_location_2d = sku_location_2d,
            sku_location_3d = sku_location_3d,
            start_gen       = current_gen,
            n_generations   = interval,
        )

        # Log each generation to console
        for r in results:
            print_generation_summary(
                generation    = r.generation,
                best_fitness  = r.best_fitness,
                avg_fitness   = r.avg_fitness,
                diversity     = r.diversity,
                mutation_rate = r.mutation_rate,
                elapsed       = r.elapsed_seconds,
            )

        current_gen += interval
        monitor.update(results)

        # ── Check if any pending interventions now have outcome data ──────────
        still_pending = []
        for record, trigger_gen in pending_interventions:
            if current_gen >= trigger_gen + args.outcome_lag:
                exp_log.update_outcome(
                    record             = record,
                    fitness_after      = ga_state.best_distance,
                    outcome_generation = current_gen,
                )
            else:
                still_pending.append((record, trigger_gen))
        pending_interventions = still_pending

        # ── LLM consultation ──────────────────────────────────────────────────
        if llm_agent is None or current_gen >= args.generations:
            continue

        opt_state = monitor.compute()
        if opt_state is None:
            continue

        print_section(f"LLM Consultation  ·  Generation {current_gen}")
        logger.info(
            "Querying LLM | trend=%s | stagnation=%d | diversity=%.3f",
            opt_state.trend, opt_state.stagnation_count, opt_state.diversity,
        )

        recommendation = llm_agent.query(opt_state)
        report         = validate_and_apply(recommendation, params)

        print_llm_decision(
            generation       = current_gen,
            phase            = recommendation.phase_assessment,
            root_cause       = recommendation.root_cause,
            reasoning        = recommendation.reasoning,
            expected_outcome = recommendation.expected_outcome,
            confidence       = recommendation.confidence,
            report_summary   = report.summary(),
        )

        # Record intervention for outcome measurement
        if recommendation.has_changes():
            record = exp_log.record_intervention(opt_state, recommendation, report)
            pending_interventions.append((record, current_gen))

    # ── 6. Flush any remaining pending outcomes ───────────────────────────────
    exp_log.flush_pending(
        current_fitness    = ga_state.best_distance,
        current_generation = current_gen,
    )

    # ── 7. Final report ───────────────────────────────────────────────────────
    print_section("Run Complete")
    logger.info("Best fitness achieved : %.2f", ga_state.best_distance)
    logger.info("Total generations     : %d",   current_gen)
    logger.info("Log file              : %s",   log_path)

    stats = exp_log.summary_stats()
    logger.info(
        "Interventions | total=%d | success=%d | neutral=%d | harmful=%d | avg_delta=%.1f",
        stats.get("total", 0),
        stats.get("success", 0),
        stats.get("neutral", 0),
        stats.get("harmful", 0),
        stats.get("avg_delta") or 0.0,
    )

    print(f"\n  ✓  Best fitness  : {ga_state.best_distance:,.2f}")
    print(f"  ✓  Best route    : {(ga_state.best_route or [])[:10]} …")
    print(f"  ✓  Log           : {log_path}\n")

    return ga_state.best_distance, ga_state.best_route


if __name__ == "__main__":
    main()
