"""
llm/prompt_builder.py
=====================
Constructs the two-part prompt sent to the LLM Meta-Controller:

  SYSTEM PROMPT  — describes the agent's role, allowed parameters,
                   bounds, and the exact JSON format it must return.

  USER PROMPT    — the current optimisation state + history table,
                   assembled fresh each invocation.

Design principles
-----------------
  - The LLM never sees raw chromosomes (too large, unnecessary).
  - All numerical context is pre-computed by StateMonitor.
  - The requested output format is JSON — parsed deterministically by agent.py.
  - Bounds are stated explicitly in the system prompt AND enforced by the
    safety validator, giving the LLM correct expectations and a safety net.
"""

from __future__ import annotations

import json
from string import Template
from typing import Any, Dict

from config.settings import (
    ALLOWED_CROSSOVER_OPS,
    ALLOWED_MUTATION_OPS,
    PARAM_BOUNDS,
)
from utils.state_monitor import OptimisationState


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (fixed for the entire run)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an expert Genetic Algorithm (GA) optimisation controller for a warehouse
order-picking problem.  Your goal is to adaptively tune the GA's hyperparameters
so that it converges faster to a lower total fitness (distance + retrieval time).

────────────────────────────────────────────────────────────────────────────────
PARAMETERS YOU MAY CHANGE
────────────────────────────────────────────────────────────────────────────────
  mutation_rate   float   Allowed range: {PARAM_BOUNDS["mutation_rate"]}
  pop_size        int     Allowed range: {PARAM_BOUNDS["pop_size"]}
  elitism_count   int     Allowed range: {PARAM_BOUNDS["elitism_count"]}
  crossover_op    str     One of: {ALLOWED_CROSSOVER_OPS}
  mutation_op     str     One of: {ALLOWED_MUTATION_OPS}

Never recommend values outside the allowed ranges above.

────────────────────────────────────────────────────────────────────────────────
WAREHOUSE CONTEXT
────────────────────────────────────────────────────────────────────────────────
  - 500 SKUs stored across 5 racks × 25 columns × 4 vertical levels.
  - Fitness = total S-shape travel distance + total vertical retrieval time.
  - Lower fitness is better.
  - Slow-moving SKUs increase retrieval time when placed at ground (z=0) or
    top (z=3) levels; fast-moving SKUs benefit most from mid-level (z=1, z=2).
  - The chromosome encodes SKU-to-slot assignments as a permutation.

────────────────────────────────────────────────────────────────────────────────
REQUIRED OUTPUT FORMAT  (return ONLY valid JSON — no prose, no markdown)
────────────────────────────────────────────────────────────────────────────────
{{
  "phase_assessment": "<one of: early_exploration | balanced_search | exploitation | stagnation | premature_convergence | oscillating>",
  "root_cause": "<1-2 sentence explanation of why performance is behaving as observed>",
  "recommendations": {{
    "mutation_rate":  <float or null if no change>,
    "pop_size":       <int   or null if no change>,
    "elitism_count":  <int   or null if no change>,
    "crossover_op":   <str   or null if no change>,
    "mutation_op":    <str   or null if no change>
  }},
  "reasoning": "<1-3 sentences explaining each non-null recommendation>",
  "expected_outcome": "<1 sentence predicting the effect of these changes>",
  "confidence": <float between 0.0 and 1.0>
}}

If no parameter change is needed, set all recommendation values to null.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT  (assembled fresh each LLM call)
# ─────────────────────────────────────────────────────────────────────────────

_USER_TEMPLATE = Template("""
════════════════════════════════════════════════════════════════════════════════
CURRENT OPTIMISATION STATE  (Generation $generation)
════════════════════════════════════════════════════════════════════════════════
Best Fitness     : $best_fitness
Average Fitness  : $avg_fitness
Worst Fitness    : $worst_fitness
Fitness Std Dev  : $fitness_std
Population Size  : $population_size
Diversity        : $diversity   (0 = fully converged, 1 = all unique)

Current Parameters
  Mutation Rate  : $mutation_rate
  Elitism Count  : $elitism_count
  Crossover Op   : $crossover_op
  Mutation Op    : $mutation_op

Trend Metrics
  Stagnation     : $stagnation_count consecutive generations with no improvement
  Improvement    : $improvement_pct% over the last $window_size generations
  Trend          : $trend
  Phase          : $phase  (fraction of total generations elapsed)
  Runtime        : $runtime_seconds seconds so far

════════════════════════════════════════════════════════════════════════════════
HISTORY — LAST $window_size GENERATIONS
════════════════════════════════════════════════════════════════════════════════
$history_table

════════════════════════════════════════════════════════════════════════════════
YOUR TASKS
════════════════════════════════════════════════════════════════════════════════
1. Identify the current optimisation phase.
2. Explain the root cause of any performance issue.
3. Recommend parameter changes (or confirm current settings are appropriate).
4. Predict the expected outcome of your recommendation.
5. Provide a confidence score.

Return ONLY the JSON object described in the system prompt.
""")


def _format_history_table(history_window: list) -> str:
    """Render the history window as a plain-text table."""
    if not history_window:
        return "  (no history yet)"

    header = (
        f"  {'Gen':>5}  {'Best Fit':>14}  {'Avg Fit':>14}  "
        f"{'Diversity':>10}  {'Mut Rate':>9}  {'Time(s)':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    rows = [header, sep]
    for row in history_window:
        rows.append(
            f"  {row['generation']:>5}  "
            f"{row['best_fitness']:>14,.1f}  "
            f"{row['avg_fitness']:>14,.1f}  "
            f"{row['diversity']:>10.3f}  "
            f"{row['mutation_rate']:>9.3f}  "
            f"{row['elapsed_s']:>7.2f}"
        )
    return "\n".join(rows)


def build_user_prompt(state: OptimisationState) -> str:
    """Assemble the user-facing prompt from the current optimisation state."""
    history_table = _format_history_table(state.history_window)

    return _USER_TEMPLATE.substitute(
        generation       = state.generation,
        best_fitness     = f"{state.best_fitness:,.1f}",
        avg_fitness      = f"{state.avg_fitness:,.1f}",
        worst_fitness    = f"{state.worst_fitness:,.1f}",
        fitness_std      = f"{state.fitness_std:,.1f}",
        population_size  = state.population_size,
        diversity        = f"{state.diversity:.3f}",
        mutation_rate    = f"{state.mutation_rate:.3f}",
        elitism_count    = state.elitism_count,
        crossover_op     = state.crossover_op,
        mutation_op      = state.mutation_op,
        stagnation_count = state.stagnation_count,
        improvement_pct  = f"{state.improvement_pct:.2f}",
        window_size      = len(state.history_window),
        trend            = state.trend,
        phase            = state.phase,
        runtime_seconds  = f"{state.runtime_seconds:.1f}",
        history_table    = history_table,
    ).strip()


def get_system_prompt() -> str:
    """Return the fixed system prompt."""
    return SYSTEM_PROMPT
