"""
utils/state_monitor.py
======================
Computes summary statistics from a sliding window of GenerationResults.
These metrics are fed into the prompt builder to give the LLM
quantitative evidence of what the GA is doing.

Metrics produced
----------------
  stagnation_count   : Consecutive generations with no improvement
  improvement_pct    : % improvement in best fitness over the window
  trend              : "improving" | "stagnating" | "oscillating"
  phase              : "early" | "mid" | "late"  (based on gen / total_gens)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ga.runner import GenerationResult


@dataclass
class OptimisationState:
    """Aggregated state summary passed to the prompt builder."""
    # Current generation values
    generation:      int
    best_fitness:    float
    avg_fitness:     float
    worst_fitness:   float
    fitness_std:     float
    diversity:       float
    mutation_rate:   float
    elitism_count:   int
    crossover_op:    str
    mutation_op:     str
    population_size: int
    runtime_seconds: float

    # Derived / trend metrics
    stagnation_count: int
    improvement_pct:  float          # over the LLM history window
    trend:            str            # "improving" | "stagnating" | "oscillating"
    phase:            str            # "early" | "mid" | "late"

    # Recent history (for the prompt table)
    history_window:  List[dict]      # list of per-generation dicts


class StateMonitor:
    """Maintains a rolling window of results and derives trend metrics."""

    def __init__(self, window_size: int = 10, total_generations: int = 200):
        self.window_size       = window_size
        self.total_generations = total_generations
        self._buffer: List[GenerationResult] = []
        self._cumulative_runtime: float = 0.0

    def update(self, results: List[GenerationResult]) -> None:
        """Add new results from the latest interval to the buffer."""
        self._buffer.extend(results)
        self._cumulative_runtime += sum(r.elapsed_seconds for r in results)
        # Keep only the last `window_size` results
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]

    def compute(self) -> Optional[OptimisationState]:
        """Derive the current OptimisationState from buffered results.

        Returns None if not enough data has accumulated yet.
        """
        if not self._buffer:
            return None

        latest  = self._buffer[-1]
        oldest  = self._buffer[0]

        # ── Stagnation: count consecutive gens with no improvement ──────────
        best_values = [r.best_fitness for r in self._buffer]
        stagnation  = 0
        for i in range(len(best_values) - 1, 0, -1):
            if best_values[i] >= best_values[i - 1]:
                stagnation += 1
            else:
                break

        # ── Improvement % over the window ────────────────────────────────────
        if oldest.best_fitness > 0:
            improvement_pct = (
                (oldest.best_fitness - latest.best_fitness)
                / oldest.best_fitness * 100
            )
        else:
            improvement_pct = 0.0

        # ── Trend classification ──────────────────────────────────────────────
        if improvement_pct > 0.5:
            trend = "improving"
        elif stagnation >= max(3, self.window_size // 2):
            trend = "stagnating"
        else:
            # Check for oscillation: fitness went up and down
            diffs = [best_values[i] - best_values[i-1] for i in range(1, len(best_values))]
            sign_changes = sum(
                1 for i in range(1, len(diffs))
                if diffs[i] * diffs[i-1] < 0
            )
            trend = "oscillating" if sign_changes >= 2 else "stagnating"

        # ── Phase (early / mid / late) ────────────────────────────────────────
        pct_done = latest.generation / max(self.total_generations, 1)
        if pct_done < 0.33:
            phase = "early"
        elif pct_done < 0.67:
            phase = "mid"
        else:
            phase = "late"

        # ── History window for the prompt table ──────────────────────────────
        history_window = [
            {
                "generation":   r.generation,
                "best_fitness": round(r.best_fitness, 1),
                "avg_fitness":  round(r.avg_fitness, 1),
                "diversity":    round(r.diversity, 3),
                "mutation_rate":round(r.mutation_rate, 3),
                "elapsed_s":    round(r.elapsed_seconds, 2),
            }
            for r in self._buffer
        ]

        return OptimisationState(
            generation       = latest.generation,
            best_fitness     = latest.best_fitness,
            avg_fitness      = latest.avg_fitness,
            worst_fitness    = latest.worst_fitness,
            fitness_std      = latest.fitness_std,
            diversity        = latest.diversity,
            mutation_rate    = latest.mutation_rate,
            elitism_count    = latest.elitism_count,
            crossover_op     = latest.crossover_op,
            mutation_op      = latest.mutation_op,
            population_size  = latest.population_size,
            runtime_seconds  = self._cumulative_runtime,
            stagnation_count = stagnation,
            improvement_pct  = round(improvement_pct, 3),
            trend            = trend,
            phase            = phase,
            history_window   = history_window,
        )
