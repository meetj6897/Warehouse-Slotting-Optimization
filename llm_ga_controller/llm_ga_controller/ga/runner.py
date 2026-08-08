"""
ga/runner.py
============
The main GA loop — refactored to accept a *mutable* GAParams object so the
LLM Meta-Controller can update hyperparameters between intervals without
restarting the algorithm.

Key differences from the original notebook:
  - `genetic_algorithm()` now yields a GenerationResult after every generation
    instead of running to completion in one call. The caller (run_experiment.py)
    drives the loop, which lets it hand off to the LLM controller at any point.
  - Crossover and mutation operators are resolved by name from `operators.py`
    registries, so the LLM can switch them at runtime.
  - All warehouse state (location dicts, orders) is passed in explicitly —
    no module-level globals.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from config.settings import GAParams
from ga.fitness import evaluate_population
from ga.operators import CROSSOVER_OPS, MUTATION_OPS
from ga.population import elitism_selection, initial_population


# ── Per-generation result ─────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Snapshot of one GA generation — consumed by StateMonitor and logger."""
    generation:       int
    best_fitness:     float
    avg_fitness:      float
    worst_fitness:    float
    fitness_std:      float
    diversity:        float          # fraction of unique chromosomes in population
    mutation_rate:    float
    elitism_count:    int
    crossover_op:     str
    mutation_op:      str
    population_size:  int
    elapsed_seconds:  float
    best_chromosome:  List[int]      # copy of the best chromosome


# ── GA state (shared across interval calls) ────────────────────────────────────

@dataclass
class GAState:
    """Persistent state passed between interval calls."""
    population:    List[List[int]]
    best_route:    Optional[List[int]]
    best_distance: float
    history:       List[float] = field(default_factory=list)


# ── Population diversity ───────────────────────────────────────────────────────

def _population_diversity(population: List[List[int]]) -> float:
    """Fraction of chromosomes that are unique (0 = fully converged, 1 = all different)."""
    unique = {tuple(c) for c in population}
    return len(unique) / len(population)


# ── One generation ────────────────────────────────────────────────────────────

def _run_one_generation(
    state:          GAState,
    params:         GAParams,
    orders:         List[List[int]],
    sku_location_2d: Dict,
    sku_location_3d: Dict,
    generation_num: int,
) -> Tuple[GAState, GenerationResult]:
    """Execute a single generation and return updated state + result snapshot."""
    t_start = time.perf_counter()

    crossover_fn = CROSSOVER_OPS[params.crossover_op]
    mutation_fn  = MUTATION_OPS[params.mutation_op]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    fitness_values = evaluate_population(
        state.population, orders, sku_location_2d, sku_location_3d
    )

    # ── Track global best ─────────────────────────────────────────────────────
    gen_best_idx = int(np.argmin(fitness_values))
    gen_best_fit = fitness_values[gen_best_idx]
    if gen_best_fit < state.best_distance:
        state.best_distance = gen_best_fit
        state.best_route    = state.population[gen_best_idx][:]

    state.history.append(state.best_distance)

    # ── Selection ─────────────────────────────────────────────────────────────
    elite, _ = elitism_selection(
        state.population, fitness_values, params.elitism_count
    )

    # ── Crossover ─────────────────────────────────────────────────────────────
    offspring: List[List[int]] = elite[:]   # elites carry forward unchanged
    while len(offspring) < params.pop_size:
        p1, p2 = random.sample(elite, 2)
        offspring.append(crossover_fn(p1, p2))

    # ── Mutation ──────────────────────────────────────────────────────────────
    state.population = [
        mutation_fn(ind, params.mutation_rate) for ind in offspring
    ]

    # ── Build result snapshot ─────────────────────────────────────────────────
    fv   = np.array(fitness_values)
    result = GenerationResult(
        generation      = generation_num,
        best_fitness    = float(fv.min()),
        avg_fitness     = float(fv.mean()),
        worst_fitness   = float(fv.max()),
        fitness_std     = float(fv.std()),
        diversity       = _population_diversity(state.population),
        mutation_rate   = params.mutation_rate,
        elitism_count   = params.elitism_count,
        crossover_op    = params.crossover_op,
        mutation_op     = params.mutation_op,
        population_size = params.pop_size,
        elapsed_seconds = time.perf_counter() - t_start,
        best_chromosome = state.best_route[:] if state.best_route else [],
    )
    return state, result


# ── Public runner ─────────────────────────────────────────────────────────────

def run_ga_interval(
    state:           GAState,
    params:          GAParams,
    orders:          List[List[int]],
    sku_location_2d: Dict,
    sku_location_3d: Dict,
    start_gen:       int,
    n_generations:   int,
) -> Tuple[GAState, List[GenerationResult]]:
    """Run the GA for exactly `n_generations` generations.

    Called repeatedly by the experiment runner; between calls the LLM
    controller may update `params` in-place.

    Parameters
    ----------
    state         : Persistent GA state (population, best route, history).
    params        : Mutable hyperparameter container — may be updated by LLM.
    orders        : Order list (fixed for the entire run).
    start_gen     : The generation number at the start of this interval.
    n_generations : How many generations to run in this interval.

    Returns
    -------
    state   : Updated GA state.
    results : List of GenerationResult, one per generation.
    """
    results: List[GenerationResult] = []
    for i in range(n_generations):
        state, result = _run_one_generation(
            state, params, orders, sku_location_2d, sku_location_3d,
            generation_num=start_gen + i,
        )
        results.append(result)
    return state, results


def initialise_ga(
    params:          GAParams,
    sku_location_3d: Dict,
) -> GAState:
    """Create the initial random population and blank state."""
    n_slots    = len(sku_location_3d)
    population = initial_population(params.pop_size, n_slots)
    return GAState(
        population    = population,
        best_route    = None,
        best_distance = float("inf"),
        history       = [],
    )
