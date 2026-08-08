"""
ga/population.py
================
Population initialisation, selection, and demand-simulation helpers.

Demand simulation (reorder rates + order generation) lives here because it
feeds directly into chromosome sizing — the number of SKUs determines the
permutation length.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from config.settings import (
    DEMAND_SEED,
    MAX_ORDER_SIZE,
    N_ORDERS,
    N_SKUS_FAST,
    N_SKUS_MEDIUM,
    N_SKUS_SLOW,
)


# ── Demand simulation ─────────────────────────────────────────────────────────

def generate_reorder_rates(
    n_fast:   int = N_SKUS_FAST,
    n_medium: int = N_SKUS_MEDIUM,
    n_slow:   int = N_SKUS_SLOW,
    seed:     int = DEMAND_SEED,
) -> List[float]:
    """Generate a sorted (descending) list of SKU reorder rates.

    Tri-modal distribution:
      Fast   SKUs — reorder rate ∈ [6.0, 10.0]
      Medium SKUs — reorder rate ∈ [3.0,  4.5]
      Slow   SKUs — reorder rate ∈ [0.1,  2.5]
    """
    random.seed(seed)
    fast   = [round(random.uniform(6.0, 10.0), 2) for _ in range(n_fast)]
    medium = [round(random.uniform(3.0,  4.5), 2) for _ in range(n_medium)]
    slow   = [round(random.uniform(0.1,  2.5), 2) for _ in range(n_slow)]
    combined = fast + medium + slow
    combined.sort(reverse=True)   # SKU 0 = fastest mover
    return combined


def generate_orders(
    sku_ids:         List[int],
    sku_reorder_rate: Dict[int, float],
    n_orders:        int = N_ORDERS,
    max_order_size:  int = MAX_ORDER_SIZE,
    seed:            int = DEMAND_SEED,
) -> List[List[int]]:
    """Simulate customer orders with demand-weighted SKU sampling.

    Each order is a unique set of SKU IDs (no repeat picks per order).
    """
    random.seed(seed)
    weights = [sku_reorder_rate[s] for s in sku_ids]
    orders: List[List[int]] = []
    for _ in range(n_orders):
        size    = random.choice(range(2, max_order_size + 1))
        sampled = random.choices(sku_ids, weights=weights, k=size)
        orders.append(list(set(sampled)))
    return orders


# ── Population initialisation ─────────────────────────────────────────────────

def initial_population(pop_size: int, n_slots: int) -> List[List[int]]:
    """Create `pop_size` random permutations of all slot indices.

    chromosome[i] = the SKU stored at slot i.
    """
    return [random.sample(range(n_slots), n_slots) for _ in range(pop_size)]


# ── Selection ─────────────────────────────────────────────────────────────────

def elitism_selection(
    population:    List[List[int]],
    fitness_values: List[float],
    elitism_count: int,
) -> Tuple[List[List[int]], List[int]]:
    """Select the top-k chromosomes by fitness (lower = better).

    Returns
    -------
    elite          : The k best chromosomes (copied, not referenced).
    sorted_indices : Indices of the full population sorted by fitness ascending.
    """
    sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
    elite = [population[i][:] for i in sorted_indices[:elitism_count]]
    return elite, sorted_indices
