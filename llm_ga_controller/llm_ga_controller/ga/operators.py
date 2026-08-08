"""
ga/operators.py
===============
All genetic operators used by the warehouse GA.

Crossover operators  (return a valid permutation child):
  - order_crossover (OX)   — default; preserves relative order
  - pmx_crossover   (PMX)  — partially-mapped crossover
  - cycle_crossover (CX)   — cycle-based crossover

Mutation operators  (modify a chromosome in-place, return it):
  - swap_mutation      — swap two random genes   (fast, mild)
  - inversion_mutation — reverse a random segment (moderate disruption)
  - scramble_mutation  — randomly shuffle a segment (strongest disruption)

The GA runner selects operators by name from the active GAParams object,
so the LLM Meta-Controller can switch operators at runtime simply by
changing 'crossover_op' or 'mutation_op' in GAParams.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np


# ── Crossover operators ───────────────────────────────────────────────────────

def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Order Crossover (OX) — preserves relative order from both parents.

    1. Copy a random contiguous segment from parent1.
    2. Fill remaining positions with genes from parent2 (in order, skipping dupes).
    """
    n = len(parent1)
    start = np.random.randint(0, n)
    end   = np.random.randint(start + 1, n + 1)

    child = [-1] * n
    child[start:end] = parent1[start:end]

    in_child = set(child[start:end])
    remaining = [g for g in parent2 if g not in in_child]

    fill_idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = remaining[fill_idx]
            fill_idx += 1
    return child


def pmx_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Partially-Mapped Crossover (PMX).

    Creates a mapping between the two parent segments, then resolves conflicts
    via the mapping until all positions are valid.
    """
    n = len(parent1)
    start = np.random.randint(0, n)
    end   = np.random.randint(start + 1, n + 1)

    child = [-1] * n
    child[start:end] = parent1[start:end]

    # Build position map: value → index in child
    pos_map = {v: i for i, v in enumerate(child) if v != -1}

    for i in range(start, end):
        val = parent2[i]
        if val not in pos_map:
            # Find a free slot following the PMX mapping chain
            pos = i
            while child[pos] != -1:
                pos = parent2.index(parent1[pos])
            child[pos] = val
            pos_map[val] = pos

    # Fill any still-unfilled positions from parent2
    remaining = [g for g in parent2 if g not in pos_map]
    for i in range(n):
        if child[i] == -1:
            child[i] = remaining.pop(0)
    return child


def cycle_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Cycle Crossover (CX).

    Identifies cycles between parent1 and parent2; alternates which parent
    contributes each cycle.
    """
    n = len(parent1)
    child   = [-1] * n
    visited = [False] * n
    cycle_num = 0

    for start in range(n):
        if visited[start]:
            continue
        # Trace the cycle starting at 'start'
        cycle = []
        idx = start
        while not visited[idx]:
            visited[idx] = True
            cycle.append(idx)
            val = parent2[idx]
            idx = parent1.index(val)

        # Even cycles come from parent1; odd cycles from parent2
        src = parent1 if cycle_num % 2 == 0 else parent2
        for i in cycle:
            child[i] = src[i]
        cycle_num += 1

    return child


# ── Mutation operators ────────────────────────────────────────────────────────

def swap_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """Swap two randomly chosen genes with probability `mutation_rate`."""
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(individual), size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


def inversion_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """Reverse a random contiguous segment with probability `mutation_rate`.

    More disruptive than swap — useful when the GA stagnates.
    """
    if np.random.rand() < mutation_rate:
        n = len(individual)
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        individual[i:j+1] = individual[i:j+1][::-1]
    return individual


def scramble_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """Randomly shuffle a random segment with probability `mutation_rate`.

    Strongest disruption — use sparingly (e.g. on severe stagnation).
    """
    if np.random.rand() < mutation_rate:
        n = len(individual)
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        segment = individual[i:j+1]
        random.shuffle(segment)
        individual[i:j+1] = segment
    return individual


# ── Operator registries ───────────────────────────────────────────────────────
# The GA runner looks up operators by the string names stored in GAParams.
# Add new operators here to make them available to the LLM meta-controller.

CROSSOVER_OPS = {
    "OX":  order_crossover,
    "PMX": pmx_crossover,
    "CX":  cycle_crossover,
}

MUTATION_OPS = {
    "swap":      swap_mutation,
    "inversion": inversion_mutation,
    "scramble":  scramble_mutation,
}
