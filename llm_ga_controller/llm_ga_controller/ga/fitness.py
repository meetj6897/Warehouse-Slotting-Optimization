"""
ga/fitness.py
=============
Fitness evaluation for the warehouse SKU-assignment GA.

Fitness  =  total S-shape travel distance  +  total vertical retrieval time
          (minimisation problem — lower is better)

This module is intentionally stateless: the warehouse location dicts and
order list are passed in as arguments so the same functions work for any
configuration (W-policy, GZA-policy, different order batches, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config.settings import AISLE_LENGTH, LEVEL_TIME


# ── Coordinate resolver ───────────────────────────────────────────────────────

def resolve_order_coordinates(
    chromosome: List[int],
    order: List[int],
    sku_location_2d: Dict[int, Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Map a single order's SKU IDs to (x, y) coordinates.

    chromosome[slot] = SKU stored at that slot, so the inverse
    (SKU → slot) is found with list.index().

    Returns unique (x, y) positions in insertion order.
    """
    coords: List[Tuple[float, float]] = []
    seen: set = set()
    for sku in order:
        slot = chromosome.index(sku)
        xy   = sku_location_2d[slot][:2]
        if xy not in seen:
            coords.append(xy)
            seen.add(xy)
    return coords


# ── Retrieval time ────────────────────────────────────────────────────────────

def compute_retrieval_time(
    chromosome: List[int],
    order: List[int],
    sku_location_3d: Dict[int, Tuple[float, float, int]],
) -> float:
    """Sum the ergonomic time penalties for every SKU in one order."""
    total = 0.0
    for sku in order:
        slot    = chromosome.index(sku)
        z_level = sku_location_3d[slot][2]
        total  += LEVEL_TIME.get(z_level, 0.0)
    return total


# ── S-shape routing ───────────────────────────────────────────────────────────

def sort_s_shape(
    coordinates: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Sort pick coordinates into S-shape (snake) traversal order."""
    coordinates.sort(key=lambda c: c[0])
    unique_x = sorted(set(c[0] for c in coordinates))

    # Group by aisle
    groups: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    prev_x = None
    for coord in coordinates:
        if coord[0] != prev_x:
            if current:
                groups.append(current)
            current = []
            prev_x  = coord[0]
        current.append(coord)
    if current:
        groups.append(current)

    # Alternate direction per aisle
    for group in groups:
        aisle_idx = unique_x.index(group[0][0])
        reverse   = (aisle_idx % 2 != 0)
        group.sort(key=lambda c: c[1], reverse=reverse)

    return [c for group in groups for c in group]


def s_shape_distance(sorted_order: List[Tuple[float, float]]) -> float:
    """Compute total travel distance for one order under S-shape routing."""
    if not sorted_order:
        return 0.0

    unique_aisles = sorted(set(c[0] for c in sorted_order))
    n_aisles      = len(unique_aisles)

    horizontal_dist = sum(
        abs(sorted_order[i + 1][0] - sorted_order[i][0])
        for i in range(len(sorted_order) - 1)
    )
    dist_to_first = sorted_order[0][0] + sorted_order[0][1]

    if n_aisles % 2 == 0:
        vertical_dist    = n_aisles * AISLE_LENGTH
        dist_from_last   = sorted_order[-1][0]
    else:
        vertical_dist    = (n_aisles - 1) * AISLE_LENGTH + sorted_order[-1][1]
        dist_from_last   = sorted_order[-1][0] + sorted_order[-1][1]

    return dist_to_first + vertical_dist + horizontal_dist + dist_from_last


# ── Population evaluator ──────────────────────────────────────────────────────

def evaluate_population(
    population: List[List[int]],
    orders: List[List[int]],
    sku_location_2d: Dict[int, Tuple[float, float]],
    sku_location_3d: Dict[int, Tuple[float, float, int]],
) -> List[float]:
    """Compute the fitness score for every chromosome in the population.

    Returns
    -------
    fitness_values : one float per chromosome (lower = better).
    """
    fitness_values: List[float] = []

    for chromosome in population:
        dist_sum = 0.0
        time_sum = 0.0

        for order in orders:
            # Distance component
            coords        = resolve_order_coordinates(chromosome, order, sku_location_2d)
            sorted_coords = sort_s_shape(coords)
            dist_sum     += s_shape_distance(sorted_coords)

            # Vertical retrieval time component
            time_sum += compute_retrieval_time(chromosome, order, sku_location_3d)

        fitness_values.append(dist_sum + time_sum)

    return fitness_values
