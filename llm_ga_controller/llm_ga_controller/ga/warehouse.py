"""
ga/warehouse.py
===============
Warehouse geometry: generates all 3-D shelf positions (x, y, z) and their
2-D projections (x, y) used by routing algorithms.

Two storage policies are supported:
  W   — random; all slots treated equally (baseline)
  GZA — Gold-Zone Assignment; high-demand slots (z=1,2) listed first so the
        GA naturally places fast-moving SKUs at ergonomic levels.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config.settings import (
    AISLE_SPACING,
    COLUMN_SPACING,
    NUM_COLUMNS,
    NUM_LEVELS,
    NUM_SHELVES,
)


# ── 3-D location generators ──────────────────────────────────────────────────

def generate_3d_locations(
    num_shelves: int = NUM_SHELVES,
    num_levels: int = NUM_LEVELS,
    num_columns: int = NUM_COLUMNS,
    aisle_spacing: float = AISLE_SPACING,
    column_spacing: float = COLUMN_SPACING,
) -> List[Tuple[float, float, int]]:
    """Return all (x, y, z) shelf positions for a regular rack layout.

    Rack rows are placed at x = aisle_spacing, 3·aisle_spacing, 5·aisle_spacing …
    so consecutive rows are separated by a full aisle width.
    """
    locations: List[Tuple[float, float, int]] = []
    for shelf_idx in range(1, num_shelves * 2, 2):
        x = shelf_idx * aisle_spacing
        for col in range(num_columns):
            y = col * column_spacing
            for level in range(num_levels):
                locations.append((x, y, level))
    return locations


def generate_2d_locations(
    num_shelves: int = NUM_SHELVES,
    num_columns: int = NUM_COLUMNS,
    aisle_spacing: float = AISLE_SPACING,
    column_spacing: float = COLUMN_SPACING,
) -> List[Tuple[float, float]]:
    """Return (x, y) positions for distance-matrix computation (ignores z)."""
    locations: List[Tuple[float, float]] = []
    for shelf_idx in range(1, num_shelves * 2, 2):
        x = shelf_idx * aisle_spacing
        for col in range(num_columns):
            y = col * column_spacing
            locations.append((x, y))
    return locations


# ── GZA sub-generators ───────────────────────────────────────────────────────

def _gold_zone_locations(
    num_shelves: int = NUM_SHELVES,
    num_columns: int = NUM_COLUMNS,
    aisle_spacing: float = AISLE_SPACING,
    column_spacing: float = COLUMN_SPACING,
) -> List[Tuple[float, float, int]]:
    """Ergonomic mid-height slots (z = 1 and z = 2)."""
    locs: List[Tuple[float, float, int]] = []
    for shelf_idx in range(1, num_shelves * 2, 2):
        x = shelf_idx * aisle_spacing
        for col in range(num_columns):
            y = col * column_spacing
            for z in (1, 2):
                locs.append((x, y, z))
    return locs


def _non_gold_zone_locations(
    num_shelves: int = NUM_SHELVES,
    num_levels: int = NUM_LEVELS,
    num_columns: int = NUM_COLUMNS,
    aisle_spacing: float = AISLE_SPACING,
    column_spacing: float = COLUMN_SPACING,
) -> List[Tuple[float, float, int]]:
    """Non-ergonomic slots (z = 0 ground, z = 3 top)."""
    locs: List[Tuple[float, float, int]] = []
    for shelf_idx in range(1, num_shelves * 2, 2):
        x = shelf_idx * aisle_spacing
        for col in range(num_columns):
            y = col * column_spacing
            for z in range(0, num_levels, 3):   # selects 0 and 3
                locs.append((x, y, z))
    return locs


# ── Public location dictionaries ─────────────────────────────────────────────

def build_location_dicts(
    policy: str = "GZA",
) -> Tuple[Dict[int, Tuple[float, float, int]], Dict[int, Tuple[float, float]]]:
    """Build the slot-index → coordinate lookup tables.

    Parameters
    ----------
    policy : "GZA" (gold-zone first) or "W" (random / uniform).

    Returns
    -------
    sku_location_3d : slot index → (x, y, z)   used for retrieval-time calc.
    sku_location_2d : slot index → (x, y)       used for routing distance calc.
    """
    if policy == "GZA":
        locs_3d = _gold_zone_locations() + _non_gold_zone_locations()
    else:
        # W policy — plain enumeration, no ergonomic bias
        locs_3d = generate_3d_locations()

    sku_location_3d: Dict[int, Tuple[float, float, int]] = {
        i: loc for i, loc in enumerate(locs_3d)
    }
    # 2-D mapping covers ALL 500 slots (one per shelf-column-level triple),
    # simply stripping the z coordinate.  This ensures every slot index
    # produced by the GA chromosome maps to a valid (x, y) position.
    sku_location_2d: Dict[int, Tuple[float, float]] = {
        i: (xyz[0], xyz[1]) for i, xyz in sku_location_3d.items()
    }
    return sku_location_3d, sku_location_2d
