"""
config/settings.py
==================
Central configuration for the LLM-Guided GA Meta-Controller.

All GA defaults, hyperparameter bounds, LLM invocation settings,
and warehouse physical parameters live here. Edit this file to
change the behaviour of the entire system — nothing else needs touching.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ── Ollama (local LLM — free, no API key needed) ─────────────────────────────
#
# Model choice: deepseek-r1:8b
#   - Best reasoning quality at 8B parameters — uses chain-of-thought internally
#   - Follows complex structured JSON instructions reliably
#   - Runs on a laptop GPU (4-6 GB VRAM) or CPU (slower but works)
#   - Ideal for the meta-controller which needs to reason about trends and
#     produce valid JSON recommendations consistently
#
# Fallback: llama3.1:8b  (faster on CPU-only, slightly weaker reasoning)
#
# Setup (one-time):
#   1. Install Ollama  →  https://ollama.com/download
#   2. Pull the model  →  ollama pull deepseek-r1:8b
#   3. Run the server  →  ollama serve          (stays running in background)
#   4. Run the GA      →  python experiments/run_experiment.py

OLLAMA_HOST:         str   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL:           str   = "deepseek-r1:8b"   # primary
LLM_FALLBACK_MODEL:  str   = "llama3.1:8b"       # fallback
LLM_MAX_TOKENS:      int   = 1024
LLM_TEMPERATURE:     float = 0.2   # low → deterministic, consistent JSON output


# ── Warehouse Physical Parameters ────────────────────────────────────────────

NUM_SHELVES:    int   = 5
NUM_LEVELS:     int   = 4
NUM_COLUMNS:    int   = 25
AISLE_LENGTH:   float = 25.0        # metres
AISLE_SPACING:  float = 10.0        # centre-to-centre between rack rows (m)
COLUMN_SPACING: float = 10.0        # distance between shelf columns (m)

LEVEL_TIME: Dict[int, float] = {
    0: 2.0 * 60,    # Ground level  — requires bending; slowest
    1: 0.5 * 60,    # Lower-mid     — golden zone; fastest
    2: 0.5 * 60,    # Upper-mid     — golden zone; fastest
    3: 2.5 * 60,    # Top level     — requires stool; slowest
}


# ── GA Default Hyperparameters ────────────────────────────────────────────────

@dataclass
class GAParams:
    """Mutable container for all GA hyperparameters.

    The LLM Meta-Controller modifies an instance of this class each time
    it makes a recommendation. Passing the instance into the GA runner
    means every generation automatically uses the current (possibly updated)
    values.
    """
    pop_size:       int   = 200
    generations:    int   = 200
    mutation_rate:  float = 0.50
    elitism_count:  int   = 20      # how many top chromosomes survive unchanged
    crossover_op:   str   = "OX"    # "OX" | "PMX" | "CX"
    mutation_op:    str   = "swap"  # "swap" | "inversion" | "scramble"

    def as_dict(self) -> dict:
        return {
            "pop_size":      self.pop_size,
            "mutation_rate": self.mutation_rate,
            "elitism_count": self.elitism_count,
            "crossover_op":  self.crossover_op,
            "mutation_op":   self.mutation_op,
        }


# ── LLM Meta-Controller Settings ─────────────────────────────────────────────

LLM_INTERVAL:       int = 10    # consult LLM every N generations
LLM_HISTORY_WINDOW: int = 10    # past generations included in prompt

# Safety bounds — the validator will CLAMP any recommendation outside these.
PARAM_BOUNDS: Dict[str, Tuple] = {
    "mutation_rate":  (0.01, 0.90),
    "pop_size":       (50,   500),
    "elitism_count":  (2,    40),
}

ALLOWED_CROSSOVER_OPS: list = ["OX", "PMX", "CX"]
ALLOWED_MUTATION_OPS:  list = ["swap", "inversion", "scramble"]


# ── Demand Simulation Defaults ────────────────────────────────────────────────

N_SKUS_FAST:    int = 100
N_SKUS_MEDIUM:  int = 200
N_SKUS_SLOW:    int = 200
N_ORDERS:       int = 1_000
MAX_ORDER_SIZE: int = 50
DEMAND_SEED:    int = 30


# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR:              str = "logs"
EXPERIENCE_LOG_PATH:  str = "logs/experience_log.jsonl"
