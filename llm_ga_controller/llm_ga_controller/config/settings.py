"""
config/settings.py
==================
Central configuration for the LLM-Guided GA Meta-Controller.

All GA defaults, hyperparameter bounds, LLM invocation settings,
and warehouse physical parameters live here. Edit this file to
change the behaviour of the entire system — nothing else needs touching.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple


# Load local secret values from the project root .env file.
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


# ── OpenRouter (active LLM API) ─────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY_gpt4o", os.getenv("OPENROUTER_API_KEY", ""))
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL:           str = "openai/gpt-4o"
LLM_FALLBACK_MODEL:  str = "openai/gpt-4o-mini"
LLM_MAX_TOKENS:      int = 1024
LLM_TEMPERATURE:     float = 0.2  # low → deterministic, consistent JSON output

# ---------------------------------------------------------------------------
# Legacy Ollama configuration kept here as commented reference only.
# ---------------------------------------------------------------------------
# OLLAMA_HOST:         str   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# LLM_MODEL:           str   = "deepseek-r1:8b"
# LLM_FALLBACK_MODEL:  str   = "llama3.1:8b"
# LLM_MAX_TOKENS:      int   = 1024
# LLM_TEMPERATURE:     float = 0.2

# ---------------------------------------------------------------------------
# Google Cloud configuration kept here as commented reference only.
# ---------------------------------------------------------------------------
# GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "project-fad3f647-9796-449e-a2e")
# GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
# GCP_MODEL: str = os.getenv("GCP_MODEL", "gemini-2.5-pro")


# ── Warehouse Physical Parameters ────────────────────────────────────────────

# Original warehouse-scale problem from the notebook.
# 5 racks × 25 columns × 4 levels = 500 slot positions.
NUM_SHELVES:    int   = 5
NUM_LEVELS:     int   = 4
NUM_COLUMNS:    int   = 25
TOTAL_SLOT_CAPACITY: int = NUM_SHELVES * NUM_LEVELS * NUM_COLUMNS
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
    pop_size:       int   = 100
    generations:    int   = 1500
    mutation_rate:  float = 0.70
    elitism_count:  int   = 10      # how many top chromosomes survive unchanged
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
