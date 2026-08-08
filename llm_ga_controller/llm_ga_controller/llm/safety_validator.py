"""
llm/safety_validator.py
=======================
Validates and clamps every LLM recommendation before it is applied to the GA.

Rules
-----
  - Numerical params are clamped to the bounds defined in config/settings.py.
  - Operator names are checked against the allowed lists; unknown names are
    silently ignored (no change applied).
  - If a recommended value is identical to the current value, the field is
    set to None (no-op) to avoid unnecessary log noise.
  - A ValidationReport is returned alongside the (possibly clamped) params
    so the caller can log exactly what changed and why.

The validator never raises — worst case it keeps current params unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import (
    ALLOWED_CROSSOVER_OPS,
    ALLOWED_MUTATION_OPS,
    GAParams,
    PARAM_BOUNDS,
)
from llm.agent import LLMRecommendation

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Records every decision made during validation."""
    accepted:  List[str] = field(default_factory=list)   # param: old → new
    clamped:   List[str] = field(default_factory=list)   # param: raw → clamped
    rejected:  List[str] = field(default_factory=list)   # param: reason
    no_change: List[str] = field(default_factory=list)   # param: already at recommended value

    def summary(self) -> str:
        lines = []
        if self.accepted:
            lines.append("  Accepted  : " + " | ".join(self.accepted))
        if self.clamped:
            lines.append("  Clamped   : " + " | ".join(self.clamped))
        if self.rejected:
            lines.append("  Rejected  : " + " | ".join(self.rejected))
        if self.no_change:
            lines.append("  No-change : " + " | ".join(self.no_change))
        return "\n".join(lines) if lines else "  No recommendations to apply."


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def validate_and_apply(
    recommendation: LLMRecommendation,
    params: GAParams,
) -> ValidationReport:
    """Apply the LLM recommendation to `params` in-place, clamping as needed.

    Parameters
    ----------
    recommendation : Raw (unchecked) output from LLMAgent.query().
    params         : Current GA hyperparameters — modified in-place.

    Returns
    -------
    ValidationReport describing every decision made.
    """
    report = ValidationReport()

    # ── mutation_rate ────────────────────────────────────────────────────────
    if recommendation.mutation_rate is not None:
        raw = float(recommendation.mutation_rate)
        lo, hi = PARAM_BOUNDS["mutation_rate"]
        clamped = _clamp(raw, lo, hi)
        if abs(clamped - params.mutation_rate) < 1e-6:
            report.no_change.append(f"mutation_rate={clamped:.3f}")
        else:
            if abs(clamped - raw) > 1e-6:
                report.clamped.append(
                    f"mutation_rate: {raw:.3f} → {clamped:.3f} (bounds {lo}–{hi})"
                )
            else:
                report.accepted.append(
                    f"mutation_rate: {params.mutation_rate:.3f} → {clamped:.3f}"
                )
            params.mutation_rate = clamped

    # ── pop_size ─────────────────────────────────────────────────────────────
    if recommendation.pop_size is not None:
        try:
            raw = int(recommendation.pop_size)
        except (TypeError, ValueError):
            report.rejected.append(f"pop_size: non-integer value '{recommendation.pop_size}'")
            raw = None

        if raw is not None:
            lo, hi = PARAM_BOUNDS["pop_size"]
            clamped = int(_clamp(raw, lo, hi))
            if clamped == params.pop_size:
                report.no_change.append(f"pop_size={clamped}")
            else:
                if clamped != raw:
                    report.clamped.append(
                        f"pop_size: {raw} → {clamped} (bounds {lo}–{hi})"
                    )
                else:
                    report.accepted.append(
                        f"pop_size: {params.pop_size} → {clamped}"
                    )
                params.pop_size = clamped

    # ── elitism_count ────────────────────────────────────────────────────────
    if recommendation.elitism_count is not None:
        try:
            raw = int(recommendation.elitism_count)
        except (TypeError, ValueError):
            report.rejected.append(
                f"elitism_count: non-integer value '{recommendation.elitism_count}'"
            )
            raw = None

        if raw is not None:
            lo, hi = PARAM_BOUNDS["elitism_count"]
            clamped = int(_clamp(raw, lo, hi))
            # Also ensure elitism_count never exceeds pop_size
            clamped = min(clamped, params.pop_size)
            if clamped == params.elitism_count:
                report.no_change.append(f"elitism_count={clamped}")
            else:
                if clamped != raw:
                    report.clamped.append(
                        f"elitism_count: {raw} → {clamped}"
                    )
                else:
                    report.accepted.append(
                        f"elitism_count: {params.elitism_count} → {clamped}"
                    )
                params.elitism_count = clamped

    # ── crossover_op ─────────────────────────────────────────────────────────
    if recommendation.crossover_op is not None:
        val = str(recommendation.crossover_op).strip().upper()
        if val not in ALLOWED_CROSSOVER_OPS:
            report.rejected.append(
                f"crossover_op: '{val}' not in {ALLOWED_CROSSOVER_OPS}"
            )
        elif val == params.crossover_op:
            report.no_change.append(f"crossover_op={val}")
        else:
            report.accepted.append(
                f"crossover_op: {params.crossover_op} → {val}"
            )
            params.crossover_op = val

    # ── mutation_op ──────────────────────────────────────────────────────────
    if recommendation.mutation_op is not None:
        val = str(recommendation.mutation_op).strip().lower()
        if val not in ALLOWED_MUTATION_OPS:
            report.rejected.append(
                f"mutation_op: '{val}' not in {ALLOWED_MUTATION_OPS}"
            )
        elif val == params.mutation_op:
            report.no_change.append(f"mutation_op={val}")
        else:
            report.accepted.append(
                f"mutation_op: {params.mutation_op} → {val}"
            )
            params.mutation_op = val

    logger.info(
        "SafetyValidator | accepted=%d clamped=%d rejected=%d no_change=%d",
        len(report.accepted),
        len(report.clamped),
        len(report.rejected),
        len(report.no_change),
    )
    return report
