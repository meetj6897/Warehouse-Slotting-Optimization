"""
memory/experience_log.py
========================
Append-only JSONL log that records every LLM intervention and its outcome.

Each entry captures:
  - The optimisation state at the time of the recommendation
  - The recommendation itself (raw + validated)
  - The outcome: fitness delta measured N generations later

This log serves two purposes:
  1. Offline analysis of which interventions worked.
  2. Future retrieval-augmented prompting — pass successful past interventions
     back to the LLM as few-shot examples ("in situation X, action Y produced
     a Z% improvement").

File format: one JSON object per line (JSONL), UTF-8.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import EXPERIENCE_LOG_PATH
from llm.agent import LLMRecommendation
from llm.safety_validator import ValidationReport
from utils.state_monitor import OptimisationState

logger = logging.getLogger(__name__)


@dataclass
class InterventionRecord:
    """One complete intervention event: state → recommendation → outcome."""
    timestamp:          str
    run_id:             str
    generation:         int
    # State at intervention time
    state_snapshot:     Dict[str, Any]
    # LLM output
    phase_assessment:   str
    root_cause:         str
    reasoning:          str
    expected_outcome:   str
    confidence:         float
    # Validated changes applied
    changes_accepted:   List[str]
    changes_clamped:    List[str]
    changes_rejected:   List[str]
    # Outcome fields — filled in later by update_outcome()
    fitness_before:     float = 0.0
    fitness_after:      Optional[float] = None
    fitness_delta:      Optional[float] = None   # negative = improvement
    outcome_label:      Optional[str]  = None    # "success" | "neutral" | "harmful"
    outcome_generation: Optional[int]  = None


class ExperienceLog:
    """Manages the JSONL experience log file."""

    def __init__(self, path: str = EXPERIENCE_LOG_PATH, run_id: Optional[str] = None):
        self.path   = Path(path)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._pending: List[InterventionRecord] = []   # awaiting outcome

        # Create parent directory if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ExperienceLog initialised → %s  (run_id=%s)", self.path, self.run_id)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_intervention(
        self,
        opt_state:      OptimisationState,
        recommendation: LLMRecommendation,
        report:         ValidationReport,
    ) -> InterventionRecord:
        """Create an InterventionRecord and add it to the pending queue.

        The record is NOT written to disk yet — call update_outcome() after
        measuring the post-intervention fitness.
        """
        record = InterventionRecord(
            timestamp        = datetime.now(timezone.utc).isoformat(),
            run_id           = self.run_id,
            generation       = opt_state.generation,
            state_snapshot   = {
                "best_fitness":    opt_state.best_fitness,
                "avg_fitness":     opt_state.avg_fitness,
                "diversity":       opt_state.diversity,
                "mutation_rate":   opt_state.mutation_rate,
                "elitism_count":   opt_state.elitism_count,
                "crossover_op":    opt_state.crossover_op,
                "mutation_op":     opt_state.mutation_op,
                "population_size": opt_state.population_size,
                "stagnation":      opt_state.stagnation_count,
                "trend":           opt_state.trend,
                "phase":           opt_state.phase,
            },
            phase_assessment = recommendation.phase_assessment,
            root_cause       = recommendation.root_cause,
            reasoning        = recommendation.reasoning,
            expected_outcome = recommendation.expected_outcome,
            confidence       = recommendation.confidence,
            changes_accepted = report.accepted,
            changes_clamped  = report.clamped,
            changes_rejected = report.rejected,
            fitness_before   = opt_state.best_fitness,
        )
        self._pending.append(record)
        logger.debug(
            "InterventionRecord created | gen=%d | changes=%d",
            record.generation,
            len(report.accepted) + len(report.clamped),
        )
        return record

    def update_outcome(
        self,
        record:           InterventionRecord,
        fitness_after:    float,
        outcome_generation: int,
        improvement_threshold: float = 0.005,   # 0.5% improvement counts as success
    ) -> None:
        """Fill in the outcome fields and flush the record to disk.

        Parameters
        ----------
        record                : The record returned by record_intervention().
        fitness_after         : Best fitness measured N generations after intervention.
        outcome_generation    : The generation at which fitness_after was measured.
        improvement_threshold : Fraction improvement required to label as "success".
        """
        record.fitness_after      = fitness_after
        record.fitness_delta      = fitness_after - record.fitness_before
        record.outcome_generation = outcome_generation

        # Label the outcome
        if record.fitness_before > 0:
            pct = -record.fitness_delta / record.fitness_before
        else:
            pct = 0.0

        if pct >= improvement_threshold:
            record.outcome_label = "success"
        elif pct <= -improvement_threshold:
            record.outcome_label = "harmful"
        else:
            record.outcome_label = "neutral"

        self._write_record(record)
        if record in self._pending:
            self._pending.remove(record)

        logger.info(
            "Outcome logged | gen=%d→%d | Δfitness=%.1f | label=%s",
            record.generation,
            outcome_generation,
            record.fitness_delta,
            record.outcome_label,
        )

    def flush_pending(self, current_fitness: float, current_generation: int) -> None:
        """Write all pending records that haven't been given outcomes yet.

        Called at the end of the run to ensure nothing is lost.
        """
        for record in list(self._pending):
            self.update_outcome(record, current_fitness, current_generation)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def load_successful_interventions(
        self,
        min_confidence: float = 0.6,
        max_records:    int   = 5,
    ) -> List[Dict[str, Any]]:
        """Return the most recent successful interventions from the log.

        Used to build few-shot examples for the LLM prompt.

        Parameters
        ----------
        min_confidence : Only return records where the LLM had high confidence.
        max_records    : Maximum number of records to return.
        """
        if not self.path.exists():
            return []

        successes: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    entry.get("outcome_label") == "success"
                    and entry.get("confidence", 0) >= min_confidence
                ):
                    successes.append(entry)

        # Return the most recent ones
        return successes[-max_records:]

    def summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics across all logged interventions."""
        if not self.path.exists():
            return {"total": 0}

        records: List[Dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            return {"total": 0}

        outcomes = [r.get("outcome_label") for r in records if r.get("outcome_label")]
        deltas   = [r["fitness_delta"] for r in records if r.get("fitness_delta") is not None]

        return {
            "total":        len(records),
            "success":      outcomes.count("success"),
            "neutral":      outcomes.count("neutral"),
            "harmful":      outcomes.count("harmful"),
            "avg_delta":    round(sum(deltas) / len(deltas), 1) if deltas else None,
            "best_delta":   round(min(deltas), 1) if deltas else None,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_record(self, record: InterventionRecord) -> None:
        """Append one record to the JSONL file."""
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), default=str) + "\n")
        except OSError as e:
            logger.error("Failed to write experience log: %s", e)
