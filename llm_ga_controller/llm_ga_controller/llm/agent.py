"""
llm/agent.py
============
Calls a local Ollama model and parses the JSON recommendation returned
by the LLM Meta-Controller.

Model used: deepseek-r1:8b
  - Chain-of-thought reasoning model — thinks before answering
  - Reliable structured JSON output
  - Runs locally — completely free, no API key needed
  - Fallback: llama3.1:8b  (set LLM_FALLBACK_MODEL in config/settings.py)

Ollama must be running before starting the experiment:
    ollama serve          # in a separate terminal
    ollama pull deepseek-r1:8b

The <think>...</think> block that deepseek-r1 produces internally is
stripped before JSON extraction so it never pollutes the parsed output.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import ollama

from config.settings import (
    LLM_FALLBACK_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_HOST,
)
from llm.prompt_builder import build_user_prompt, get_system_prompt
from utils.state_monitor import OptimisationState

logger = logging.getLogger(__name__)

MAX_RETRIES  = 3
RETRY_DELAY  = 2.0   # seconds between retries


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMRecommendation:
    """Parsed (but not yet safety-validated) recommendation from the LLM."""
    phase_assessment:  str
    root_cause:        str
    mutation_rate:     Optional[float]
    pop_size:          Optional[int]
    elitism_count:     Optional[int]
    crossover_op:      Optional[str]
    mutation_op:       Optional[str]
    reasoning:         str
    expected_outcome:  str
    confidence:        float

    # Preserved for logging / auditing
    raw_response:      str = field(default="", repr=False)
    model_used:        str = field(default="", repr=False)
    prompt_tokens:     int = 0
    completion_tokens: int = 0

    def has_changes(self) -> bool:
        """True if at least one parameter change was recommended."""
        return any([
            self.mutation_rate  is not None,
            self.pop_size       is not None,
            self.elitism_count  is not None,
            self.crossover_op   is not None,
            self.mutation_op    is not None,
        ])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase_assessment": self.phase_assessment,
            "root_cause":       self.root_cause,
            "recommendations": {
                "mutation_rate":  self.mutation_rate,
                "pop_size":       self.pop_size,
                "elitism_count":  self.elitism_count,
                "crossover_op":   self.crossover_op,
                "mutation_op":    self.mutation_op,
            },
            "reasoning":        self.reasoning,
            "expected_outcome": self.expected_outcome,
            "confidence":       self.confidence,
            "model_used":       self.model_used,
        }


# ── JSON extractor ────────────────────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """Remove deepseek-r1 chain-of-thought <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object found in `text`.

    Handles:
      - deepseek-r1 <think> blocks
      - Markdown code fences  ```json ... ```
      - Leading / trailing prose around the JSON object
    """
    # 1. Strip chain-of-thought reasoning block (deepseek-r1)
    cleaned = _strip_think_tags(text)

    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    cleaned = cleaned.replace("```", "").strip()

    # 3. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Find the outermost { ... } block
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No valid JSON object found in LLM response.\n"
        f"First 600 chars:\n{text[:600]}"
    )


# ── Model availability check ──────────────────────────────────────────────────

def _resolve_model(preferred: str, fallback: str) -> str:
    """Return the first model that is available locally; warn if falling back."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        available = {m["name"] for m in client.list()["models"]}
        # Ollama stores names like "deepseek-r1:8b" or "deepseek-r1:8b-..."
        for model in (preferred, fallback):
            if any(model in name for name in available):
                return model
        logger.warning(
            "Neither '%s' nor '%s' found locally. "
            "Run: ollama pull %s",
            preferred, fallback, preferred,
        )
        return preferred   # let the call fail with a clear error from Ollama
    except Exception as e:
        logger.warning("Could not list Ollama models (%s). Using '%s'.", e, preferred)
        return preferred


# ── Ollama LLM Agent ──────────────────────────────────────────────────────────

class LLMAgent:
    """Sends prompts to a local Ollama model and parses structured responses."""

    def __init__(self):
        self.client        = ollama.Client(host=OLLAMA_HOST)
        self.model         = _resolve_model(LLM_MODEL, LLM_FALLBACK_MODEL)
        self.system_prompt = get_system_prompt()
        logger.info("LLMAgent initialised | model=%s | host=%s", self.model, OLLAMA_HOST)

    def query(self, opt_state: OptimisationState) -> LLMRecommendation:
        """Query the local LLM with the current optimisation state.

        Retries up to MAX_RETRIES times on connection errors or bad JSON.
        Returns a no-change recommendation if all retries are exhausted.
        """
        user_prompt = build_user_prompt(opt_state)
        logger.debug("User prompt (%d chars):\n%s", len(user_prompt), user_prompt)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug("Ollama query attempt %d/%d | model=%s",
                             attempt, MAX_RETRIES, self.model)

                response = self.client.chat(
                    model   = self.model,
                    options = {
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS,
                    },
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )

                raw_text = response["message"]["content"]
                logger.debug("Raw response (%d chars):\n%s", len(raw_text), raw_text[:800])

                parsed = _extract_json(raw_text)
                rec    = parsed.get("recommendations", {})

                recommendation = LLMRecommendation(
                    phase_assessment  = parsed.get("phase_assessment", "unknown"),
                    root_cause        = parsed.get("root_cause", ""),
                    mutation_rate     = rec.get("mutation_rate"),
                    pop_size          = rec.get("pop_size"),
                    elitism_count     = rec.get("elitism_count"),
                    crossover_op      = rec.get("crossover_op"),
                    mutation_op       = rec.get("mutation_op"),
                    reasoning         = parsed.get("reasoning", ""),
                    expected_outcome  = parsed.get("expected_outcome", ""),
                    confidence        = float(parsed.get("confidence", 0.5)),
                    raw_response      = raw_text,
                    model_used        = self.model,
                    prompt_tokens     = response.get("prompt_eval_count", 0),
                    completion_tokens = response.get("eval_count", 0),
                )

                logger.info(
                    "LLM decision | phase=%-25s | confidence=%.0f%% | changes=%s | "
                    "tokens: %d in / %d out",
                    recommendation.phase_assessment,
                    recommendation.confidence * 100,
                    recommendation.has_changes(),
                    recommendation.prompt_tokens,
                    recommendation.completion_tokens,
                )
                return recommendation

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("JSON parse error on attempt %d: %s", attempt, e)
            except ollama.ResponseError as e:
                logger.warning("Ollama response error on attempt %d: %s", attempt, e)
                if "model" in str(e).lower() and "not found" in str(e).lower():
                    logger.error(
                        "Model '%s' is not pulled. Run:  ollama pull %s",
                        self.model, self.model,
                    )
                    break   # no point retrying a missing model
            except Exception as e:
                logger.warning("Unexpected error on attempt %d: %s", attempt, e)

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

        # All retries exhausted — keep current params
        logger.error("LLM query failed after %d attempts. Keeping current params.", MAX_RETRIES)
        return LLMRecommendation(
            phase_assessment = "unknown",
            root_cause       = "LLM query failed — keeping current parameters.",
            mutation_rate    = None,
            pop_size         = None,
            elitism_count    = None,
            crossover_op     = None,
            mutation_op      = None,
            reasoning        = "All LLM query attempts failed.",
            expected_outcome = "No change.",
            confidence       = 0.0,
            raw_response     = "",
            model_used       = self.model,
        )
