# LLM-Guided Adaptive Meta-Controller for Warehouse GA

Genetic Algorithm for warehouse order-picking optimisation, controlled by a
**local LLM (deepseek-r1:8b via Ollama) — completely free, no API key needed.**

## Project Structure

```
llm_ga_controller/
│
├── config/
│   └── settings.py             # All hyperparameter bounds, GA defaults, Ollama config
│
├── ga/
│   ├── warehouse.py            # Warehouse geometry, location generation (W & GZA)
│   ├── operators.py            # OX / PMX / CX crossover + swap / inversion / scramble mutation
│   ├── fitness.py              # Fitness evaluation, S-shape routing
│   ├── population.py           # Population init, elitism selection, demand simulation
│   └── runner.py               # Interval-based GA loop — accepts dynamic params
│
├── llm/
│   ├── prompt_builder.py       # Builds the structured prompt from GA state
│   ├── agent.py                # Calls Ollama API, parses JSON response
│   └── safety_validator.py     # Clamps every recommendation to allowed bounds
│
├── memory/
│   └── experience_log.py       # Records (state → action → outcome) per run (JSONL)
│
├── utils/
│   ├── state_monitor.py        # Computes diversity, stagnation, improvement %
│   └── logger.py               # Console + file logging helpers
│
├── experiments/
│   ├── run_experiment.py       # Main entry point
│   └── compare_baseline.py     # Plain GA vs LLM-GA side-by-side comparison
│
├── logs/                       # Auto-created at runtime
├── requirements.txt
└── README.md
```

## Why deepseek-r1:8b?

| Model | Reasoning | JSON reliability | VRAM | Speed (CPU) |
|---|---|---|---|---|
| **deepseek-r1:8b** ✓ | Chain-of-thought | ★★★★★ | ~5 GB | Moderate |
| llama3.1:8b | Standard | ★★★★☆ | ~5 GB | Fast |
| mistral:7b | Standard | ★★★☆☆ | ~4 GB | Fast |
| phi3:mini | Limited | ★★☆☆☆ | ~2 GB | Very fast |

`deepseek-r1:8b` uses internal chain-of-thought (`<think>` blocks) before
producing its answer, which makes it significantly better at reasoning about
optimisation trends and producing valid structured JSON — exactly what the
meta-controller needs.

## Quick Start

### 1. Install Ollama
```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows → download installer from https://ollama.com/download
```

### 2. Pull the model & start the server
```bash
ollama pull deepseek-r1:8b   # ~5 GB download, one-time only
ollama serve                  # keep this running in a separate terminal
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the experiment
```bash
# LLM-guided GA (default)
python experiments/run_experiment.py

# Custom settings
python experiments/run_experiment.py \
    --generations 300 \
    --pop-size 200 \
    --llm-interval 10 \
    --policy GZA \
    --seed 30

# Plain GA baseline (no LLM — for comparison)
python experiments/run_experiment.py --no-llm

# Side-by-side comparison plot
python experiments/compare_baseline.py --generations 200
```

## How It Works

```
Every LLM_INTERVAL generations (default: 10):

  GA runs N generations
        │
        ▼
  StateMonitor computes:
    - trend (improving / stagnating / oscillating)
    - stagnation count
    - population diversity
    - improvement % over window
        │
        ▼
  PromptBuilder assembles structured prompt
  (state table + history window + JSON schema)
        │
        ▼
  deepseek-r1:8b (Ollama) reasons and returns JSON:
    {
      "phase_assessment": "stagnation",
      "root_cause": "diversity collapsed, mutation too low",
      "recommendations": {
        "mutation_rate": 0.35,
        "crossover_op": "PMX",
        ...
      },
      "confidence": 0.82
    }
        │
        ▼
  SafetyValidator clamps values to allowed bounds
        │
        ▼
  GA continues with updated parameters
        │
        ▼
  ExperienceLog records intervention + outcome
```

## Configuration

Edit `config/settings.py` to change:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `deepseek-r1:8b` | Primary Ollama model |
| `LLM_FALLBACK_MODEL` | `llama3.1:8b` | Used if primary not pulled |
| `LLM_INTERVAL` | `10` | Consult LLM every N generations |
| `LLM_HISTORY_WINDOW` | `10` | Past generations in prompt |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic |
| `DEFAULT_POP_SIZE` | `200` | GA population size |
| `DEFAULT_MUTATION_RATE` | `0.5` | Initial mutation rate |

## Output Files

| File | Description |
|---|---|
| `logs/run_<timestamp>.log` | Full debug log of every generation |
| `logs/experience_log.jsonl` | Every LLM intervention + outcome label |
| `logs/comparison.png` | Convergence plot (compare_baseline.py) |
