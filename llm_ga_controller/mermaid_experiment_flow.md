# LLM-Guided GA Experiment Flow

```mermaid
flowchart TD
    A[Start run_experiment.py] --> B[Parse CLI args]
    B --> C[Load settings and .env]
    C --> D[Configure OpenRouter / LLM model]
    D --> E[Build warehouse geometry]
    E --> F[Generate reorder rates and demand orders]
    F --> G[Initialize GAParams]
    G --> H[pop_size = CLI value
    generations = CLI value
    mutation_rate = CLI value
    elitism_count = CLI value
    crossover_op = OX
    mutation_op = swap]

    H --> I[Initialise GA state]
    I --> J[Set monitor, experience log, and LLMAgent]

    J --> K{LLM enabled?}
    K -- No --> Z1[Run plain GA baseline]
    K -- Yes --> L[Current generation = 0]

    L --> M{current_gen < total_generations?}
    M -- Yes --> N[Run GA interval for LLM_INTERVAL generations]
    N --> O[Evaluate population fitness]
    O --> P[Selection + crossover + mutation]
    P --> Q[Compute generation metrics:
    best_fitness
    avg_fitness
    diversity
    mutation_rate
    elitism_count
    population_size]
    Q --> R[Append to history arrays]

    R --> S[StateMonitor updates rolling window]
    S --> T{Enough history for LLM prompt?}
    T -- No --> U[current_gen += interval]
    U --> M

    T -- Yes --> V[Build prompt from
    best fitness trend
    diversity
    stagnation
    improvement %
    recent history]
    V --> W[Call OpenRouter / DeepSeek model]
    W --> X[Parse JSON recommendation]
    X --> Y[SafetyValidator clamps values]
    Y --> AA{Recommendation changed?}
    AA -- Yes --> AB[Apply new GA params in-place]
    AB --> AC[Record intervention in experience log]
    AC --> U
    AA -- No --> AD[Keep current params]
    AD --> AC

    Z1 --> AE[Finish GA without LLM]
    U --> M
    M -- No --> AF[Flush remaining intervention outcomes]
    AF --> AG[Final summary report]
    AG --> AH[Best fitness + log path]
    AH --> AI[Generate summary plot:
    GA loss curve
    mutation rate
    population / elitism over generations]

    AI --> AJ[Experiment complete]
```

## Short explanation

This project runs a Genetic Algorithm, monitors the state each interval, asks an LLM to recommend changes when needed, validates those changes, and then keeps evolving until the configured generation count is reached.
