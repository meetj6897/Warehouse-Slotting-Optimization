# Detailed LLM-Guided GA Experiment Flow

```mermaid
flowchart TD
    A[User runs: python experiments/run_experiment.py] --> B[parse_args()]
    B --> C[Read CLI values]
    C --> D[settings.py loads .env]
    D --> E[OPENROUTER_API_KEY loaded]
    E --> F[LLM_MODEL = deepseek/deepseek-v4-flash-0731]
    F --> G[Build warehouse geometry: build_location_dicts(policy)]
    G --> H[Generate reorder rates and orders]

    H --> I[GAParams(
    pop_size,
    generations,
    mutation_rate,
    elitism_count,
    crossover_op,
    mutation_op
    )]

    I --> J[initialise_ga(params, sku_location_3d)]
    J --> K[Create initial population]
    K --> L[Create GAState:
    population,
    best_route,
    best_distance,
    history]

    L --> M[Create StateMonitor]
    M --> N[Create ExperienceLog]
    N --> O{no_llm flag?}
    O -- Yes --> P[Disable LLMAgent]
    O -- No --> Q[LLMAgent() initialized with OpenRouter]

    P --> R[current_gen = 0]
    Q --> R

    R --> S{current_gen < args.generations?}

    S -- Yes --> T[interval = min(llm_interval, remaining_generations)]
    T --> U[run_ga_interval(
    state,
    params,
    orders,
    sku_location_2d,
    sku_location_3d,
    start_gen=current_gen,
    n_generations=interval
    )]

    U --> U1[For each generation in interval]
    U1 --> U2[_run_one_generation()]
    U2 --> U3[Evaluate fitness of each chromosome]
    U3 --> U4[Find best individual in this generation]
    U4 --> U5[Update global best_route and best_distance]
    U5 --> U6[elitism_selection(state.population, fitness_values, elitism_count)]
    U6 --> U7[Take elite individuals]
    U7 --> U8[Generate offspring via crossover_op]
    U8 --> U9[Mutate offspring using mutation_rate and mutation_op]
    U9 --> U10[Build GenerationResult with:
    generation
    best_fitness
    avg_fitness
    worst_fitness
    diversity
    mutation_rate
    elitism_count
    pop_size
    crossover_op
    mutation_op
    elapsed_seconds]
    U10 --> U11[Append results to result list]
    U11 --> U12[Return updated GAState + results]

    U12 --> V[For each result in results]
    V --> V1[print_generation_summary()]
    V1 --> V2[best_history.append(best_fitness)]
    V2 --> V3[mutation_history.append(mutation_rate)]
    V3 --> V4[elitism_history.append(elitism_count)]
    V4 --> V5[pop_history.append(pop_size)]

    V5 --> W[current_gen += interval]
    W --> X[monitor.update(results)]
    X --> X1[StateMonitor keeps rolling window]
    X1 --> X2[Compute trend metrics:
    stagnation_count,
    improvement_pct,
    trend,
    phase]
    X2 --> X3[Check if pending interventions have outcome lag satisfied]
    X3 --> X4[ExperienceLog.update_outcome() if ready]

    X4 --> Y{llm_agent is not None and current_gen < total_generations?}
    Y -- No --> S
    Y -- Yes --> Z[opt_state = monitor.compute()]
    Z --> Z1{opt_state is None?}
    Z1 -- Yes --> S
    Z1 -- No --> AA[print_section('LLM Consultation')]
    AA --> AB[logger.info trend, stagnation, diversity]
    AB --> AC[build_user_prompt(opt_state)]
    AC --> AD[LLMAgent.query(opt_state)]
    AD --> AE[Send OpenRouter POST request]
    AE --> AF[Parse JSON response]
    AF --> AG[Extract recommendations:
    mutation_rate,
    pop_size,
    elitism_count,
    crossover_op,
    mutation_op]
    AG --> AH[validate_and_apply(recommendation, params)]
    AH --> AI[Clamp values to allowed bounds]
    AI --> AJ[Reject invalid operators]
    AJ --> AK[Update params in-place if valid]
    AK --> AL[Return ValidationReport]
    AL --> AM[print_llm_decision(...)]
    AM --> AN{recommendation.has_changes()?}
    AN -- Yes --> AO[exp_log.record_intervention(opt_state, recommendation, report)]
    AO --> AP[Pending intervention stored for later outcome evaluation]
    AP --> S
    AN -- No --> AQ[No change; keep current params]
    AQ --> S

    S -- No --> AR[exp_log.flush_pending(current_fitness, current_generation)]
    AR --> AS[Final stats: total interventions, success, neutral, harmful]
    AS --> AT[Print best fitness and route]
    AT --> AU[summary_plot_path = logs/<run_id>_summary.png]
    AU --> AV[plot_run_summary(best_history, mutation_history, elitism_history, pop_history)]
    AV --> AW[Create 3-panel chart:
    1. GA loss curve
    2. mutation rate over generations
    3. population and elitism over generations]
    AW --> AX[Save PNG to logs folder]
    AX --> AY[Experiment complete]

    classDef llm fill:#e3f2fd,stroke:#1e88e5,color:#000;
    classDef ga fill:#e8f5e9,stroke:#43a047,color:#000;
    classDef monitor fill:#fff3e0,stroke:#fb8c00,color:#000;
    classDef output fill:#f3e5f5,stroke:#8e24aa,color:#000;

    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,U1,U2,U3,U4,U5,U6,U7,U8,U9,U10,U11,U12,V,V1,V2,V3,V4,V5,W,X,X1,X2,X3,X4,Y,Z,Z1,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM,AN,AO,AP,AQ,AR,AS,AT,AU,AV,AW,AX,AY ga;
    class AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM,AN,AO,AP,AQ llm;
    class X1,X2,X3,X4,Z,Z1 monitor;
    class AV,AW,AX,AY output;
```

## Interpretation

This is the detailed execution path of the project:

1. Setup and config loading
2. Warehouse and demand generation
3. GA initialization
4. Repeated generation loop
5. Fitness evaluation and state tracking
6. State monitoring for stagnation and diversity
7. LLM decision-making at intervals
8. Safety validation and clamp logic
9. Parameter application
10. Logging and final visualization

This is the version to use when you want to explain the experiment in depth rather than just overview it.
