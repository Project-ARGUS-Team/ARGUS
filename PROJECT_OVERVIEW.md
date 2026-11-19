# ARGUS — Project Overview

> **Project:** Adaptive Cognitive Resource Allocation for Large-Scale Agent Simulation  
> **Repository:** `Project-ARGUS-Team/ARGUS`  
> **Document status:** Living project overview  
> **Last reviewed:** 2026-11-19  
> **Audience:** Project team, supervisors, reviewers, and future contributors

This document explains what ARGUS is, what the current implementation can do, what the research is intended to achieve, and how the project is expected to evolve. It should be updated whenever a feature meaningfully changes the system's capabilities, architecture, or research status.

**Status convention**
- **Implemented** — present in the repository; this does not automatically mean fully validated for research use.
- **In progress** — partially implemented or being integrated.
- **Planned** — part of the target design, but not yet implemented.
- **Exploratory** — a possible direction that still requires a design or feasibility decision.

---

## 1. Project in brief

ARGUS is a research-oriented framework for simulating large populations of autonomous, LLM-driven agents while controlling the computational cost of their cognition.

In a conventional LLM-driven simulation, agents may be asked to reason at every simulation tick, even when little has changed around them. This can produce a large number of expensive model calls as the population and simulation duration increase.

ARGUS investigates whether an **adaptive cognitive resource-allocation system** can decide how much reasoning each agent needs, and how frequently that reasoning should run, based on the agent's current relevance to the evolving world.

The central research question is:

> Can per-agent adaptive cognitive scheduling reduce LLM usage and computational cost in large-scale agent simulations while maintaining acceptable behavioral fidelity?

ARGUS is not intended to be merely a town simulator or a wrapper around an LLM API. The simulation is the controlled environment; the primary research contribution is the mechanism that allocates cognitive resources across agents.

## 2. What ARGUS is intended to do

The target system models a populated environment containing agents with individual state, routines, goals, activities, social relationships, memories, and interactions.

At each simulation step, the system should determine which agents need expensive cognitive processing, which can safely use a cheaper or less frequent update, and which can continue using a prediction of their likely state until a full update is required.

The intended adaptive loop is:

1. Observe each agent's current context and relevant world conditions.
2. Estimate how important it is for that agent to receive cognitive processing now.
3. Convert the estimate into a cognitive budget or update schedule.
4. Run full cognition for agents that need it.
5. Use reduced-frequency processing or prediction for agents that do not.
6. Monitor predicted behavior against the authoritative simulation state.
7. Trigger an unscheduled full update when an agent's behavior diverges beyond an acceptable threshold.
8. Record decisions and outcomes so different scheduling strategies can be compared.

The goal is not to minimize model calls at any cost. It is to find a useful balance between computational efficiency and believable, sufficiently consistent agent behavior.

## 3. The research mechanism

### 3.1 Relevance estimation

The current design evaluates five signals for each agent:

| Signal | What it represents |
|---|---|
| Spatial relevance | How relevant nearby world activity or events are to the agent |
| Interaction probability | How likely the agent is to interact with nearby agents |
| Goal importance | The importance of the agent's current goal |
| Event participation | Whether the agent is involved in active world events |
| Social connectivity | The agent's social connections in its current local context |

The Stage 2 implementation contains a deterministic rule-based scorer. It normalizes these signals and combines them using configurable non-negative weights. Its current default weights are equal.

**Important:** relevance scoring is currently descriptive. It can be calculated and recorded, but it does not yet control the agent's cognitive update frequency.

### 3.2 Adaptive cognitive scheduling — planned

The intended scheduler will translate relevance into an agent-level cognitive budget and update interval. The design distinguishes between:

- **Full cognitive update:** execute the agent's normal cognitive processing.
- **Reduced-frequency update:** update less often when the agent is less relevant.
- **Prediction / dead reckoning:** advance an agent using a cheaper prediction between full cognitive updates.
- **Emergency update:** promote an agent to full cognition when a trigger or divergence condition requires it.

The exact budget thresholds, interval mapping, hysteresis, and promotion rules must remain explicit configuration or documented research parameters. They should not be treated as validated until experiments support them.

### 3.3 Divergence monitoring — planned

Prediction is only useful when its errors remain controlled. ARGUS is intended to compare predicted agent state with the authoritative simulation state, identify excessive divergence, and trigger corrective full cognition when needed.

The project should measure both the efficiency gained from prediction and the behavioral cost of using it.

### 3.4 Experimental comparison

The research should compare strategies on the same scenarios and simulation-time basis:

1. **Full-frequency baseline:** every active agent receives a cognitive update every tick.
2. **Simple LOD baseline:** a simpler scheduling strategy, such as distance- or local-relevance-based update levels.
3. **ARGUS adaptive strategy:** the proposed relevance-guided cognitive allocation and recovery mechanism.

Relevant measurements include:
- LLM/cognitive update count and frequency;
- latency and computational/resource usage;
- memory usage;
- supported population and simulation throughput;
- behavioral fidelity and state divergence;
- goal completion and interaction outcomes;
- prediction usage and emergency updates.

The exact metrics and experimental protocol should be specified before formal evaluation.

## 4. What the current implementation can do

The repository currently contains a working foundation and two separate demonstration scenarios. The presence of code is not, by itself, evidence that a feature has been validated at scale.

### 4.1 Simulation foundation — Implemented

The simulation layer currently provides:
- deterministic simulation creation using a seed;
- agents with positions, velocities, goals, actions, and plans;
- agent profiles, occupations, daily routines, and activities;
- world dimensions, movement, roads, and world events;
- per-agent context construction;
- structured cognitive state updates;
- social connections and relationship strengths;
- agent memories and a timeline of recorded events/thoughts;
- interaction-encounter tracking.

Simulation state remains authoritative in the simulation layer. Cognitive processing is requested explicitly; advancing the simulation clock does not itself automatically call an LLM.

### 4.2 Cognitive update and baseline scheduling — Implemented

The uniform full-frequency `BaselineScheduler` updates every active agent on each scheduler step through the configured LLM gateway. It can also invoke optional relevance scoring.

This is the control/baseline behavior needed before introducing adaptive scheduling. It provides a reference against which later approaches can be measured.

### 4.3 Relevance scoring — Implemented, Stage 2

`RuleBasedRelevanceScorer` calculates the five relevance signals and a weighted aggregate score. The score is currently observational: the baseline scheduler does not use it to skip or defer cognitive updates.

### 4.4 LLM-driven agent behavior — Implemented in the current demos

The repository includes an LLM gateway/provider abstraction and a local Ollama-backed conversation provider. In the interaction demonstration, agents can generate short, structured conversation summaries when an interaction begins.

The conversation path includes:
- a local Ollama request using `qwen2.5-coder:7b-instruct`;
- JSON response parsing and duplicate-summary checking;
- recent conversation memories supplied as context;
- conversation memories written for both participants;
- a shared timeline entry;
- a deterministic conversation fallback when provider generation fails;
- terminal diagnostics for request, response, acceptance, and fallback behavior.

The conversation provider currently uses a **90-second request timeout**. This is the configuration that was tested successfully in the current village demonstration and should not be changed casually.

The project also contains memory-management, retrieval, and daily reflection components. Their behavior and reliability should be evaluated separately before being described as fully validated long-term cognition.

### 4.5 Telemetry — Implemented foundation

A SQLite telemetry repository currently supports simulation runs, participating agents, cognitive-update events, and relevance-score records.

The telemetry foundation is intended to grow to cover adaptive scheduling decisions, prediction use, divergence, recovery events, and evaluation outputs as those mechanisms are implemented.

### 4.6 Visual demonstration — Implemented

A Tkinter viewer provides:
- a map of the simulated environment;
- agent movement and interaction indicators;
- an agent inspector;
- agent activity, goals, position, and social information;
- a scrollable timeline;
- simulation controls for play, pause, step, reset, and speed.

The viewer is a development and demonstration tool, not the final research evaluation interface.

## 5. Demonstration environments

The repository intentionally keeps two launchers separate. They serve different development purposes and should not be conflated.

| Environment | Launcher | Population | Main purpose |
|---|---|---:|---|
| Village interaction sandbox | `examples/run_viewer.py` | 2 agents | Debug LLM conversations, memories, social interactions, and timeline behavior |
| Town baseline | `examples/run_city.py` | 30 agents | Observe the existing multi-agent town, routines, movement, and baseline behavior |

### Village interaction sandbox

The small village scenario focuses on two connected residents, Arun and Maya. It is deliberately small so interaction and memory behavior can be inspected without a large number of concurrent LLM requests.

Run it with:

```bash
python examples/run_viewer.py
```

### 30-agent town baseline

The city scenario is a separate deterministic Smallville-style environment with a 180 × 120 world, named landmarks, connected road segments, homes, varied occupations, daily routines, and social connections. Its baseline setup is pedestrian-only, and random world events are currently disabled in this scenario.

Run it with:

```bash
python examples/run_city.py
```

The two launchers use separate scenario constructors. Keep the village launcher and scenario stable while developing or experimenting with the city.

## 6. High-level architecture

The intended system is divided into components with clear responsibilities:

| Component | Responsibility |
|---|---|
| Simulation Core | Owns authoritative world and agent state, time progression, movement, context construction, and state-delta application |
| Scheduling Controller | Decides when agents receive full cognition, reduced updates, prediction, or emergency updates |
| Relevance Scorer | Calculates the agent relevance signals and aggregate score |
| Dead-Reckoning Predictor | Predicts lower-cost agent state between full cognitive updates |
| Divergence Monitor | Detects when predicted behavior deviates too far from authoritative or expected state |
| LLMGateway | Provides the controlled boundary for external LLM calls |
| Telemetry Repository | Persists run, update, relevance, and later scheduling/evaluation measurements |
| Visualization / Unity Bridge | Displays the simulation and experiment state without owning simulation logic |
| Run Configuration / Comparison | Configures repeatable runs and compares strategies under common conditions |

The architecture should preserve these boundaries. In particular:
- authoritative simulation state belongs to the simulation;
- scheduling decisions belong to the scheduling layer, not `Simulation.tick()`;
- external LLM access should go through the gateway boundary;
- visualization must not become a dependency of the research core;
- telemetry records experiments but does not own live simulation state.

## 7. Planned development roadmap

The roadmap follows the project's staged development approach. Update each stage as implementation and validation progress.

| Stage | Focus | Intended outcome | Current status |
|---|---|---|---|
| 1 | Baseline core | Reproducible simulation, uniform full-frequency updates, LLM gateway, foundational telemetry | Implemented foundation; continue validation |
| 2 | Relevance scoring | Calculate and record five relevance signals without changing scheduling | Implemented |
| 3 | Adaptive scheduling and dead reckoning | Cognitive budgets, update intervals, reduced-frequency cognition, prediction | Planned |
| 4 | Divergence monitoring | Prediction comparison, thresholds, emergency updates, recovery behavior | Planned |
| 5 | Visualization and integration | Dashboard/telemetry views and Unity integration | Tkinter development viewer exists; dashboard and Unity integration planned |
| 6 | Evaluation | Compare full-frequency, simple LOD, and ARGUS strategies | Planned |

A stage should only be marked complete when its behavior is implemented, tested, and sufficiently validated for the claim being made.

## 8. Current boundaries and known limitations

The following points are important when interpreting the current project:

- The current baseline scheduler still performs full-frequency cognitive updates for every active agent. Relevance scores do not yet reduce LLM usage.
- Adaptive cognitive budgets, scheduling intervals, dead reckoning, and divergence-triggered emergency updates are not yet implemented.
- The 30-agent city is a baseline demonstration, not evidence that large-scale LLM-driven populations are already supported efficiently.
- The two-agent village is the current focused environment for LLM conversation and memory behavior. These features have not yet been integrated into a validated 30-agent adaptive experiment.
- The current viewer is a Tkinter development visualization. The planned dashboard and Unity integration are separate future work.
- Current telemetry is a foundation; it does not yet capture every metric required for the final research comparison.
- Local LLM latency and hardware constraints affect demonstration speed. Performance claims must be based on recorded experiments rather than anecdotal observations.

## 9. Research goals and success criteria

ARGUS should ultimately demonstrate, through repeatable experiments, that its adaptive strategy can:

1. Reduce the number of expensive cognitive/LLM updates compared with full-frequency scheduling.
2. Improve computational efficiency or increase the population that can be simulated under a fixed resource budget.
3. Maintain acceptable agent behavior, measured using explicitly defined fidelity and task/goal metrics.
4. Detect when cheaper predictions become unreliable and recover through timely full cognitive updates.
5. Produce reproducible, auditable results across multiple seeds and comparable scenarios.

No numerical success threshold is declared in this overview. The team should define thresholds in the evaluation plan before final experiments rather than selecting them after observing results.

## 10. How to keep this document updated

Treat this file as the high-level explanation of the project, not a replacement for detailed specifications.

Update it when:
- a roadmap stage changes status;
- a new subsystem or capability is implemented;
- a demonstration launcher or scenario changes;
- a major architectural decision is made;
- an important limitation is resolved or discovered;
- evaluation metrics or success criteria are agreed.

When updating:
1. Preserve the distinction between implemented, in-progress, planned, and exploratory work.
2. Do not describe a feature as complete solely because a class or prototype exists.
3. Keep detailed interfaces, requirements, design decisions, and test instructions in their dedicated documents.
4. Update the date in the header.
5. Include meaningful limitations and validation status, not just feature announcements.

## 11. Related project documents

- `README.md` — repository entry point and setup instructions.
- `DEVELOPMENT_GUIDE.md` — engineering rules, subsystem ownership, implementation stages, testing expectations, and contribution workflow.
- **SRS** — authoritative system requirements specification.
- **SDD** — authoritative software design description.

The SRS defines what ARGUS must do; the SDD defines how the system is designed. This overview explains the project and summarizes the current state. Keep the three documents consistent when the design or requirements change.
