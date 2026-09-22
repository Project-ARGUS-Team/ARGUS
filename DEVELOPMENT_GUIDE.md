# ARGUS Development Guide

> Project: Adaptive Cognitive Resource Allocation for Large-Scale Agent Simulation  
> Repository: Project-ARGUS-Team/ARGUS  
> Audience: ARGUS developers and AI coding assistants  
> Status: Living development document

This document defines how the ARGUS team should develop, extend, test, and integrate the system.

## 1. Source of Truth

Use this hierarchy:

1. SRS — what the system must do.
2. SDD — how the system is architected.
3. This guide — how the team implements and integrates that architecture.
4. Source code and tests — the current implemented state.

Do not silently replace an SRS/SDD requirement with a more convenient implementation. Architectural changes should be discussed and documented.

## 2. Core Development Principles

### Keep simulation state authoritative

The simulation owns authoritative agent and world state.

Cognitive processing receives a read-only AgentContext and returns a structured StateDelta. Cognitive components must not directly mutate simulation state.

### Keep scheduling separate from simulation

Simulation.tick() should advance the simulation; it should not decide whether an agent receives cognition.

Scheduling belongs in the scheduling/controller layer so that ARGUS can compare:

- uniform full-frequency scheduling;
- simpler LOD/distance baselines;
- ARGUS adaptive scheduling.

### Keep LLM access behind LLMGateway

All external LLM API calls must go through LLMGateway. No simulation, scheduler, telemetry, dashboard, or Unity component should call a provider directly.

### Keep visualization separate

Unity and the dashboard are presentation/experiment interfaces. The simulation research core must work without Unity.

### Prefer measurable, reproducible behavior

Use deterministic seeds and mock providers in tests. Research behavior should be observable through telemetry.

## 3. Current Architecture

High-level flow:

    Simulation Core
          |
          v
    Scheduling Controller
          |
          +---- cognitive update ----> LLMGateway ----> Provider
          |
          +---- prediction ----------> Dead Reckoning
          |
          v
    Authoritative Simulation State
          |
          v
       Telemetry

Major subsystem boundaries:

- argus.simulation — world, agents, state, events, context, state application
- argus.scheduling — scheduling/controller logic
- argus.llm — gateway and provider implementations
- argus.telemetry — experiment/run persistence
- argus.config — configuration and run settings

## 4. Current Implementation State

The foundational simulation/cognitive loop is implemented, including:

- Vector2
- Goal
- ActionType and Action
- AgentState
- WorldEvent
- World
- SimulationState
- deterministic Simulation creation
- simulation ticking and movement
- AgentContext
- StateDelta
- explicit cognitive update requests
- LLMGateway
- deterministic MockLLMProvider
- unit tests for simulation, context, LLM behavior, and the cognitive loop

Important: Simulation.tick() does NOT automatically call the LLM. Cognitive updates are explicitly requested. This is intentional and must be preserved.

Immediate next step: implement the baseline cognitive scheduler/controller.

## 5. Development Stages

### Stage 1 — Baseline Core

Build a deliberately simple uniform full-frequency baseline.

- Every active agent receives a cognitive update every simulation tick.
- Use LLMGateway.
- Add minimal SQLite telemetry.
- Establish measurable baseline resource/cognitive-update behavior.

### Stage 2 — Relevance Scoring

Compute and record five relevance signals:

1. Spatial relevance
2. Interaction probability
3. Goal importance
4. Event participation
5. Social connectivity

At this stage, scoring must not change scheduling. This isolates scoring from adaptive behavior.

### Stage 3 — Adaptive Scheduling and Dead Reckoning

Introduce cognitive budgets, update intervals/frequencies, adaptive scheduling, reduced-frequency processing, and prediction/dead reckoning.

### Stage 4 — Divergence Monitoring

Introduce prediction-vs-authoritative-state comparison, divergence thresholds, emergency/unscheduled cognitive updates, and recovery behavior.

### Stage 5 — Visualization

Add Unity visualization and dashboard/telemetry views. Neither should become a dependency of the simulation core.

### Stage 6 — Evaluation

Compare full-frequency, simpler LOD, and ARGUS strategies on a common simulation-time axis.

Measure as applicable:

- cognitive/LLM update count
- update frequency
- CPU/resource cost
- memory usage
- population capacity
- behavioral fidelity
- goal completion
- prediction divergence
- emergency updates

## 6. Developer Ownership

### Simulation & Scheduling Lead

Owns:

- simulation core;
- agent/world state;
- AgentContext and StateDelta contracts;
- Scheduling Controller;
- baseline scheduler;
- relevance scoring;
- adaptive scheduling;
- dead reckoning;
- divergence monitoring;
- scheduling experiments and research logic.

This role owns the central ARGUS research mechanism.

### LLM & Infrastructure Lead

Owns:

- real LLM providers;
- provider configuration;
- request/response handling;
- timeouts, retries, and provider errors;
- gateway integration;
- telemetry repository/schema;
- experiment data export.

Build against the existing LLMGateway, AgentContext, and StateDelta contracts. Do not redesign the simulation around a particular provider.

### Visualization & Evaluation Lead

When the third developer is available, primarily owns:

- UnityBridge;
- Unity visualization;
- dashboard/UI;
- evaluation tooling;
- result visualization;
- baseline comparison presentation.

Ownership may change, but coordinate before modifying another developer's subsystem.

## 7. Parallel Development

Parallel work is encouraged when subsystem contracts already exist.

Good parallel work:

- one developer implements scheduling while another implements a provider;
- one developer implements telemetry against agreed data contracts;
- visualization work uses mocks/stubs rather than waiting for the simulation to be finished.

Coordinate before:

- changing a shared interface;
- changing authoritative state ownership;
- changing the SRS/SDD architecture;
- introducing incompatible representations of shared data.

Rule of thumb: share interfaces, isolate implementations.

## 8. Git Strategy

Do not use main for active development.

Use focused feature branches, for example:

    main
     |
     +-- feature/simulation-core
     +-- feature/llm-provider
     +-- feature/telemetry
     +-- feature/unity
     +-- feature/evaluation

Prefer small, focused commits.

Good examples:

- Add baseline cognitive scheduler
- Add SQLite telemetry repository
- Add real LLM provider
- Add relevance scoring
- Add divergence monitor

Avoid vague commits such as "stuff", "changes", or "final".

## 9. Pull Requests

Before opening a PR:

1. Run the complete test suite.
2. Run Ruff.
3. Review the diff.
4. Ensure only related changes are included.
5. Explain architectural changes.

A PR should state:

- what changed;
- why it changed;
- which SRS/SDD requirement it supports;
- tests added/changed;
- whether an interface changed;
- known limitations.

Never delete or weaken existing tests merely to make a new implementation pass.

## 10. Testing

Every new behavior requires tests.

Use:

- unit tests for classes/functions;
- integration tests for subsystem boundaries;
- regression tests for bug fixes.

Tests must not require a live external LLM. Use MockLLMProvider or another deterministic test double.

Important scheduler tests should verify measurable behavior, such as:

- baseline updates = active agents × ticks;
- inactive agents are not unnecessarily updated;
- cognitive output changes state only through StateDelta;
- adaptive scheduling reduces updates under controlled low-relevance conditions;
- emergency updates occur when divergence exceeds the configured threshold.

## 11. LLM Provider Rules

LLM providers are implementations behind LLMGateway.

Provider-specific code should handle:

- authentication/configuration;
- request construction;
- response parsing;
- timeouts;
- retries where appropriate;
- provider errors;
- conversion into StateDelta.

The simulation must not depend on a particular vendor.

Provider failure must not force provider-specific behavior into the simulation architecture.

## 12. State and Interface Rules

AgentContext is a read-only cognitive view, not authoritative agent state.

StateDelta is the structured result of cognitive processing. Do not turn it into an unrestricted mechanism for arbitrary state mutation.

ARGUS studies when and how often cognition is executed. Do not add free-form reasoning/chain-of-thought fields merely to make the demonstration appear more sophisticated.

LLMGateway is the only external-LLM boundary.

## 13. Scheduling Rules

Scheduling is part of the research contribution.

Do not put adaptive scheduling logic directly inside:

- Simulation.tick();
- AgentState;
- an LLM provider;
- Unity code.

The scheduler should eventually determine whether an agent receives:

- a full cognitive update;
- a reduced-frequency update;
- prediction/dead reckoning;
- an emergency cognitive update.

Scheduling decisions must eventually be observable through telemetry.

## 14. Telemetry Rules

Telemetry should make experiments auditable.

Record information such as:

- simulation run;
- tick/time;
- agent;
- relevance score/signals;
- assigned cognitive interval;
- cognitive update events;
- prediction usage;
- divergence;
- emergency/unscheduled updates.

SQLite is the intended authoritative telemetry store for the current single-machine research implementation.

Telemetry is not the authoritative live simulation state.

## 15. AI-Assisted Development

Any AI coding assistant working on ARGUS should:

1. Read this guide before architectural changes.
2. Inspect existing code before creating files.
3. Preserve the existing project structure.
4. Avoid recreating existing components.
5. Follow the SRS and SDD.
6. Avoid silently changing public interfaces.
7. Add tests for new behavior.
8. Run tests after changes.
9. Run Ruff after changes.
10. Never bypass LLMGateway for external LLM calls.
11. Never put adaptive scheduling directly into the simulation tick loop.
12. Never delete existing tests just to make an implementation pass.
13. Prefer small, reviewable changes over large rewrites.

When uncertain about architecture, ask the project developers rather than inventing a new architecture.

## 16. Avoid Premature Complexity

Do not introduce these unless explicitly required:

- complex physics;
- sophisticated pathfinding;
- elaborate personality systems;
- unrestricted memory architectures;
- distributed multi-machine simulation;
- unnecessary microservices;
- a custom LLM;
- a learned scheduler before the rule-based research baseline is validated;
- Unity dependencies in the simulation core;
- dashboard dependencies in the simulation core.

The first objective is a small, measurable research system.

## 17. Definition of Done

A feature is ready when:

- it is consistent with the architecture;
- relevant tests exist;
- existing tests still pass;
- Ruff passes;
- necessary interfaces are documented;
- telemetry is added when experimental behavior changes;
- the implementation can be explained in terms of the research objective.

## 18. Immediate Work Queue

### Simulation & Scheduling

1. Baseline cognitive scheduler/controller.
2. Tests proving uniform full-frequency behavior.
3. Minimal telemetry integration.
4. Stage 2 relevance scoring.
5. Adaptive scheduling.
6. Dead reckoning.
7. Divergence monitoring.

### LLM & Infrastructure

1. Real provider behind LLMGateway.
2. Provider configuration through project configuration/environment variables.
3. Robust timeout/error handling.
4. Telemetry persistence/query support.
5. Experiment data export.

### Later

1. UnityBridge.
2. Dashboard/telemetry views.
3. Baseline comparison tooling.
4. Experiment result visualization.

## 19. Final Rule

When choosing between implementations, prefer the one that makes ARGUS:

- easier to measure;
- easier to reproduce;
- easier to compare against baselines;
- easier to test;
- easier to replace components in;
- clearer as a research contribution.

The goal is not to build the largest simulation. The goal is to build a controlled, measurable demonstration that adaptive cognitive allocation can reduce computational work while maintaining acceptable agent behavior.
