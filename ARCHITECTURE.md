# ARCHITECTURE.md: The Orchestra Framework

## 1. Philosophy: Scheduling, Not Stepping

Orchestra is built on a strict decoupling of:

- **Orchestration**: task compilation, dependency tracking, scheduling, persistence
- **Labor**: agent execution on bounded inputs

The framework does **not** iterate through a workflow with an instruction pointer. Instead, it compiles the workflow into an initial set of tasks and then lets a dependency-aware scheduler expand and execute the graph dynamically.

## 2. Core Model

### Workflow Spec
A workflow is a declarative description of intended work.

### Task Graph
The framework manages concrete tasks:
- read tasks
- agent tasks
- write tasks
- map item tasks
- pairwise reduction tasks
- seed, alias, and collect tasks for dynamic expansion

### Scheduler
The scheduler owns execution:
- tracks task states
- determines which tasks are ready
- executes ready tasks
- records outputs
- unlocks dependents
- rewires downstream dependencies when dynamic patterns materialize into concrete finalization tasks

## 3. Infrastructure: Workspace vs. Stores

### Workspace
- Raw filesystem
- Used by tools and agents for side-effectful labor
- Not managed as a communication surface

### Stores
- Managed document repositories
- Formal communication boundary between tasks
- Sidecar metadata persists lineage and timestamps

## 4. Agents

Agents remain stateless workers.

They:
- consume documents
- use tools
- produce one or more documents

They do **not** know:
- where they are in the workflow
- what comes before or after
- whether they were triggered by a map, reduce, or ordinary task

## 5. Pattern Expansion

Patterns are dynamic expansion primitives, not runtime control loops.

### Map
A map node is materialized into:
- one seed task per input document
- one concrete agent task per document
- one final collect task

### Tree-Reduce
A tree-reduce node is materialized into:
- one seed task per input document
- a balanced hierarchy of pairwise merge tasks
- one final alias task

### Read/Write
Read and write compile directly into concrete persistence tasks.

## 6. Dependency Semantics

A task may begin only when:
- all upstream dependencies completed successfully
- all required inputs are present

Dynamic pattern nodes initially appear as planner tasks. Once they materialize concrete work, downstream tasks are rewired to depend on the pattern's finalization task rather than the planner task itself.

This allows:
- fan-out
- fan-in
- dynamic task insertion
- partial recomputation
- future parallel execution

## 7. Temporal Grounding

Temporal grounding remains tool-based.

Agents may call `get_context` to discover the current project reality rather than relying on pretraining-era assumptions.

## 8. Execution Flow

1. Load workflow spec
2. Compile spec into initial tasks
3. Seed scheduler with tasks that have no unmet dependencies
4. When a materializer task becomes ready, expand it into concrete tasks
5. Rewire pre-existing downstream dependencies to the materialized finalization task
6. Execute ready concrete tasks
7. Persist outputs and mark tasks complete
8. Unlock downstream tasks
9. Continue until all reachable tasks finish
