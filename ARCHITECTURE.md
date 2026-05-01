# ARCHITECTURE.md: The Orchestra Framework

## 1. Purpose

Orchestra is a workflow engine for document-oriented reasoning systems.

It is designed to coordinate:

- persistent document collections
- bounded agent work
- dependency-aware scheduling
- explicit transformation stages
- pairwise synthesis

The framework is meant to remain agnostic to any particular workflow domain. Research, code analysis, planning, evaluation, summarization, and other workflows should all be expressible using the same core execution model.

---

## 2. Separation of Concerns

Orchestra separates **orchestration** from **labor**.

### Orchestration
The framework is responsible for:

- admitting workflow nodes
- tracking dependencies
- materializing patterns into concrete tasks
- routing documents between stores
- running non-agent infrastructure steps
- enforcing execution invariants

### Labor
Agents are responsible for:

- bounded reasoning over provided inputs
- using approved tools when helpful
- producing one raw-text output document per task

Agents do not own scheduling, persistence policy, or graph mutation semantics directly.

---

## 3. Core Data Model

### 3.1 Document
The fundamental unit of work is the **Document**.

A document consists of:

- raw text content
- metadata managed by the framework

The content channel is plain text. The framework must not depend on agent output being encoded in JSON, Markdown envelopes, XML, or any other structured text protocol.

### 3.2 Store
A **Store** is a persistent collection of documents.

Stores are first-class. They are not special-case implementation details. A workflow may use many stores with different roles, such as:

- source stores
- intermediate stores
- analysis stores
- report stores
- control stores
- scratch stores

No fixed store taxonomy is required by the framework.

### 3.3 Collection Semantics
The framework treats stores and task outputs as collections.

A collection may contain:
- zero documents
- one document
- many documents

Singleton inputs are not special. Patterns should operate uniformly over collections regardless of size.

---

## 4. Execution Model

Orchestra executes a dependency graph of tasks.

A workflow is authored in terms of high-level nodes. Those nodes may compile directly into executable tasks or may materialize into internal task subgraphs.

The scheduler is responsible for:

- determining task readiness
- executing ready tasks
- updating dependency counts
- managing materialized subgraphs
- ensuring that downstream tasks do not run until their true dependencies are satisfied

The scheduler remains the authority over execution order.

---

## 5. Patterns

Patterns are the primary abstraction for expressing work over collections.

A pattern defines:
- how inputs are enumerated
- how many concrete tasks are created
- whether agent work is applied
- how outputs are persisted or collected

Patterns should be generic and workflow-independent.

### 5.1 Map
`map` applies bounded work independently to each document in a collection.

Semantics:
- enumerate input documents
- create one task per document
- collect resulting outputs

### 5.2 Transform
`transform` is the semantic form of local document processing.

Typical uses:
- extraction
- rewriting
- classification
- filtering
- normalization
- structured interpretation

A transform step operates on one bounded input unit at a time.

### 5.3 Chunk Map
`chunk_map` is a framework-controlled expansion pattern.

Semantics:
- enumerate input documents
- split them into bounded chunks
- run one local task per chunk
- collect or persist chunk outputs

Chunking is part of orchestration because it changes graph structure and boundedness.

### 5.4 Reduce
`reduce` is the synthesis primitive.

Semantics:
- take a collection of documents
- merge them pairwise
- use a balanced binary reduction tree
- carry odd singletons forward unchanged
- continue until one root artifact remains

Reduction is the only semantic merge primitive.

---

## 6. Pairwise Merge Invariant

A central invariant of Orchestra is:

**All semantic synthesis must be pairwise.**

If an LLM is asked to merge several sibling documents in a single synthesis prompt, that is a framework error.

This invariant exists to preserve:

- bounded reasoning
- synthesis quality
- predictable merge topology
- model portability
- explicit provenance

Balanced pairwise reduction is the only valid merge topology for semantic synthesis.

---

## 7. Raw Text Output Contract

Each agent task produces exactly one raw-text output document.

This has several consequences:

- agents do not multiplex several artifacts into one response
- agents do not choose canonical output filenames
- output structure is determined by the graph, not by text formatting
- multi-artifact workflows are expressed as multiple tasks or patterns

The framework names, routes, and persists outputs. The model contributes text, not transport structure.

---

## 8. Persistence as a Workflow Boundary

Stores are not merely caches. They are explicit workflow boundaries.

Persisting results between stages provides:

- inspectability
- replayability
- human intervention points
- deterministic recovery
- auditable intermediate artifacts

A well-structured workflow may use several stores to mark distinct phases of work.

Examples of generic phase boundaries include:
- source acquisition
- normalization
- chunk generation
- extraction
- synthesis
- reporting

These are architectural roles, not fixed built-in store names.

---

## 9. Agent Role

Agents are bounded workers.

An agent may:
- inspect its provided documents
- use tools
- return one raw-text output document
- in some configurations, propose additional work through framework-approved mechanisms

An agent does not:
- directly mutate scheduler queues
- directly change dependency counters
- directly author internal runtime artifacts
- define persistence policy by formatting its output

The framework remains responsible for graph semantics.

---

## 10. Non-Agent Work

Not all work should be delegated to an LLM.

The framework should own non-semantic infrastructure tasks such as:

- downloading documents
- chunking
- persistence
- basic routing
- deterministic bookkeeping

Agent work should be reserved for tasks that require judgment, interpretation, extraction, comparison, or synthesis.

This keeps workflows stable and prevents accidental prompt-based simulation of infrastructure behavior.

---

## 11. Boundedness

Orchestra is built around bounded reasoning.

Every agent task should receive bounded input units. This may be achieved through:

- collection-level fanout
- chunking
- filtering
- pairwise reduction

Boundedness is not a prompt convention. It is a framework responsibility.

---

## 12. Generic Workflow Shape

Although Orchestra should remain independent of any specific workflow, many workflows naturally decompose into stages such as:

1. acquire or discover source material
2. normalize or filter it
3. split it into bounded units if needed
4. transform bounded units into extracted information
5. judge sufficiency or coverage where required
6. synthesize information pairwise
7. write final artifacts

This is not a required template, but it reflects the general design philosophy:
**local work first, synthesis later, synthesis only pairwise.**

---

## 13. Dynamic Graph Growth

The framework may support dynamic graph growth, but only through framework-controlled admission.

If agents propose new work, the scheduler still owns:

- validation
- dependency integration
- task admission
- execution order

This keeps graph growth explicit and inspectable.

Dynamic execution should not bypass the framework’s invariants.

---

## 14. Internal Execution Artifacts

High-level patterns may materialize into lower-level runtime artifacts, such as:

- seed tasks
- per-item agent tasks
- per-chunk tasks
- pairwise merge tasks
- collection/finalization tasks
- aliasing tasks

These are internal execution details. They are framework-private and need not appear in user-authored workflows.

The external model remains:
- stores
- patterns
- dependencies
- outputs

---

## 15. Design Commitments

Orchestra is built around the following commitments:

- stores are first-class collections
- patterns operate over collections
- singleton and multi-document inputs are treated uniformly
- chunking is explicit
- transform and reduce are distinct concerns
- semantic synthesis is always pairwise
- agent output remains raw text
- orchestration owns routing, persistence, and boundedness

This is the core design stance of the framework:

**Collections are explicit, local work is bounded, and synthesis is strictly pairwise.**
