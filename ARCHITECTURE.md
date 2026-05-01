# ARCHITECTURE.md: The Orchestra Framework

## 1. Purpose
Orchestra is a workflow engine for document-oriented reasoning systems. It coordinates persistent document collections, bounded agent work, dependency-aware scheduling, and strictly pairwise synthesis/selection.

---

## 2. Separation of Concerns
Orchestra separates **orchestration** from **labor**. 
- **Orchestration (Python):** Admitting nodes, tracking dependencies, materializing subgraphs, routing documents, and enforcing invariants.
- **Labor (LLM):** Bounded reasoning, tool usage, and producing raw-text outputs.

Agents may influence orchestration only through framework-approved tools (e.g., `schedule_nodes`), ensuring that the framework remains the authority over the execution graph.

---

## 3. Core Data Model

### 3.1 Document
The fundamental unit of work. Contains raw text content and metadata. Content is plain text.

### 3.2 Store
A persistent collection of documents. Stores act as workflow boundaries, providing inspectability and deterministic recovery.

---

## 4. Execution Model
Orchestra executes a dependency graph of tasks. High-level "Nodes" in a workflow are materialized into low-level "Task" subgraphs (seeds, agent-items, merges, tournaments, and finalizers).

---

## 5. Patterns

### 5.1 Map / Transform
Applies work independently to each document in a collection.

### 5.2 Chunk Map
Splits documents into bounded units and applies work to each chunk.

### 5.3 Filter Store (Triage)
Uses a bounded look-ahead (e.g., the first chunk of a document) to decide whether the entire document should be admitted to the output store.

### 5.4 Reduce (Synthesis)
Synthesizes a collection into a single artifact using a balanced binary reduction tree.

### 5.5 Tournament (Selection)
Selects the "best" document from a collection using a balanced binary bracket. In each match, an agent compares two documents and selects a winner. The winner moves up the bracket.

---

## 6. Pairwise Invariants
A central invariant of Orchestra is that all semantic aggregation is pairwise.
- **Pairwise Merge:** Synthesis prompts only ever see two sibling documents.
- **Pairwise Tournament:** Selection prompts only ever compare two documents.

This preserves bounded reasoning and prevents context window bloat.

---

## 7. Output Contracts
Most tasks produce raw-text documents. However, "Judge" roles (Filter/Tournament) follow a specific text-response protocol:
- **Filter:** Responds with `KEEP` or `DROP`.
- **Tournament:** Responds with `FIRST WINS` or `SECOND WINS`.

---

## 8. Agent Role and Tooling
Agents are bounded workers. They can use tools for research or to propose new workflow nodes via `schedule_nodes`. The framework allocates fresh stores and integrates these proposals into the dependency graph.

---

## 9. Dynamic Graph Growth
If an agent proposes new work, the scheduler owns validation and execution. Dynamic execution does not bypass the framework’s invariants.
