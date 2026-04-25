# ARCHITECTURE.md: The Orchestra Framework

## 1. Philosophy: Separation of Powers
Orchestra is built on a strict decoupling of **Orchestration** (The Framework) and **Labor** (The Agent).

*   **The Framework (Runner):** Acts as the Project Manager/Kernel. It is the only entity aware of the "10,000-foot view." It manages the instruction pointer, handles data logistics, and executes patterns.
*   **The Agent (Laborer):** Acts as a stateless "Reasoning Factory." It is a specialized worker that consumes specific inputs and produces specific outputs (Documents). It has no knowledge of its position in a workflow.

## 2. Infrastructure: Workspace vs. Stores

A fundamental boundary exists between where labor happens and how results are communicated.

### The Workspace (The Project Site)
*   **Nature:** Raw filesystem.
*   **Purpose:** This is the physical site of the project. Agents use tools (`write_file`, `sh`, etc.) to perform auxiliary labor, generate code, or create temporary artifacts.
*   **Managed?** No. The framework does not track metadata, lineage, or state within the workspace. It is the "Real World" where the agent's tools have side effects.

### The Stores (The Communication Infrastructure)
*   **Nature:** Managed repository with sidecar metadata.
*   **Purpose:** Stores are the formal communication channels between steps. 
*   **The Sidecar Pattern:** Every document in a store has a hidden `.filename.json` companion. This persists technical metadata (timestamps, lineage, IDs) without polluting the raw text content that the LLM needs to process.
*   **Handoffs:** Agent A communicates to Agent B by returning a document that the Framework then writes to a Store. Agent B then reads from that Store.

## 3. The Runner: Pattern-Based Orchestration
Complexity is handled by the framework through hard-coded **Patterns**. This prevents "agent drift" and ensures deterministic execution of complex logic.

*   **Map Pattern:** Parallelizes labor by creating one task per input document.
*   **Tree-Reduce Pattern:** A hierarchical synthesis pattern. It merges $N$ documents into 1 via pairwise reduction (Log N depth). This prevents context-window saturation and ensures high-fidelity synthesis by only asking an LLM to merge two concepts at a time.
*   **Read/Write Patterns:** Explicit steps that move data between the internal **Blackboard** (in-memory state) and the physical **Stores** (persistent communication).

## 4. The Agent: The Document Factory
The Agent's primary output is the **Document**.

*   **Statelessness:** The agent does not remember previous tasks. It relies entirely on the documents provided in its current context.
*   **Multi-Document Returns:** In gathering or "Scouting" tasks, the Agent can produce multiple documents in a single turn using the `---DOC:filename---` syntax. The framework parses these and dispatches them to the appropriate stores.

## 5. Temporal Grounding: The 2024 Problem
LLMs suffer from "Pre-training Bias," where their internal weights suggest a cutoff date (e.g., 2024) is the present.

*   **The Solution:** The `get_context` tool.
*   **Mechanism:** Instead of injecting the date into the prompt (which the agent might ignore or treat as a "future" anomaly), the agent is provided a tool to discover its environment.
*   **Reasoning:** When the agent encounters contradictory information (search results from 2026), it uses `get_context` to ground itself in the "Project Reality" (April 2026).

## 6. Execution Flow
1.  **Hydration:** The Runner reads `workflow.json` and initializes the Blackboard.
2.  **Pattern Execution:** The Runner iterates through steps.
3.  **Task Dispatch:** If a step requires an Agent, the Runner prepares the inputs and tools.
4.  **Handoff:** The Agent returns Documents $\rightarrow$ The Runner writes them to a Store $\rightarrow$ The Blackboard is updated for the next pattern.
5.  **Completion:** The process continues until the instruction pointer reaches the end of the workflow.
