This is a lightweight LLM orchestration framework.

This version uses a dependency-aware scheduler with dynamic materialization for map and tree-reduce patterns.

It includes runtime dependency rewiring so downstream tasks wait for dynamically materialized finalization tasks rather than only the planner tasks.

It also avoids over-constraining planners on binding provenance and rewires only pre-existing downstream dependents to avoid deadlocks.

This code has not been hardened and should be run only in a sandboxed environment.
