from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict, deque
import json
import re

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI


# --------------------------
# Models
# --------------------------

@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    kind: str
    deps: Set[str] = field(default_factory=set)
    payload: dict[str, Any] = field(default_factory=dict)


def docs_to_payload(docs: List[Document]) -> List[dict[str, Any]]:
    return [
        {
            "id": d.id,
            "content": d.content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


def docs_from_payload(items: List[dict[str, Any]]) -> List[Document]:
    return [
        Document(
            id=item["id"],
            content=item["content"],
            metadata=item.get("metadata", {}),
        )
        for item in items
    ]


# --------------------------
# Trace
# --------------------------

class Trace:
    def __init__(self, root: Path, run_id: str):
        self.root = root / run_id
        (self.root / "calls").mkdir(parents=True, exist_ok=True)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)

    def log_call(self, task_id: str, system: str, user: str) -> None:
        d = self.root / "calls" / task_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "system.txt").write_text(system)
        (d / "user.txt").write_text(user)

    def log_task_state(self, task_id: str, state: str, meta: Optional[dict[str, Any]] = None) -> None:
        d = self.root / "tasks" / task_id
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "state": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        }
        (d / f"{state}.json").write_text(json.dumps(payload, indent=2))


# --------------------------
# Store
# --------------------------

class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, doc_id: str) -> Document:
        txt = (self.root / doc_id).read_text()
        meta_path = self.root / f".{doc_id}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"id": doc_id}
        return Document(id=doc_id, content=txt, metadata=meta)

    def write(self, doc: Document) -> None:
        (self.root / doc.id).write_text(doc.content)
        meta = {
            **doc.metadata,
            "id": doc.id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / f".{doc.id}.json").write_text(json.dumps(meta, indent=2))

    def all(self) -> List[Document]:
        docs: List[Document] = []
        if not self.root.exists():
            return docs
        for p in sorted(self.root.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                docs.append(self.read(p.name))
        return docs


# --------------------------
# Agent
# --------------------------

class Agent:
    def __init__(self, cfg: dict[str, Any], trace: Trace):
        self.client = OpenAI(base_url=cfg["openai_base_url"], api_key="dummy")
        self.cfg = cfg
        self.trace = trace

    def run(
        self,
        docs: List[Document],
        conf: dict[str, Any],
        task_id: str,
        tools: dict[str, Callable[[dict[str, Any]], str]],
    ) -> List[Document]:
        system = conf["system"]
        user = "\n\n".join(f"## {d.id}\n{d.content}" for d in docs)
        self.trace.log_call(task_id, system, user)

        messages: List[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for _ in range(12):
            resp = self.client.chat.completions.create(
                model=self.cfg["default_model"],
                messages=messages,
                tools=[{"type": "function", "function": {"name": name}} for name in tools],
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    result = tools[tc.function.name](args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                continue

            content = msg.content or ""
            doc_pattern = r"---DOC:([\w\.-]+)---\n(.*?)(?=\n---DOC:|$)"
            found = re.findall(doc_pattern, content, re.DOTALL)

            if found:
                return [Document(id=name.strip(), content=body.strip()) for name, body in found]

            output_id = conf.get("output_id", f"{task_id}.md")
            return [Document(id=output_id, content=content)]

        return []


# --------------------------
# Execution Context
# --------------------------

class ExecutionContext:
    """
    bindings: symbolic names visible to workflow nodes
    binding_producers: which task most recently produced a binding
    task_outputs: concrete outputs written by task id
    """
    def __init__(self, cfg: dict[str, Any], trace: Trace):
        self.cfg = cfg
        self.trace = trace
        self.agent = Agent(cfg, trace)
        self.store_root = Path(cfg["store_root"])
        self.store_root.mkdir(parents=True, exist_ok=True)

        self.stores: Dict[str, Store] = {}
        self.bindings: Dict[str, List[Document]] = {}
        self.binding_producers: Dict[str, str] = {}
        self.task_outputs: Dict[str, List[Document]] = {}

    def get_store(self, name: str) -> Store:
        if name not in self.stores:
            self.stores[name] = Store(self.store_root / name)
        return self.stores[name]

    def resolve_binding(self, binding: str) -> List[Document]:
        return list(self.bindings.get(binding, []))

    def resolve_task_output(self, task_id: str) -> List[Document]:
        return list(self.task_outputs.get(task_id, []))

    def publish_task_output(
        self,
        task_id: str,
        docs: List[Document],
        bind: Optional[str] = None,
    ) -> None:
        self.task_outputs[task_id] = list(docs)
        if bind is not None:
            self.bindings[bind] = list(docs)
            self.binding_producers[bind] = task_id

    def get_binding_producer(self, binding: str) -> Optional[str]:
        return self.binding_producers.get(binding)

    def tools(self) -> dict[str, Callable[[dict[str, Any]], str]]:
        def get_context(_: dict[str, Any]) -> str:
            return json.dumps({
                "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "project": self.cfg.get("project_name", "Orchestra"),
            })

        def duckduckgo_search(args: dict[str, Any]) -> str:
            query = args.get("query") or args.get("q", "")
            return json.dumps(list(DDGS().text(query, max_results=5)))

        def fetch_url(args: dict[str, Any]) -> str:
            url = args.get("url", "")
            text = httpx.get(url, timeout=20.0).text
            return BeautifulSoup(text, "html.parser").get_text()[:8000]

        return {
            "get_context": get_context,
            "duckduckgo_search": duckduckgo_search,
            "fetch_url": fetch_url,
        }


# --------------------------
# Planner
# --------------------------

class TaskPlanner:
    """
    Expands dynamic pattern tasks into concrete scheduler tasks.

    The planner only creates tasks. It does not execute them.
    """
    def __init__(self, ctx: ExecutionContext):
        self.ctx = ctx

    def materialize(self, task: Task) -> List[Task]:
        if task.kind == "map_materialize":
            return self._materialize_map(task)
        if task.kind == "tree_reduce_materialize":
            return self._materialize_tree_reduce(task)
        raise ValueError(f"Task kind is not materializable: {task.kind}")

    def _materialize_map(self, task: Task) -> List[Task]:
        payload = task.payload
        source_binding = payload["input"]
        final_bind = payload["bind"]
        conf = payload["agent_config"]

        input_docs = self.ctx.resolve_binding(source_binding)
        upstream_producer = self.ctx.get_binding_producer(source_binding)
        if upstream_producer is None:
            raise ValueError(
                f"Cannot materialize map '{task.id}': binding '{source_binding}' has no producer"
            )

        created: List[Task] = []

        if not input_docs:
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="collect",
                deps={task.id},
                payload={
                    "source_task_ids": [],
                    "bind": final_bind,
                },
            ))
            return created

        item_task_ids: List[str] = []

        for idx, doc in enumerate(input_docs):
            leaf_task_id = f"{task.id}__seed_{idx}"
            item_task_id = f"{task.id}__item_{idx}"

            created.append(Task(
                id=leaf_task_id,
                kind="seed",
                deps={task.id},
                payload={"docs": docs_to_payload([doc])},
            ))

            created.append(Task(
                id=item_task_id,
                kind="agent_map_item",
                deps={leaf_task_id},
                payload={
                    "source_task_id": leaf_task_id,
                    "agent_config": conf,
                },
            ))
            item_task_ids.append(item_task_id)

        created.append(Task(
            id=f"{task.id}__finalize",
            kind="collect",
            deps=set(item_task_ids),
            payload={
                "source_task_ids": item_task_ids,
                "bind": final_bind,
            },
        ))

        return created

    def _materialize_tree_reduce(self, task: Task) -> List[Task]:
        payload = task.payload
        source_binding = payload["input"]
        final_bind = payload["bind"]
        conf = payload["agent_config"]

        input_docs = self.ctx.resolve_binding(source_binding)
        upstream_producer = self.ctx.get_binding_producer(source_binding)
        if upstream_producer is None:
            raise ValueError(
                f"Cannot materialize tree_reduce '{task.id}': binding '{source_binding}' has no producer"
            )

        created: List[Task] = []

        if not input_docs:
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="collect",
                deps={task.id},
                payload={
                    "source_task_ids": [],
                    "bind": final_bind,
                },
            ))
            return created

        if len(input_docs) == 1:
            passthrough_id = f"{task.id}__seed_single"
            created.append(Task(
                id=passthrough_id,
                kind="seed",
                deps={task.id},
                payload={"docs": docs_to_payload(input_docs)},
            ))
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="alias",
                deps={passthrough_id},
                payload={
                    "source_task_id": passthrough_id,
                    "bind": final_bind,
                },
            ))
            return created

        current_layer: List[str] = []
        for idx, doc in enumerate(input_docs):
            seed_id = f"{task.id}__seed_{idx}"
            created.append(Task(
                id=seed_id,
                kind="seed",
                deps={task.id},
                payload={"docs": docs_to_payload([doc])},
            ))
            current_layer.append(seed_id)

        round_idx = 0
        while len(current_layer) > 1:
            next_layer: List[str] = []
            for pair_idx in range(0, len(current_layer), 2):
                pair = current_layer[pair_idx:pair_idx + 2]

                if len(pair) == 1:
                    next_layer.append(pair[0])
                    continue

                left_task_id, right_task_id = pair
                merge_task_id = f"{task.id}__merge_r{round_idx}_n{pair_idx // 2}"

                created.append(Task(
                    id=merge_task_id,
                    kind="agent_pair_merge",
                    deps={left_task_id, right_task_id},
                    payload={
                        "left_task_id": left_task_id,
                        "right_task_id": right_task_id,
                        "agent_config": conf,
                    },
                ))
                next_layer.append(merge_task_id)

            current_layer = next_layer
            round_idx += 1

        created.append(Task(
            id=f"{task.id}__finalize",
            kind="alias",
            deps={current_layer[0]},
            payload={
                "source_task_id": current_layer[0],
                "bind": final_bind,
            },
        ))

        return created


# --------------------------
# Executor
# --------------------------

class TaskExecutor:
    def __init__(self, ctx: ExecutionContext):
        self.ctx = ctx

    def execute(self, task: Task) -> None:
        kind = task.kind
        payload = task.payload

        if kind == "seed":
            docs = docs_from_payload(payload["docs"])
            self.ctx.publish_task_output(task.id, docs)
            return

        if kind == "read_store":
            docs = self.ctx.get_store(payload["store"]).all()
            self.ctx.publish_task_output(task.id, docs, bind=payload["bind"])
            return

        if kind == "write_store":
            docs = self.ctx.resolve_binding(payload["input"])
            store = self.ctx.get_store(payload["store"])
            for doc in docs:
                store.write(doc)
            self.ctx.publish_task_output(task.id, [])
            return

        if kind == "agent":
            docs = self.ctx.resolve_binding(payload["input"])
            result = self.ctx.agent.run(
                docs=docs,
                conf=payload["agent_config"],
                task_id=task.id,
                tools=self.ctx.tools(),
            )
            self.ctx.publish_task_output(task.id, result, bind=payload["bind"])
            return

        if kind == "agent_map_item":
            docs = self.ctx.resolve_task_output(payload["source_task_id"])
            result = self.ctx.agent.run(
                docs=docs,
                conf=payload["agent_config"],
                task_id=task.id,
                tools=self.ctx.tools(),
            )
            self.ctx.publish_task_output(task.id, result)
            return

        if kind == "agent_pair_merge":
            left_docs = self.ctx.resolve_task_output(payload["left_task_id"])
            right_docs = self.ctx.resolve_task_output(payload["right_task_id"])
            result = self.ctx.agent.run(
                docs=left_docs + right_docs,
                conf=payload["agent_config"],
                task_id=task.id,
                tools=self.ctx.tools(),
            )
            self.ctx.publish_task_output(task.id, result)
            return

        if kind == "alias":
            docs = self.ctx.resolve_task_output(payload["source_task_id"])
            self.ctx.publish_task_output(task.id, docs, bind=payload["bind"])
            return

        if kind == "collect":
            combined: List[Document] = []
            for source_task_id in payload["source_task_ids"]:
                combined.extend(self.ctx.resolve_task_output(source_task_id))
            self.ctx.publish_task_output(task.id, combined, bind=payload["bind"])
            return

        if kind in {"map_materialize", "tree_reduce_materialize"}:
            self.ctx.publish_task_output(task.id, [])
            return

        raise ValueError(f"Unknown task kind: {kind}")


# --------------------------
# Scheduler
# --------------------------

class Scheduler:
    def __init__(self, executor: TaskExecutor, planner: TaskPlanner, trace: Trace):
        self.executor = executor
        self.planner = planner
        self.trace = trace

        self.tasks: Dict[str, Task] = {}
        self.state: Dict[str, str] = {}
        self.remaining_deps: Dict[str, int] = {}
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self.ready: deque[str] = deque()

    def add_task(self, task: Task) -> None:
        if task.id in self.tasks:
            raise ValueError(f"Duplicate task id: {task.id}")

        unresolved = [dep for dep in task.deps if dep not in self.state]
        if unresolved:
            raise ValueError(f"Task {task.id} has unknown dependencies: {unresolved}")

        self.tasks[task.id] = task
        self.state[task.id] = "pending"

        remaining = 0
        for dep in task.deps:
            self.dependents[dep].add(task.id)
            if self.state[dep] != "done":
                remaining += 1

        self.remaining_deps[task.id] = remaining
        if remaining == 0:
            self.ready.append(task.id)

    def add_tasks(self, tasks: List[Task]) -> None:
        for task in tasks:
            self.add_task(task)

    def mark_done(self, task_id: str) -> None:
        self.state[task_id] = "done"
        self.trace.log_task_state(task_id, "done", {"kind": self.tasks[task_id].kind})

        for child in sorted(self.dependents.get(task_id, [])):
            self.remaining_deps[child] -= 1
            if self.remaining_deps[child] == 0 and self.state[child] == "pending":
                self.ready.append(child)

    def mark_failed(self, task_id: str, exc: Exception) -> None:
        self.state[task_id] = "failed"
        self.trace.log_task_state(task_id, "failed", {"error": repr(exc)})

    def _run_materializer(self, task: Task) -> None:
        new_tasks = self.planner.materialize(task)
        self.executor.execute(task)
        self.mark_done(task.id)
        self.add_tasks(new_tasks)

    def _run_concrete(self, task: Task) -> None:
        self.executor.execute(task)
        self.mark_done(task.id)

    def run(self) -> None:
        while self.ready:
            task_id = self.ready.popleft()
            if self.state[task_id] != "pending":
                continue

            task = self.tasks[task_id]
            self.state[task_id] = "running"
            self.trace.log_task_state(task_id, "running", {"kind": task.kind})

            try:
                if task.kind in {"map_materialize", "tree_reduce_materialize"}:
                    self._run_materializer(task)
                else:
                    self._run_concrete(task)
            except Exception as exc:
                self.mark_failed(task_id, exc)
                raise

        unfinished = [tid for tid, st in self.state.items() if st in {"pending", "running"}]
        if unfinished:
            raise RuntimeError(f"Unfinished tasks remain: {unfinished}")


# --------------------------
# Workflow Compiler
# --------------------------

class WorkflowCompiler:
    """
    Static compiler from workflow nodes to initial tasks.

    Dynamic patterns compile to materializer tasks, but downstream deps target
    their finalization tasks.
    """
    def __init__(self, wf: dict[str, Any]):
        self.wf = wf
        self.nodes = wf.get("nodes")
        if self.nodes is None:
            legacy_steps = wf.get("steps", [])
            self.nodes = [self._convert_legacy_step(step) for step in legacy_steps]

    def compile(self) -> List[Task]:
        tasks: List[Task] = []

        final_task_id_by_node_id: Dict[str, str] = {}
        for node in self.nodes:
            if node["kind"] in {"map", "tree_reduce"}:
                final_task_id_by_node_id[node["id"]] = f"{node['id']}__finalize"
            else:
                final_task_id_by_node_id[node["id"]] = node["id"]

        for node in self.nodes:
            deps = {final_task_id_by_node_id[dep] for dep in node.get("deps", [])}
            kind = node["kind"]

            if kind in {"read_store", "write_store", "agent"}:
                tasks.append(Task(
                    id=node["id"],
                    kind=kind,
                    deps=deps,
                    payload=node,
                ))
                continue

            if kind == "map":
                tasks.append(Task(
                    id=node["id"],
                    kind="map_materialize",
                    deps=deps,
                    payload=node,
                ))
                continue

            if kind == "tree_reduce":
                tasks.append(Task(
                    id=node["id"],
                    kind="tree_reduce_materialize",
                    deps=deps,
                    payload=node,
                ))
                continue

            raise ValueError(f"Unknown node kind: {kind}")

        return tasks

    def _convert_legacy_step(self, step: dict[str, Any]) -> dict[str, Any]:
        stype = step["type"]

        if stype == "read_documents":
            return {
                "id": step["id"],
                "kind": "read_store",
                "store": step["store"],
                "bind": step["output"],
                "deps": step.get("deps", []),
            }

        if stype == "write_documents":
            return {
                "id": step["id"],
                "kind": "write_store",
                "store": step["store"],
                "input": step["input"],
                "deps": step.get("deps", []),
            }

        if stype == "run_agent":
            return {
                "id": step["id"],
                "kind": "agent",
                "input": step["input"],
                "bind": step["output"],
                "agent_config": step["agent_config"],
                "deps": step.get("deps", []),
            }

        if stype == "tree_reduce":
            return {
                "id": step["id"],
                "kind": "tree_reduce",
                "input": step["input"],
                "bind": step["output"],
                "agent_config": step["agent_config"],
                "deps": step.get("deps", []),
            }

        if stype == "map":
            return {
                "id": step["id"],
                "kind": "map",
                "input": step["input"],
                "bind": step["output"],
                "agent_config": step["agent_config"],
                "deps": step.get("deps", []),
            }

        raise ValueError(f"Unsupported legacy step type: {stype}")


# --------------------------
# Main
# --------------------------

def main() -> None:
    cfg = json.loads(Path(".orchestra.json").read_text())
    wf = json.loads(Path(cfg["workflow_path"]).read_text())

    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace = Trace(Path(cfg["traces_root"]), run_id)

    ctx = ExecutionContext(cfg, trace)
    planner = TaskPlanner(ctx)
    executor = TaskExecutor(ctx)
    scheduler = Scheduler(executor, planner, trace)

    tasks = WorkflowCompiler(wf).compile()
    scheduler.add_tasks(tasks)
    scheduler.run()

    print(f"DONE: {run_id}")


if __name__ == "__main__":
    main()
