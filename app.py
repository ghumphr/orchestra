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
from openai import OpenAI

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

DDGS_SOURCE = DDGS.__module__


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
# Logging helpers
# --------------------------

VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


def short_text(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarize_docs(docs: List[Document]) -> str:
    if not docs:
        return "0 docs"
    parts = [f"{d.id}({len(d.content)} chars)" for d in docs[:5]]
    if len(docs) > 5:
        parts.append(f"... +{len(docs) - 5} more")
    return f"{len(docs)} docs: " + ", ".join(parts)


def build_tool_specs() -> List[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_context",
                "description": "Get the current project context including the current date.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "duckduckgo_search",
                "description": "Search the web for a single query string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A single search query.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Fetch and extract readable text from a single URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch.",
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
    ]


# --------------------------
# Trace
# --------------------------

class Trace:
    def __init__(self, root: Path, run_id: str):
        self.root = root / run_id
        (self.root / "calls").mkdir(parents=True, exist_ok=True)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        log(f"[trace] initialized trace root at {self.root}")

    def log_call(self, task_id: str, system: str, user: str) -> None:
        d = self.root / "calls" / task_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "system.txt").write_text(system)
        (d / "user.txt").write_text(user)
        log(f"[trace] logged call for task={task_id}")

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
        log(f"[store] ready: {self.root}")

    def read(self, doc_id: str) -> Document:
        txt = (self.root / doc_id).read_text()
        meta_path = self.root / f".{doc_id}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"id": doc_id}
        doc = Document(id=doc_id, content=txt, metadata=meta)
        log(f"[store] read {self.root / doc_id} ({len(txt)} chars)")
        return doc

    def write(self, doc: Document) -> None:
        (self.root / doc.id).write_text(doc.content)
        meta = {
            **doc.metadata,
            "id": doc.id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / f".{doc.id}.json").write_text(json.dumps(meta, indent=2))
        log(f"[store] wrote {self.root / doc.id} ({len(doc.content)} chars)")

    def all(self) -> List[Document]:
        docs: List[Document] = []
        if not self.root.exists():
            log(f"[store] listing {self.root}: directory missing, returning 0 docs")
            return docs
        for p in sorted(self.root.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                docs.append(self.read(p.name))
        log(f"[store] listed {self.root}: {summarize_docs(docs)}")
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

        log(f"[agent] task={task_id} starting with {summarize_docs(docs)}")
        log(f"[agent] task={task_id} system prompt preview: {short_text(system, 160)}")

        messages: List[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        tool_specs = build_tool_specs()

        for turn_idx in range(12):
            log(f"[agent] task={task_id} model call turn={turn_idx + 1}")
            resp = self.client.chat.completions.create(
                model=self.cfg["default_model"],
                messages=messages,
                tools=tool_specs,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                log(f"[agent] task={task_id} requested {len(msg.tool_calls)} tool call(s)")
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    log(f"[agent] task={task_id} tool={tc.function.name} args={json.dumps(args, ensure_ascii=False)}")
                    result = tools[tc.function.name](args)
                    log(f"[agent] task={task_id} tool={tc.function.name} result preview={short_text(str(result), 200)}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                continue

            content = msg.content or ""
            log(f"[agent] task={task_id} final response chars={len(content)} preview={short_text(content, 200)}")

            doc_pattern = r"---DOC:([\w\.-]+)---\n(.*?)(?=\n---DOC:|$)"
            found = re.findall(doc_pattern, content, re.DOTALL)

            if found:
                docs_out = [Document(id=name.strip(), content=body.strip()) for name, body in found]
                log(f"[agent] task={task_id} parsed multi-doc output: {summarize_docs(docs_out)}")
                return docs_out

            output_id = conf.get("output_id", f"{task_id}.md")
            docs_out = [Document(id=output_id, content=content)]
            log(f"[agent] task={task_id} produced single doc: {summarize_docs(docs_out)}")
            return docs_out

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

        log(f"[ctx] store_root={self.store_root}")

    def get_store(self, name: str) -> Store:
        if name not in self.stores:
            self.stores[name] = Store(self.store_root / name)
        return self.stores[name]

    def resolve_binding(self, binding: str) -> List[Document]:
        docs = list(self.bindings.get(binding, []))
        log(f"[ctx] resolve binding '{binding}' -> {summarize_docs(docs)}")
        return docs

    def resolve_task_output(self, task_id: str) -> List[Document]:
        docs = list(self.task_outputs.get(task_id, []))
        log(f"[ctx] resolve task output '{task_id}' -> {summarize_docs(docs)}")
        return docs

    def publish_task_output(
        self,
        task_id: str,
        docs: List[Document],
        bind: Optional[str] = None,
    ) -> None:
        self.task_outputs[task_id] = list(docs)
        log(f"[ctx] publish task output task={task_id}: {summarize_docs(docs)}")
        if bind is not None:
            self.bindings[bind] = list(docs)
            self.binding_producers[bind] = task_id
            log(f"[ctx] bind '{bind}' now produced by task={task_id}: {summarize_docs(docs)}")

    def get_binding_producer(self, binding: str) -> Optional[str]:
        producer = self.binding_producers.get(binding)
        log(f"[ctx] binding producer for '{binding}' -> {producer}")
        return producer

    def tools(self) -> dict[str, Callable[[dict[str, Any]], str]]:
        def get_context(_: dict[str, Any]) -> str:
            result = json.dumps({
                "ok": True,
                "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "project": self.cfg.get("project_name", "Orchestra"),
            })
            log(f"[tool:get_context] -> {result}")
            return result

        def duckduckgo_search(args: dict[str, Any]) -> str:
            query = (args.get("query") or "").strip()
            if not query:
                result = json.dumps({
                    "ok": False,
                    "error": "missing_query",
                    "message": "duckduckgo_search requires a non-empty 'query' argument",
                })
                log("[tool:duckduckgo_search] missing query")
                return result

            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=8))
                result = json.dumps({
                    "ok": True,
                    "query": query,
                    "results": results,
                })
                log(f"[tool:duckduckgo_search] query='{query}' results={len(results)}")
                if results:
                    log(f"[tool:duckduckgo_search] first result preview={short_text(json.dumps(results[0], ensure_ascii=False), 200)}")
                return result
            except Exception as exc:
                result = json.dumps({
                    "ok": False,
                    "error": "search_failed",
                    "query": query,
                    "message": repr(exc),
                })
                log(f"[tool:duckduckgo_search] query='{query}' failed: {repr(exc)}")
                return result

        def fetch_url(args: dict[str, Any]) -> str:
            url = (args.get("url") or "").strip()
            if not url:
                result = json.dumps({
                    "ok": False,
                    "error": "missing_url",
                    "message": "fetch_url requires a non-empty 'url' argument",
                })
                log("[tool:fetch_url] missing url")
                return result

            try:
                resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                resp.raise_for_status()
                text = BeautifulSoup(resp.text, "html.parser").get_text()[:8000]
                result = json.dumps({
                    "ok": True,
                    "url": url,
                    "content": text,
                })
                log(f"[tool:fetch_url] url='{url}' chars={len(text)}")
                return result
            except Exception as exc:
                result = json.dumps({
                    "ok": False,
                    "error": "fetch_failed",
                    "url": url,
                    "message": repr(exc),
                })
                log(f"[tool:fetch_url] url='{url}' failed: {repr(exc)}")
                return result

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
        log(f"[planner] materializing task={task.id} kind={task.kind}")
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
        log(f"[planner] map task={task.id} source_binding='{source_binding}' -> {summarize_docs(input_docs)}")

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
            log(f"[planner] map task={task.id} produced empty finalize task")
            return created

        item_task_ids: List[str] = []

        for idx, doc in enumerate(input_docs):
            seed_task_id = f"{task.id}__seed_{idx}"
            item_task_id = f"{task.id}__item_{idx}"

            created.append(Task(
                id=seed_task_id,
                kind="seed",
                deps={task.id},
                payload={"docs": docs_to_payload([doc])},
            ))

            created.append(Task(
                id=item_task_id,
                kind="agent_map_item",
                deps={seed_task_id},
                payload={
                    "source_task_id": seed_task_id,
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

        log(f"[planner] map task={task.id} created {len(created)} task(s)")
        return created

    def _materialize_tree_reduce(self, task: Task) -> List[Task]:
        payload = task.payload
        source_binding = payload["input"]
        final_bind = payload["bind"]
        conf = payload["agent_config"]

        input_docs = self.ctx.resolve_binding(source_binding)
        log(f"[planner] tree_reduce task={task.id} source_binding='{source_binding}' -> {summarize_docs(input_docs)}")

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
            log(f"[planner] tree_reduce task={task.id} produced empty finalize task")
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
            log(f"[planner] tree_reduce task={task.id} passthrough single doc")
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

            log(f"[planner] tree_reduce task={task.id} round={round_idx} nodes={len(next_layer)}")
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

        log(f"[planner] tree_reduce task={task.id} created {len(created)} task(s)")
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
        log(f"[exec] task={task.id} kind={kind} payload_keys={list(payload.keys())}")

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
            log(f"[exec] write_store task={task.id} input_binding='{payload['input']}' -> {summarize_docs(docs)}")
            if not docs:
                log(f"[exec] write_store task={task.id} has no docs to write")
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
            log(f"[exec] pair_merge task={task.id} left={summarize_docs(left_docs)} right={summarize_docs(right_docs)}")
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
            log(f"[exec] collect task={task.id} combined -> {summarize_docs(combined)}")
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

        log(f"[sched] add task={task.id} kind={task.kind} deps={sorted(task.deps)} remaining={remaining}")

    def add_tasks(self, tasks: List[Task]) -> None:
        log(f"[sched] adding batch of {len(tasks)} task(s)")
        for task in tasks:
            self.add_task(task)

    def replace_dependency(
        self,
        old_dep: str,
        new_dep: str,
        restrict_to: Optional[Set[str]] = None,
    ) -> None:
        affected_set = set(self.dependents.get(old_dep, set()))
        if restrict_to is not None:
            affected_set &= restrict_to
        affected = list(sorted(affected_set))

        log(f"[sched] rewire old_dep={old_dep} -> new_dep={new_dep} affected={affected}")

        for child_id in affected:
            child = self.tasks[child_id]

            if old_dep not in child.deps:
                continue

            child.deps.remove(old_dep)
            child.deps.add(new_dep)

            self.dependents[old_dep].discard(child_id)
            self.dependents[new_dep].add(child_id)

            old_was_done = self.state.get(old_dep) == "done"
            new_is_done = self.state.get(new_dep) == "done"

            if old_was_done and not new_is_done:
                self.remaining_deps[child_id] += 1

            if not old_was_done and new_is_done:
                self.remaining_deps[child_id] -= 1
                if self.remaining_deps[child_id] == 0 and self.state[child_id] == "pending":
                    self.ready.append(child_id)

            log(f"[sched] rewired child={child_id} deps={sorted(child.deps)} remaining={self.remaining_deps[child_id]}")

    def mark_done(self, task_id: str) -> None:
        self.state[task_id] = "done"
        self.trace.log_task_state(task_id, "done", {"kind": self.tasks[task_id].kind})
        log(f"[sched] done task={task_id}")

        for child in sorted(self.dependents.get(task_id, [])):
            self.remaining_deps[child] -= 1
            log(f"[sched] decrement child={child} remaining={self.remaining_deps[child]}")
            if self.remaining_deps[child] == 0 and self.state[child] == "pending":
                self.ready.append(child)
                log(f"[sched] child ready: {child}")

    def mark_failed(self, task_id: str, exc: Exception) -> None:
        self.state[task_id] = "failed"
        self.trace.log_task_state(task_id, "failed", {"error": repr(exc)})
        log(f"[sched] FAILED task={task_id} error={repr(exc)}")

    def _run_materializer(self, task: Task) -> None:
        finalize_task_id = f"{task.id}__finalize"

        prior_dependents = set(self.dependents.get(task.id, set()))
        log(f"[sched] materializer start task={task.id} prior_dependents={sorted(prior_dependents)}")

        new_tasks = self.planner.materialize(task)

        self.executor.execute(task)
        self.mark_done(task.id)
        self.add_tasks(new_tasks)

        if finalize_task_id in self.tasks:
            self.replace_dependency(task.id, finalize_task_id, restrict_to=prior_dependents)

    def _run_concrete(self, task: Task) -> None:
        self.executor.execute(task)
        self.mark_done(task.id)

    def run(self) -> None:
        log("[sched] run loop starting")
        while self.ready:
            log(f"[sched] ready queue={list(self.ready)}")
            task_id = self.ready.popleft()

            if self.state[task_id] != "pending":
                log(f"[sched] skipping task={task_id} because state={self.state[task_id]}")
                continue

            if self.remaining_deps[task_id] != 0:
                log(
                    f"[sched] skipping task={task_id} because remaining_deps={self.remaining_deps[task_id]} "
                    f"(stale ready-queue entry)"
                )
                continue

            task = self.tasks[task_id]
            self.state[task_id] = "running"
            self.trace.log_task_state(task_id, "running", {"kind": task.kind})
            log(f"[sched] RUNNING task={task.id} kind={task.kind}")

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

        log("[sched] run loop complete; all tasks finished")


# --------------------------
# Workflow Compiler
# --------------------------

class WorkflowCompiler:
    """
    Static compiler from workflow nodes to initial tasks.

    Dynamic patterns compile to materializer tasks. Downstream dependencies
    initially target the materializer itself and are rewired at runtime to
    the materialized finalization task.
    """
    def __init__(self, wf: dict[str, Any]):
        self.wf = wf
        self.nodes = wf.get("nodes")
        if self.nodes is None:
            legacy_steps = wf.get("steps", [])
            self.nodes = [self._convert_legacy_step(step) for step in legacy_steps]

    def compile(self) -> List[Task]:
        tasks: List[Task] = []
        log(f"[compile] compiling {len(self.nodes)} node(s)")

        for node in self.nodes:
            deps = set(node.get("deps", []))
            kind = node["kind"]

            if kind in {"read_store", "write_store", "agent"}:
                t = Task(
                    id=node["id"],
                    kind=kind,
                    deps=deps,
                    payload=node,
                )
                tasks.append(t)
                log(f"[compile] task={t.id} kind={t.kind} deps={sorted(t.deps)}")
                continue

            if kind == "map":
                t = Task(
                    id=node["id"],
                    kind="map_materialize",
                    deps=deps,
                    payload=node,
                )
                tasks.append(t)
                log(f"[compile] task={t.id} kind={t.kind} deps={sorted(t.deps)}")
                continue

            if kind == "tree_reduce":
                t = Task(
                    id=node["id"],
                    kind="tree_reduce_materialize",
                    deps=deps,
                    payload=node,
                )
                tasks.append(t)
                log(f"[compile] task={t.id} kind={t.kind} deps={sorted(t.deps)}")
                continue

            raise ValueError(f"Unknown node kind: {kind}")

        log(f"[compile] produced {len(tasks)} initial task(s)")
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
    log(f"[main] run_id={run_id}")
    log(f"[main] workflow_path={cfg['workflow_path']}")
    log(f"[main] store_root={cfg['store_root']}")
    log(f"[main] traces_root={cfg['traces_root']}")
    log(f"[main] DDGS source={DDGS_SOURCE}")

    trace = Trace(Path(cfg["traces_root"]), run_id)

    ctx = ExecutionContext(cfg, trace)
    planner = TaskPlanner(ctx)
    executor = TaskExecutor(ctx)
    scheduler = Scheduler(executor, planner, trace)

    tasks = WorkflowCompiler(wf).compile()
    scheduler.add_tasks(tasks)

    log("[main] initial stores snapshot:")
    for store_name in ["input_store", "research_store", "output_store"]:
        docs = ctx.get_store(store_name).all()
        log(f"[main]   {store_name}: {summarize_docs(docs)}")

    scheduler.run()

    log("[main] final bindings snapshot:")
    for binding_name, docs in sorted(ctx.bindings.items()):
        log(f"[main]   binding '{binding_name}': {summarize_docs(docs)}")

    log("[main] final binding producers:")
    for binding_name, producer in sorted(ctx.binding_producers.items()):
        log(f"[main]   binding '{binding_name}' producer={producer}")

    log("[main] final task states:")
    for task_id in sorted(scheduler.tasks):
        log(
            f"[main]   task={task_id} "
            f"kind={scheduler.tasks[task_id].kind} "
            f"state={scheduler.state.get(task_id)} "
            f"remaining_deps={scheduler.remaining_deps.get(task_id)}"
        )

    log("[main] final stores snapshot:")
    for store_name in ["input_store", "research_store", "output_store"]:
        docs = ctx.get_store(store_name).all()
        log(f"[main]   {store_name}: {summarize_docs(docs)}")

    print(f"DONE: {run_id}")


if __name__ == "__main__":
    main()
