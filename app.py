from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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

VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


def summarize_docs(docs: List["Document"]) -> str:
    if not docs:
        return "0 docs"
    parts = [f"{d.id}({len(d.content)} chars)" for d in docs[:5]]
    if len(docs) > 5:
        parts.append(f"... +{len(docs) - 5} more")
    return f"{len(docs)} docs: " + ", ".join(parts)


def sanitize_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"


def chunk_text_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    if not text:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((start, end, text[start:end]))
        if end >= n:
            break
        start = end - chunk_overlap

    return chunks


def first_chunk_text(text: str, chunk_size: int) -> Tuple[int, int, str]:
    chunks = chunk_text_chars(text, chunk_size, 0)
    if not chunks:
        return (0, 0, "")
    return chunks[0]


def parse_keep_drop(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip().upper()
        if not stripped:
            continue
        if stripped.startswith("KEEP"):
            return True
        if stripped.startswith("DROP"):
            return False
        break
    return False


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


@dataclass
class NodeProposal:
    id: str
    kind: str
    payload: dict[str, Any]
    deps: Set[str] = field(default_factory=set)
    proposed_by: Optional[str] = None


def docs_to_payload(docs: List[Document]) -> List[dict[str, Any]]:
    return [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in docs]


def docs_from_payload(items: List[dict[str, Any]]) -> List[Document]:
    return [Document(id=x["id"], content=x["content"], metadata=x.get("metadata", {})) for x in items]


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
                        "query": {"type": "string"},
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
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_nodes",
                "description": "Schedule one or more workflow nodes to run after the current task completes. The framework will allocate fresh output stores when output_store_base is provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "kind": {
                                        "type": "string",
                                        "enum": ["map", "transform", "chunk_map", "reduce", "write_store", "filter_store"],
                                    },
                                    "input_stores": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "input": {"type": "string"},
                                    "output_store": {"type": "string"},
                                    "output_store_base": {"type": "string"},
                                    "bind": {"type": "string"},
                                    "prompt": {"type": "string"},
                                    "reduce_prompt": {"type": "string"},
                                    "chunk_size": {"type": "integer"},
                                    "chunk_overlap": {"type": "integer"},
                                    "deps": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["id", "kind"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["nodes"],
                    "additionalProperties": False,
                },
            },
        },
    ]


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
        meta = {**doc.metadata, "id": doc.id, "updated_at": datetime.now(timezone.utc).isoformat()}
        (self.root / f".{doc.id}.json").write_text(json.dumps(meta, indent=2))
        log(f"[store] wrote {self.root / doc.id} ({len(doc.content)} chars)")

    def all(self) -> List[Document]:
        docs: List[Document] = []
        if not self.root.exists():
            return docs
        for p in sorted(self.root.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                docs.append(self.read(p.name))
        log(f"[store] listed {self.root}: {summarize_docs(docs)}")
        return docs


class Agent:
    def __init__(self, cfg: dict[str, Any], trace: Trace):
        self.client = OpenAI(base_url=cfg["openai_base_url"], api_key="dummy")
        self.cfg = cfg
        self.trace = trace

    def run(
        self,
        docs: List[Document],
        prompt: str,
        task_id: str,
        tools: dict[str, Callable[[dict[str, Any]], str]],
        output_id: Optional[str] = None,
    ) -> List[Document]:
        user = "\n\n".join(f"## {d.id}\n{d.content}" for d in docs)
        self.trace.log_call(task_id, prompt, user)
        log(f"[agent] task={task_id} starting with {summarize_docs(docs)}")

        messages: List[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
        tool_specs = build_tool_specs()

        for turn_idx in range(16):
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
                    log(f"[agent] task={task_id} tool={tc.function.name} result preview={str(result)[:220]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                continue

            content = msg.content or ""
            doc_id = output_id or f"{task_id}.md"
            docs_out = [Document(id=doc_id, content=content)]
            log(f"[agent] task={task_id} produced single doc: {summarize_docs(docs_out)}")
            return docs_out

        log(f"[agent] task={task_id} exhausted max turns without final content")
        return []


class ExecutionContext:
    def __init__(self, cfg: dict[str, Any], trace: Trace):
        self.cfg = cfg
        self.trace = trace
        self.agent = Agent(cfg, trace)
        self.store_root = Path(cfg["store_root"])
        self.store_root.mkdir(parents=True, exist_ok=True)

        self.stores: Dict[str, Store] = {}
        self.bindings: Dict[str, List[Document]] = {}
        self.task_outputs: Dict[str, List[Document]] = {}
        self.current_task_id: Optional[str] = None
        self.pending_node_proposals: List[NodeProposal] = []
        self.allocated_store_names: Set[str] = set()

        log(f"[ctx] store_root={self.store_root}")

    def get_store(self, name: str) -> Store:
        if name not in self.stores:
            self.stores[name] = Store(self.store_root / name)
        return self.stores[name]

    def allocate_store_name(self, base: str, task_id: str) -> str:
        base_clean = sanitize_name(base)
        task_clean = sanitize_name(task_id)
        candidate = f"{base_clean}__{task_clean}"
        suffix = 1
        while candidate in self.allocated_store_names or (self.store_root / candidate).exists():
            suffix += 1
            candidate = f"{base_clean}__{task_clean}__{suffix}"
        self.allocated_store_names.add(candidate)
        log(f"[ctx] allocated store name base='{base}' task='{task_id}' -> '{candidate}'")
        return candidate

    def set_current_task(self, task_id: Optional[str]) -> None:
        self.current_task_id = task_id
        log(f"[ctx] current_task = {task_id}")

    def add_node_proposal(self, proposal: NodeProposal) -> None:
        self.pending_node_proposals.append(proposal)
        log(f"[ctx] proposed node id={proposal.id} kind={proposal.kind} deps={sorted(proposal.deps)} payload_keys={list(proposal.payload.keys())}")

    def consume_node_proposals(self) -> List[NodeProposal]:
        proposals = list(self.pending_node_proposals)
        self.pending_node_proposals.clear()
        log(f"[ctx] consuming {len(proposals)} node proposal(s)")
        return proposals

    def resolve_binding(self, binding: str) -> List[Document]:
        docs = list(self.bindings.get(binding, []))
        log(f"[ctx] resolve binding '{binding}' -> {summarize_docs(docs)}")
        return docs

    def resolve_task_output(self, task_id: str) -> List[Document]:
        docs = list(self.task_outputs.get(task_id, []))
        log(f"[ctx] resolve task output '{task_id}' -> {summarize_docs(docs)}")
        return docs

    def publish_task_output(self, task_id: str, docs: List[Document], bind: Optional[str] = None) -> None:
        self.task_outputs[task_id] = list(docs)
        log(f"[ctx] publish task output task={task_id}: {summarize_docs(docs)}")
        if bind:
            self.bindings[bind] = list(docs)
            log(f"[ctx] bind '{bind}' now has {summarize_docs(docs)}")

    def resolve_store_union(self, store_names: List[str]) -> List[Document]:
        docs: List[Document] = []
        for store_name in store_names:
            docs.extend(self.get_store(store_name).all())
        log(f"[ctx] resolve input_stores {store_names} -> {summarize_docs(docs)}")
        return docs

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
                result = json.dumps({"ok": False, "error": "missing_query"})
                log("[tool:duckduckgo_search] missing query")
                return result
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=8))
                result = json.dumps({"ok": True, "query": query, "results": results})
                log(f"[tool:duckduckgo_search] query='{query}' results={len(results)}")
                return result
            except Exception as exc:
                result = json.dumps({"ok": False, "error": "search_failed", "message": repr(exc)})
                log(f"[tool:duckduckgo_search] failed: {repr(exc)}")
                return result

        def fetch_url(args: dict[str, Any]) -> str:
            url = (args.get("url") or "").strip()
            if not url:
                result = json.dumps({"ok": False, "error": "missing_url"})
                log("[tool:fetch_url] missing url")
                return result
            try:
                resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                resp.raise_for_status()
                text = BeautifulSoup(resp.text, "html.parser").get_text()[:8000]
                result = json.dumps({"ok": True, "url": url, "content": text})
                log(f"[tool:fetch_url] url='{url}' chars={len(text)}")
                return result
            except Exception as exc:
                result = json.dumps({"ok": False, "error": "fetch_failed", "message": repr(exc)})
                log(f"[tool:fetch_url] failed: {repr(exc)}")
                return result

        def schedule_nodes(args: dict[str, Any]) -> str:
            if self.current_task_id is None:
                result = json.dumps({"ok": False, "error": "no_current_task"})
                log("[tool:schedule_nodes] no current task")
                return result

            raw_nodes = args.get("nodes")
            if not isinstance(raw_nodes, list) or not raw_nodes:
                result = json.dumps({"ok": False, "error": "missing_nodes"})
                log("[tool:schedule_nodes] missing nodes")
                return result

            accepted: List[dict[str, Any]] = []
            rejected: List[dict[str, Any]] = []

            for raw_node in raw_nodes:
                if not isinstance(raw_node, dict):
                    rejected.append({"reason": "node_not_object"})
                    continue

                node_id = (raw_node.get("id") or "").strip()
                kind = raw_node.get("kind")
                deps = {self.current_task_id}
                for dep in raw_node.get("deps", []) or []:
                    if isinstance(dep, str) and dep.strip():
                        deps.add(dep.strip())

                if not node_id:
                    rejected.append({"reason": "missing_id"})
                    continue

                if kind not in {"map", "transform", "chunk_map", "reduce", "write_store", "filter_store"}:
                    rejected.append({"id": node_id, "reason": f"invalid_kind:{kind}"})
                    continue

                try:
                    payload = self._build_payload_from_proposed_node(raw_node, node_id)
                except ValueError as exc:
                    rejected.append({"id": node_id, "reason": str(exc)})
                    continue

                proposal = NodeProposal(
                    id=node_id,
                    kind=kind,
                    payload=payload,
                    deps=deps,
                    proposed_by=self.current_task_id,
                )
                self.add_node_proposal(proposal)
                accepted.append({
                    "id": node_id,
                    "kind": kind,
                    "output_store": payload.get("output_store", ""),
                })

            result = json.dumps({
                "ok": len(accepted) > 0,
                "accepted": accepted,
                "rejected": rejected,
            })
            log(f"[tool:schedule_nodes] accepted={accepted} rejected={rejected}")
            return result

        return {
            "get_context": get_context,
            "duckduckgo_search": duckduckgo_search,
            "fetch_url": fetch_url,
            "schedule_nodes": schedule_nodes,
        }

    def _build_payload_from_proposed_node(self, raw_node: dict[str, Any], node_id: str) -> dict[str, Any]:
        kind = raw_node["kind"]

        if kind == "write_store":
            input_binding = (raw_node.get("input") or "").strip()
            output_store = (raw_node.get("output_store") or "").strip()
            output_store_base = (raw_node.get("output_store_base") or "").strip()
            if not input_binding:
                raise ValueError("missing_input")
            if output_store and output_store_base:
                raise ValueError("cannot_specify_both_output_store_and_output_store_base")
            if not output_store and not output_store_base:
                raise ValueError("missing_output_store_or_output_store_base")
            final_store = output_store or self.allocate_store_name(output_store_base, node_id)
            return {
                "input": input_binding,
                "output_store": final_store,
            }

        input_stores = [s.strip() for s in raw_node.get("input_stores", []) if isinstance(s, str) and s.strip()]
        bind = (raw_node.get("bind") or "").strip()
        output_store = (raw_node.get("output_store") or "").strip()
        output_store_base = (raw_node.get("output_store_base") or "").strip()

        if not input_stores:
            raise ValueError("missing_input_stores")
        if output_store and output_store_base:
            raise ValueError("cannot_specify_both_output_store_and_output_store_base")
        if not bind and not output_store and not output_store_base:
            raise ValueError("missing_bind_or_output_store")
        final_store = output_store or (self.allocate_store_name(output_store_base, node_id) if output_store_base else "")

        if kind in {"map", "transform"}:
            prompt = (raw_node.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("missing_prompt")
            return {
                "input_stores": input_stores,
                "bind": bind,
                "output_store": final_store,
                "prompt": prompt,
            }

        if kind == "chunk_map":
            prompt = (raw_node.get("prompt") or "").strip()
            chunk_size = raw_node.get("chunk_size")
            chunk_overlap = raw_node.get("chunk_overlap", 0)
            if not prompt:
                raise ValueError("missing_prompt")
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("invalid_chunk_size")
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                raise ValueError("invalid_chunk_overlap")
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap_must_be_lt_chunk_size")
            return {
                "input_stores": input_stores,
                "bind": bind,
                "output_store": final_store,
                "prompt": prompt,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

        if kind == "filter_store":
            prompt = (raw_node.get("prompt") or "").strip()
            chunk_size = raw_node.get("chunk_size")
            if not prompt:
                raise ValueError("missing_prompt")
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("invalid_chunk_size")
            return {
                "input_stores": input_stores,
                "bind": bind,
                "output_store": final_store,
                "prompt": prompt,
                "chunk_size": chunk_size,
            }

        if kind == "reduce":
            reduce_prompt = (raw_node.get("reduce_prompt") or "").strip()
            if not reduce_prompt:
                raise ValueError("missing_reduce_prompt")
            return {
                "input_stores": input_stores,
                "bind": bind,
                "output_store": final_store,
                "reduce_prompt": reduce_prompt,
            }

        raise ValueError(f"unsupported_kind:{kind}")


class TaskPlanner:
    def __init__(self, ctx: ExecutionContext):
        self.ctx = ctx

    def materialize(self, task: Task) -> List[Task]:
        log(f"[planner] materializing task={task.id} kind={task.kind}")
        if task.kind == "map_materialize":
            return self._materialize_transform(task)
        if task.kind == "transform_materialize":
            return self._materialize_transform(task)
        if task.kind == "chunk_map_materialize":
            return self._materialize_chunk_map(task)
        if task.kind == "filter_store_materialize":
            return self._materialize_filter_store(task)
        if task.kind == "reduce_materialize":
            return self._materialize_reduce(task)
        raise ValueError(f"Task kind is not materializable: {task.kind}")

    def _materialize_transform(self, task: Task) -> List[Task]:
        payload = task.payload
        final_bind = payload.get("bind", "")
        output_store = payload.get("output_store", "")
        prompt = payload["prompt"]

        input_docs = self.ctx.resolve_store_union(payload["input_stores"])
        created: List[Task] = []

        if not input_docs:
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="collect_or_write",
                deps={task.id},
                payload={
                    "source_task_ids": [],
                    "bind": final_bind,
                    "output_store": output_store,
                },
            ))
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
                    "prompt": prompt,
                },
            ))
            item_task_ids.append(item_task_id)

        created.append(Task(
            id=f"{task.id}__finalize",
            kind="collect_or_write",
            deps=set(item_task_ids),
            payload={
                "source_task_ids": item_task_ids,
                "bind": final_bind,
                "output_store": output_store,
            },
        ))

        log(f"[planner] transform task={task.id} created {len(created)} task(s)")
        return created

    def _materialize_chunk_map(self, task: Task) -> List[Task]:
        payload = task.payload
        final_bind = payload.get("bind", "")
        output_store = payload.get("output_store", "")
        prompt = payload["prompt"]
        chunk_size = int(payload["chunk_size"])
        chunk_overlap = int(payload.get("chunk_overlap", 0))

        input_docs = self.ctx.resolve_store_union(payload["input_stores"])
        created: List[Task] = []

        if not input_docs:
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="collect_or_write",
                deps={task.id},
                payload={
                    "source_task_ids": [],
                    "bind": final_bind,
                    "output_store": output_store,
                },
            ))
            return created

        item_task_ids: List[str] = []

        for doc_idx, doc in enumerate(input_docs):
            chunks = chunk_text_chars(doc.content, chunk_size, chunk_overlap)
            log(f"[planner] chunk_map task={task.id} doc={doc.id} chunks={len(chunks)}")

            for chunk_idx, (start, end, chunk_text) in enumerate(chunks):
                chunk_doc = Document(
                    id=f"{doc.id}.chunk{chunk_idx:03d}",
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "source_id": doc.id,
                        "chunk_index": chunk_idx,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                )

                seed_task_id = f"{task.id}__seed_{doc_idx}_{chunk_idx}"
                item_task_id = f"{task.id}__item_{doc_idx}_{chunk_idx}"

                created.append(Task(
                    id=seed_task_id,
                    kind="seed",
                    deps={task.id},
                    payload={"docs": docs_to_payload([chunk_doc])},
                ))

                created.append(Task(
                    id=item_task_id,
                    kind="agent_map_item",
                    deps={seed_task_id},
                    payload={
                        "source_task_id": seed_task_id,
                        "prompt": prompt,
                    },
                ))
                item_task_ids.append(item_task_id)

        created.append(Task(
            id=f"{task.id}__finalize",
            kind="collect_or_write",
            deps=set(item_task_ids),
            payload={
                "source_task_ids": item_task_ids,
                "bind": final_bind,
                "output_store": output_store,
            },
        ))

        log(f"[planner] chunk_map task={task.id} created {len(created)} task(s)")
        return created

    def _materialize_filter_store(self, task: Task) -> List[Task]:
        payload = task.payload
        final_bind = payload.get("bind", "")
        output_store = payload.get("output_store", "")
        prompt = payload["prompt"]
        chunk_size = int(payload["chunk_size"])

        input_docs = self.ctx.resolve_store_union(payload["input_stores"])
        created: List[Task] = []

        if not input_docs:
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="collect_or_write",
                deps={task.id},
                payload={
                    "source_task_ids": [],
                    "bind": final_bind,
                    "output_store": output_store,
                },
            ))
            return created

        item_task_ids: List[str] = []

        for idx, doc in enumerate(input_docs):
            start, end, opening_text = first_chunk_text(doc.content, chunk_size)
            opening_doc = Document(
                id=f"{doc.id}.opening",
                content=opening_text,
                metadata={
                    **doc.metadata,
                    "source_id": doc.id,
                    "chunk_index": 0,
                    "chunk_start": start,
                    "chunk_end": end,
                    "judge_mode": "opening_chunk",
                },
            )

            seed_task_id = f"{task.id}__seed_{idx}"
            item_task_id = f"{task.id}__item_{idx}"

            created.append(Task(
                id=seed_task_id,
                kind="seed",
                deps={task.id},
                payload={"docs": docs_to_payload([opening_doc])},
            ))

            created.append(Task(
                id=item_task_id,
                kind="agent_filter_item",
                deps={seed_task_id},
                payload={
                    "source_task_id": seed_task_id,
                    "original_doc": docs_to_payload([doc])[0],
                    "prompt": prompt,
                },
            ))
            item_task_ids.append(item_task_id)

        created.append(Task(
            id=f"{task.id}__finalize",
            kind="collect_or_write",
            deps=set(item_task_ids),
            payload={
                "source_task_ids": item_task_ids,
                "bind": final_bind,
                "output_store": output_store,
            },
        ))

        log(f"[planner] filter_store task={task.id} created {len(created)} task(s)")
        return created

    def _materialize_reduce(self, task: Task) -> List[Task]:
        payload = task.payload
        final_bind = payload.get("bind", "")
        output_store = payload.get("output_store", "")
        reduce_prompt = payload["reduce_prompt"]

        input_docs = self.ctx.resolve_store_union(payload["input_stores"])
        created: List[Task] = []

        if not input_docs:
            created.append(Task(
                id=f"{task.id}__finalize",
                kind="collect_or_write",
                deps={task.id},
                payload={
                    "source_task_ids": [],
                    "bind": final_bind,
                    "output_store": output_store,
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

                merge_id = f"{task.id}__merge_r{round_idx}_n{pair_idx // 2}"
                created.append(Task(
                    id=merge_id,
                    kind="agent_pair_merge",
                    deps=set(pair),
                    payload={
                        "left_task_id": pair[0],
                        "right_task_id": pair[1],
                        "prompt": reduce_prompt,
                    },
                ))
                next_layer.append(merge_id)

            current_layer = next_layer
            round_idx += 1

        created.append(Task(
            id=f"{task.id}__finalize",
            kind="alias_or_write",
            deps={current_layer[0]},
            payload={
                "source_task_id": current_layer[0],
                "bind": final_bind,
                "output_store": output_store,
            },
        ))

        log(f"[planner] reduce task={task.id} created {len(created)} task(s)")
        return created


class TaskExecutor:
    def __init__(self, ctx: ExecutionContext):
        self.ctx = ctx

    def execute(self, task: Task) -> None:
        self.ctx.set_current_task(task.id)
        try:
            kind = task.kind
            payload = task.payload
            log(f"[exec] task={task.id} kind={kind} payload_keys={list(payload.keys())}")

            if kind == "seed":
                docs = docs_from_payload(payload["docs"])
                self.ctx.publish_task_output(task.id, docs)
                return

            if kind == "write_store":
                docs = self.ctx.resolve_binding(payload["input"])
                store = self.ctx.get_store(payload["output_store"])
                for doc in docs:
                    store.write(doc)
                self.ctx.publish_task_output(task.id, [])
                return

            if kind == "agent_map_item":
                docs = self.ctx.resolve_task_output(payload["source_task_id"])
                result = self.ctx.agent.run(
                    docs=docs,
                    prompt=payload["prompt"],
                    task_id=task.id,
                    tools=self.ctx.tools(),
                )
                self.ctx.publish_task_output(task.id, result)
                return

            if kind == "agent_filter_item":
                opening_docs = self.ctx.resolve_task_output(payload["source_task_id"])
                decision_docs = self.ctx.agent.run(
                    docs=opening_docs,
                    prompt=payload["prompt"],
                    task_id=task.id,
                    tools=self.ctx.tools(),
                )
                decision_text = decision_docs[0].content if decision_docs else ""
                keep = parse_keep_drop(decision_text)
                original_doc = Document(
                    id=payload["original_doc"]["id"],
                    content=payload["original_doc"]["content"],
                    metadata=payload["original_doc"].get("metadata", {}),
                )

                if keep:
                    log(f"[exec] task={task.id} decision=KEEP source={original_doc.id}")
                    self.ctx.publish_task_output(task.id, [original_doc])
                else:
                    log(f"[exec] task={task.id} decision=DROP source={original_doc.id} decision_text={decision_text[:160]!r}")
                    self.ctx.publish_task_output(task.id, [])
                return

            if kind == "agent_pair_merge":
                left_docs = self.ctx.resolve_task_output(payload["left_task_id"])
                right_docs = self.ctx.resolve_task_output(payload["right_task_id"])
                result = self.ctx.agent.run(
                    docs=left_docs + right_docs,
                    prompt=payload["prompt"],
                    task_id=task.id,
                    tools=self.ctx.tools(),
                )
                self.ctx.publish_task_output(task.id, result)
                return

            if kind == "collect_or_write":
                combined: List[Document] = []
                for source_task_id in payload["source_task_ids"]:
                    combined.extend(self.ctx.resolve_task_output(source_task_id))

                if payload.get("bind"):
                    self.ctx.publish_task_output(task.id, combined, bind=payload["bind"])
                else:
                    self.ctx.publish_task_output(task.id, combined)

                if payload.get("output_store"):
                    store = self.ctx.get_store(payload["output_store"])
                    for doc in combined:
                        store.write(doc)
                return

            if kind == "alias_or_write":
                docs = self.ctx.resolve_task_output(payload["source_task_id"])
                if payload.get("bind"):
                    self.ctx.publish_task_output(task.id, docs, bind=payload["bind"])
                else:
                    self.ctx.publish_task_output(task.id, docs)
                if payload.get("output_store"):
                    store = self.ctx.get_store(payload["output_store"])
                    for doc in docs:
                        store.write(doc)
                return

            if kind in {"map_materialize", "transform_materialize", "chunk_map_materialize", "reduce_materialize", "filter_store_materialize"}:
                self.ctx.publish_task_output(task.id, [])
                return

            raise ValueError(f"Unknown task kind: {kind}")
        finally:
            self.ctx.set_current_task(None)


class Scheduler:
    def __init__(self, executor: TaskExecutor, planner: TaskPlanner, trace: Trace, ctx: ExecutionContext):
        self.executor = executor
        self.planner = planner
        self.trace = trace
        self.ctx = ctx

        self.tasks: Dict[str, Task] = {}
        self.state: Dict[str, str] = {}
        self.remaining_deps: Dict[str, int] = {}
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self.ready: deque[str] = deque()
        self.dynamic_tasks: Set[str] = set()
        self.task_provenance: Dict[str, str] = {}

    def add_task(self, task: Task, is_dynamic: bool = False, proposed_by: Optional[str] = None) -> None:
        if task.id in self.tasks:
            raise ValueError(f"Duplicate task id: {task.id}")

        unresolved = [dep for dep in task.deps if dep not in self.state]
        if unresolved:
            raise ValueError(f"Task {task.id} has unknown dependencies: {unresolved}")

        self.tasks[task.id] = task
        self.state[task.id] = "pending"

        if is_dynamic:
            self.dynamic_tasks.add(task.id)
        if proposed_by is not None:
            self.task_provenance[task.id] = proposed_by

        remaining = 0
        for dep in task.deps:
            self.dependents[dep].add(task.id)
            if self.state[dep] != "done":
                remaining += 1

        self.remaining_deps[task.id] = remaining
        if remaining == 0:
            self.ready.append(task.id)

        log(f"[sched] add task={task.id} kind={task.kind} deps={sorted(task.deps)} remaining={remaining} dynamic={is_dynamic}")

    def add_tasks(self, tasks: List[Task], is_dynamic: bool = False, proposed_by: Optional[str] = None) -> None:
        log(f"[sched] adding batch of {len(tasks)} task(s)")
        for task in tasks:
            self.add_task(task, is_dynamic=is_dynamic, proposed_by=proposed_by)

    def admit_proposals(self, proposals: List[NodeProposal]) -> None:
        if not proposals:
            return

        log(f"[sched] admitting {len(proposals)} node proposal(s)")
        for proposal in proposals:
            if proposal.id in self.tasks:
                log(f"[sched] rejecting proposed node={proposal.id} reason=duplicate_task_id")
                continue

            unresolved = [dep for dep in proposal.deps if dep not in self.state]
            if unresolved:
                log(f"[sched] rejecting proposed node={proposal.id} reason=unknown_dependencies unresolved={unresolved}")
                continue

            admitted_kind = proposal.kind
            if admitted_kind == "map":
                admitted_kind = "map_materialize"
            elif admitted_kind == "transform":
                admitted_kind = "transform_materialize"
            elif admitted_kind == "chunk_map":
                admitted_kind = "chunk_map_materialize"
            elif admitted_kind == "filter_store":
                admitted_kind = "filter_store_materialize"
            elif admitted_kind == "reduce":
                admitted_kind = "reduce_materialize"

            task = Task(
                id=proposal.id,
                kind=admitted_kind,
                deps=set(proposal.deps),
                payload=dict(proposal.payload),
            )
            self.add_task(task, is_dynamic=True, proposed_by=proposal.proposed_by)
            log(f"[sched] admitted dynamic node={task.id} kind={task.kind} proposed_by={proposal.proposed_by}")

    def _drain_proposals(self) -> None:
        proposals = self.ctx.consume_node_proposals()
        self.admit_proposals(proposals)

    def replace_dependency(self, old_dep: str, new_dep: str, restrict_to: Optional[Set[str]] = None) -> None:
        affected = set(self.dependents.get(old_dep, set()))
        if restrict_to is not None:
            affected &= restrict_to

        for child_id in sorted(affected):
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

    def mark_done(self, task_id: str) -> None:
        self.state[task_id] = "done"
        self.trace.log_task_state(task_id, "done", {"kind": self.tasks[task_id].kind})

        for child in sorted(self.dependents.get(task_id, [])):
            self.remaining_deps[child] -= 1
            if self.remaining_deps[child] == 0 and self.state[child] == "pending":
                self.ready.append(child)

    def _run_materializer(self, task: Task) -> None:
        finalize_task_id = f"{task.id}__finalize"
        prior_dependents = set(self.dependents.get(task.id, set()))
        new_tasks = self.planner.materialize(task)
        self.executor.execute(task)
        self.mark_done(task.id)
        self.add_tasks(new_tasks)
        if finalize_task_id in self.tasks:
            self.replace_dependency(task.id, finalize_task_id, restrict_to=prior_dependents)
        self._drain_proposals()

    def _run_concrete(self, task: Task) -> None:
        self.executor.execute(task)
        self.mark_done(task.id)
        self._drain_proposals()

    def run(self) -> None:
        while self.ready:
            task_id = self.ready.popleft()

            if self.state[task_id] != "pending":
                continue
            if self.remaining_deps[task_id] != 0:
                continue

            task = self.tasks[task_id]
            self.state[task_id] = "running"
            self.trace.log_task_state(task_id, "running", {"kind": task.kind})

            if task.kind in {"map_materialize", "transform_materialize", "chunk_map_materialize", "reduce_materialize", "filter_store_materialize"}:
                self._run_materializer(task)
            else:
                self._run_concrete(task)


class WorkflowCompiler:
    def __init__(self, wf: dict[str, Any]):
        self.nodes = wf.get("nodes", [])

    def compile(self) -> List[Task]:
        tasks: List[Task] = []
        log(f"[compile] compiling {len(self.nodes)} node(s)")

        for node in self.nodes:
            deps = set(node.get("deps", []))
            kind = node["kind"]

            if kind == "write_store":
                tasks.append(Task(node["id"], "write_store", deps, node))
            elif kind == "map":
                tasks.append(Task(node["id"], "map_materialize", deps, node))
            elif kind == "transform":
                tasks.append(Task(node["id"], "transform_materialize", deps, node))
            elif kind == "chunk_map":
                tasks.append(Task(node["id"], "chunk_map_materialize", deps, node))
            elif kind == "filter_store":
                tasks.append(Task(node["id"], "filter_store_materialize", deps, node))
            elif kind == "reduce":
                tasks.append(Task(node["id"], "reduce_materialize", deps, node))
            else:
                raise ValueError(f"Unknown node kind: {kind}")

            log(f"[compile] task={tasks[-1].id} kind={tasks[-1].kind} deps={sorted(tasks[-1].deps)}")

        log(f"[compile] produced {len(tasks)} initial task(s)")
        return tasks


def main() -> None:
    cfg = json.loads(Path(".orchestra.json").read_text())
    wf = json.loads(Path(cfg["workflow_path"]).read_text())

    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log(f"[main] run_id={run_id}")
    log(f"[main] DDGS source={DDGS_SOURCE}")

    trace = Trace(Path(cfg["traces_root"]), run_id)
    ctx = ExecutionContext(cfg, trace)
    planner = TaskPlanner(ctx)
    executor = TaskExecutor(ctx)
    scheduler = Scheduler(executor, planner, trace, ctx)

    tasks = WorkflowCompiler(wf).compile()
    scheduler.add_tasks(tasks)
    scheduler.run()

    print(f"DONE: {run_id}")


if __name__ == "__main__":
    main()

