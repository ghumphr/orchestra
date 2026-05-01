from __future__ import annotations

import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup
from jinja2 import Environment, BaseLoader, StrictUndefined
from openai import OpenAI

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# --- LOGGING ---

def log(msg: str, color: str = "0") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\033[{color}m[{ts}] {msg}\033[0m", flush=True)

def log_block(title: str, content: str, color: str = "0") -> None:
    border = "=" * 80
    log(border, color)
    log(f"--- {title} ---", color)
    print(content)
    log(border, color)

# --- UTILS ---

def sanitize_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"

def chunk_text_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int, str]]:
    if chunk_size <= 0: raise ValueError("chunk_size must be > 0")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((start, end, text[start:end]))
        if end >= n: break
        start = end - chunk_overlap
    return chunks

def first_chunk_text(text: str, chunk_size: int) -> Tuple[int, int, str]:
    chunks = chunk_text_chars(text, chunk_size, 0)
    return chunks[0] if chunks else (0, 0, "")

def parse_keep_drop(text: str) -> bool:
    for line in text.splitlines():
        s = line.strip().upper()
        if not s: continue
        if s.startswith("KEEP"): return True
        if s.startswith("DROP"): return False
        break
    return False

def parse_tournament_winner(text: str) -> Optional[int]:
    for line in text.splitlines():
        s = line.strip().upper()
        if not s: continue
        if "FIRST WINS" in s: return 0
        if "SECOND WINS" in s: return 1
        break
    return None

# --- DATA MODELS ---

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

# --- TOOL DEFINITIONS ---

def get_tool_definitions() -> List[dict[str, Any]]:
    """Strictly specified JSON schemas for the agent's toolset."""
    return [
        {
            "type": "function",
            "function": {
                "name": "duckduckgo_search",
                "description": "Performs a live web search for a specific query to find relevant links and sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string (e.g., 'latest solid state battery news 2026')."
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Retrieves the raw text content from a specific web URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to fetch (must start with http/https)."
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_memo",
                "description": "Writes a new individual document to the current output store. Use this to submit found links or specific findings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "A short, descriptive title for the memo. This will be sanitized into a filename."
                        },
                        "content": {
                            "type": "string",
                            "description": "The full text content of the memo. For links, include the URL here."
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional key-value pairs of metadata (e.g. source_id, author).",
                            "additionalProperties": True
                        }
                    },
                    "required": ["title", "content"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_nodes",
                "description": "Proposes new nodes to be added to the workflow dependency graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique ID for the new node."},
                                    "kind": {
                                        "type": "string", 
                                        "enum": ["map", "transform", "filter_store", "chunk_map", "reduce", "tournament", "write_store"],
                                        "description": "The pattern type of the node."
                                    },
                                    "input_stores": {
                                        "type": "array", 
                                        "items": {"type": "string"},
                                        "description": "Names of stores this node should read from."
                                    },
                                    "output_store_base": {
                                        "type": "string",
                                        "description": "Base name for the output store (the framework will ensure uniqueness)."
                                    },
                                    "prompt": {
                                        "type": "string",
                                        "description": "The Jinja2 prompt template for the node's agents."
                                    }
                                },
                                "required": ["id", "kind"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["nodes"],
                    "additionalProperties": False
                }
            }
        }
    ]

# --- AGENT ENGINE ---

class Agent:
    def __init__(self, cfg: dict[str, Any], trace_root: Path, run_id: str):
        self.client = OpenAI(base_url=cfg["openai_base_url"], api_key="dummy")
        self.cfg = cfg
        self.trace_path = trace_root / run_id / "calls"
        self.trace_path.mkdir(parents=True, exist_ok=True)
        self.jinja = Environment(loader=BaseLoader(), undefined=StrictUndefined)

    def run(self, docs: List[Document], prompt_template: str, task_id: str, tools: dict[str, Callable], is_chat: bool = False) -> List[Document]:
        ctx = {"docs": docs, "config": self.cfg, "now": datetime.now(timezone.utc).isoformat(), "task_id": task_id}
        rendered_instructions = self.jinja.from_string(prompt_template).render(**ctx)
        
        # Build Manifest from Strict Specs
        tool_manifest = ""
        if tools:
            tool_manifest = "\n\n### AVAILABLE TOOLS\nYou have access to tools. Use them by providing parameters in JSON format according to their schema:\n"
            for spec in get_tool_definitions():
                if spec["function"]["name"] in tools:
                    tool_manifest += f"- {spec['function']['name']}: {spec['function']['description']}\n"
                    tool_manifest += f"  Parameters: {json.dumps(spec['function']['parameters'], indent=2)}\n"

        full_system_prompt = rendered_instructions + tool_manifest
        t_dir = self.trace_path / task_id
        t_dir.mkdir(parents=True, exist_ok=True)
        (t_dir / "system.txt").write_text(full_system_prompt)

        messages = [{"role": "system", "content": full_system_prompt}]
        if not is_chat:
            messages.append({"role": "user", "content": "Begin task. Use tools as needed."})

        for turn in range(50):
            # Safe Serialization for logging
            log_msgs = [m if isinstance(m, dict) else m for m in messages]
            log_block(f"REQUEST (Task: {task_id}, Turn: {turn})", json.dumps(log_msgs, indent=2, default=lambda o: o.model_dump() if hasattr(o, 'model_dump') else str(o)), "35")
            
            resp = self.client.chat.completions.create(
                model=self.cfg["default_model"], 
                messages=messages, 
                tools=get_tool_definitions() if tools else None
            )
            msg_obj = resp.choices[0].message
            
            if msg_obj.tool_calls:
                messages.append(msg_obj)
                for tc in msg_obj.tool_calls:
                    fn, args = tc.function.name, json.loads(tc.function.arguments or "{}")
                    log_block(f"CALLING TOOL: {fn}", json.dumps(args, indent=2), "33")
                    res = tools[fn](args) if fn in tools else "Error: Tool not found"
                    log_block(f"TOOL RESULT", str(res), "32")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)})
                continue
            
            content = msg_obj.content or ""
            log_block(f"FINAL RESPONSE (Task: {task_id})", content, "36")
            return [Document(id=f"{task_id}.md", content=content)]
        return []

# --- ORCHESTRATION ---

class ExecutionContext:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.agent = Agent(cfg, Path(cfg["traces_root"]), self.run_id)
        self.store_root = Path(cfg["store_root"])
        self.stores: Dict[str, Store] = {}
        self.bindings: Dict[str, List[Document]] = {}
        self.task_outputs: Dict[str, List[Document]] = {}
        self.current_task_id: Optional[str] = None
        self.active_output_store: Optional[str] = None
        self.pending_proposals: List[NodeProposal] = []
        self.allocated_names: Set[str] = set()

    def get_store(self, name: str) -> Store:
        if name not in self.stores:
            self.stores[name] = Store(self.store_root / name)
        return self.stores[name]

    def allocate_store_name(self, base: str, task_id: str) -> str:
        cand = f"{sanitize_name(base)}__{sanitize_name(task_id)}"
        i = 1
        while cand in self.allocated_names or (self.store_root / cand).exists():
            cand = f"{sanitize_name(base)}__{sanitize_name(task_id)}__{i}"; i += 1
        self.allocated_names.add(cand)
        return cand

    def publish_task_output(self, task_id: str, docs: List[Document], bind: Optional[str] = None) -> None:
        self.task_outputs[task_id] = list(docs)
        if bind: self.bindings[bind] = list(docs)

    def tools(self) -> dict[str, Callable]:
        def ddg(args):
            try:
                with DDGS() as d: return json.dumps(list(d.text(args["query"], max_results=8)))
            except Exception as e: return str(e)
        def fetch(args):
            try:
                r = httpx.get(args["url"], timeout=15)
                return BeautifulSoup(r.text, "html.parser").get_text()[:8000]
            except Exception as e: return str(e)
        def schedule(args):
            for n in args.get("nodes", []):
                kind, nid = n["kind"], n["id"]
                out_base = n.get("output_store_base")
                out_s = n.get("output_store") or (self.allocate_store_name(out_base, nid) if out_base else "")
                payload = {**n, "output_store": out_s}
                deps = {self.current_task_id} | set(n.get("deps", []))
                self.pending_proposals.append(NodeProposal(nid, kind, payload, deps, self.current_task_id))
            return "Scheduled."
        def send_memo(args):
            if not self.active_output_store: return "Error: No output store active for this task."
            title, content = args.get("title", "memo"), args.get("content", "")
            doc_id = sanitize_name(title) + ".md"
            doc = Document(id=doc_id, content=content, metadata=args.get("metadata", {}))
            self.get_store(self.active_output_store).write(doc)
            return f"Memo '{doc_id}' successfully saved to {self.active_output_store}."
            
        return {
            "duckduckgo_search": ddg, 
            "fetch_url": fetch, 
            "schedule_nodes": schedule,
            "send_memo": send_memo
        }

class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, doc: Document):
        (self.root / doc.id).write_text(doc.content)
        m = {**doc.metadata, "id": doc.id, "ts": datetime.now(timezone.utc).isoformat()}
        (self.root / f".{doc.id}.json").write_text(json.dumps(m, indent=2))

    def all(self) -> List[Document]:
        docs = []
        if not self.root.exists(): return []
        for p in sorted(self.root.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                txt = p.read_text()
                mp = self.root / f".{p.name}.json"
                m = json.loads(mp.read_text()) if mp.exists() else {}
                docs.append(Document(p.name, txt, m))
        return docs

class TaskPlanner:
    def __init__(self, ctx: ExecutionContext): self.ctx = ctx

    def materialize(self, task: Task) -> List[Task]:
        p, kind = task.payload, task.kind.replace("_materialize", "")
        docs = []
        for sn in p.get("input_stores", []): docs.extend(self.ctx.get_store(sn).all())
        created, item_ids = [], []

        if kind in ("map", "transform", "filter_store", "chunk_map"):
            for idx, doc in enumerate(docs):
                units = [doc]
                if kind == "filter_store":
                    _, _, txt = first_chunk_text(doc.content, p["chunk_size"])
                    units = [Document(f"{doc.id}.op", txt, doc.metadata)]
                elif kind == "chunk_map":
                    chunks = chunk_text_chars(doc.content, p["chunk_size"], p.get("chunk_overlap", 0))
                    units = [Document(f"{doc.id}.c{i}", t, {**doc.metadata, "source_id": doc.id}) for i, (_,_,t) in enumerate(chunks)]

                for w_idx, w_doc in enumerate(units):
                    sid, iid = f"{task.id}__s{idx}_{w_idx}", f"{task.id}__i{idx}_{w_idx}"
                    created.append(Task(sid, "seed", set(), {"docs": docs_to_payload([w_doc])}))
                    ikind = "agent_filter_item" if kind == "filter_store" else "agent_map_item"
                    ipay = {"source_task_id": sid, "prompt": p["prompt"], "output_store": p.get("output_store")}
                    if kind == "filter_store": ipay["original_doc"] = docs_to_payload([doc])[0]
                    created.append(Task(iid, ikind, {sid}, ipay))
                    item_ids.append(iid)
            created.append(Task(f"{task.id}__finalize", "collect_or_write", set(item_ids), 
                                {"source_task_ids": item_ids, "bind": p.get("bind"), "output_store": p.get("output_store")}))

        elif kind in ("reduce", "tournament"):
            layer = []
            for idx, doc in enumerate(docs):
                sid = f"{task.id}__s{idx}"; created.append(Task(sid, "seed", set(), {"docs": docs_to_payload([doc])}))
                layer.append(sid)
            rnd = 0
            while len(layer) > 1:
                next_l = []
                for i in range(0, len(layer), 2):
                    pair = layer[i:i+2]
                    if len(pair) == 1: next_l.append(pair[0]); continue
                    mid = f"{task.id}__m_r{rnd}_n{i//2}"
                    mkind = "agent_pair_merge" if kind == "reduce" else "agent_pair_tournament"
                    created.append(Task(mid, mkind, set(pair), {"left_task_id": pair[0], "right_task_id": pair[1], "prompt": p.get("prompt") or p.get("reduce_prompt"), "output_store": p.get("output_store")}))
                    next_l.append(mid)
                layer, rnd = next_l, rnd + 1
            if layer: 
                created.append(Task(f"{task.id}__finalize", "alias_or_write", {layer[0]}, 
                                          {"source_task_id": layer[0], "bind": p.get("bind"), "output_store": p.get("output_store")}))
        return created

class Scheduler:
    def __init__(self, cfg: dict[str, Any]):
        self.ctx = ExecutionContext(cfg)
        self.planner = TaskPlanner(self.ctx)
        self.tasks: Dict[str, Task] = {}
        self.state: Dict[str, str] = {}
        self.remaining_deps: Dict[str, int] = {}
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self.ready: deque[str] = deque()

    def add_task(self, task: Task):
        self.tasks[task.id], self.state[task.id] = task, "pending"
        rd = 0
        for d in task.deps:
            self.dependents[d].add(task.id)
            if self.state.get(d) != "done": rd += 1
        self.remaining_deps[task.id] = rd
        if rd == 0: self.ready.append(task.id)

    def execute_task(self, task: Task):
        self.ctx.current_task_id = task.id
        p = task.payload
        self.ctx.active_output_store = p.get("output_store")
        
        if task.kind == "seed":
            self.ctx.publish_task_output(task.id, docs_from_payload(p["docs"]))
        elif task.kind == "agent_map_item":
            docs = self.ctx.task_outputs[p["source_task_id"]]
            self.ctx.publish_task_output(task.id, self.ctx.agent.run(docs, p["prompt"], task.id, self.ctx.tools()))
        elif task.kind == "agent_filter_item":
            res = self.ctx.agent.run(self.ctx.task_outputs[p["source_task_id"]], p["prompt"], task.id, self.ctx.tools())
            if parse_keep_drop(res[0].content if res else ""):
                self.ctx.publish_task_output(task.id, docs_from_payload([p["original_doc"]]))
            else: self.ctx.publish_task_output(task.id, [])
        elif task.kind == "agent_pair_merge":
            docs = self.ctx.task_outputs[p["left_task_id"]] + self.ctx.task_outputs[p["right_task_id"]]
            self.ctx.publish_task_output(task.id, self.ctx.agent.run(docs, p["prompt"], task.id, self.ctx.tools()))
        elif task.kind == "agent_pair_tournament":
            l, r = self.ctx.task_outputs[p["left_task_id"]], self.ctx.task_outputs[p["right_task_id"]]
            jp = f"{p['prompt']}\n\nRespond 'FIRST WINS' or 'SECOND WINS' on the first line."
            res = self.ctx.agent.run([Document("Doc 1", l[0].content), Document("Doc 2", r[0].content)], jp, task.id, {})
            idx = parse_tournament_winner(res[0].content if res else "")
            self.ctx.publish_task_output(task.id, [r[0] if idx == 1 else l[0]])
        elif task.kind == "collect_or_write":
            all_d = []
            for tid in p["source_task_ids"]: all_d.extend(self.ctx.task_outputs.get(tid, []))
            self.ctx.publish_task_output(task.id, all_d, p.get("bind"))
            if p.get("output_store"):
                s = self.ctx.get_store(p["output_store"])
                for d in all_d: s.write(d)
        elif task.kind == "alias_or_write":
            docs = self.ctx.task_outputs.get(p["source_task_id"], [])
            self.ctx.publish_task_output(task.id, docs, p.get("bind"))
            if p.get("output_store"):
                s = self.ctx.get_store(p["output_store"])
                for d in docs: s.write(d)
        elif task.kind == "write_store":
            docs = self.ctx.bindings.get(p["input"], [])
            s = self.ctx.get_store(p["output_store"])
            for d in docs: s.write(d)
            self.ctx.publish_task_output(task.id, [])
            
        self.ctx.current_task_id = None
        self.ctx.active_output_store = None

    def run(self):
        while self.ready:
            tid = self.ready.popleft()
            if self.state[tid] != "pending": continue
            task = self.tasks[tid]
            log(f"EXECUTING: {tid} ({task.kind})", "1")
            
            if task.kind.endswith("_materialize"):
                new_tasks = self.planner.materialize(task)
                self.state[tid] = "done"
                for nt in new_tasks: self.add_task(nt)
                
                fid = f"{tid}__finalize"
                if fid in self.tasks:
                    for cid in list(self.dependents[tid]):
                        if cid == fid: continue
                        child = self.tasks[cid]
                        if tid in child.deps:
                            child.deps.remove(tid); child.deps.add(fid)
                            self.dependents[fid].add(cid)
                            if self.state.get(fid) != "done": self.remaining_deps[cid] += 1
                
                for cid in sorted(self.dependents[tid]):
                    if self.state[cid] == "pending":
                        self.remaining_deps[cid] -= 1
                        if self.remaining_deps[cid] == 0: self.ready.append(cid)
            else:
                self.execute_task(task)
                self.state[tid] = "done"
                for cid in sorted(self.dependents[tid]):
                    self.remaining_deps[cid] -= 1
                    if self.remaining_deps[cid] == 0: self.ready.append(cid)
            
            while self.ctx.pending_proposals:
                p = self.ctx.pending_proposals.pop(0)
                kind = p.kind + "_materialize" if p.kind != "write_store" else "write_store"
                self.add_task(Task(p.id, kind, p.deps, p.payload))

def main():
    cfg = json.loads(Path(".orchestra.json").read_text())
    # tool_specs are now derived from the central function in Agent.run
    wf = json.loads(Path(cfg["workflow_path"]).read_text())
    sched = Scheduler(cfg)
    for n in wf.get("nodes", []):
        k = n["kind"] + "_materialize" if n["kind"] != "write_store" else "write_store"
        sched.add_task(Task(n["id"], k, set(n.get("deps", [])), n))
    sched.run()

if __name__ == "__main__": main()
