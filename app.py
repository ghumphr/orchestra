# ORCHESTRA v3.9 - Communication Infrastructure & Pattern Kernel

"""
ORCHESTRA ARCHITECTURAL DESIGN INSIGHTS:

1. ORCHESTRATION BY PATTERN (THE KERNEL):
   The Runner is the "Project Manager." It interprets the workflow.json and 
   executes hard-coded patterns (Map, Tree-Reduce, Read/Write). This ensures 
   the process is deterministic. The LLM is never in charge of the 10,000-foot 
   view; it only handles the task immediately in front of it.

2. STORES AS THE COMMUNICATION BUS:
   Agents are strictly decoupled. They do not pass messages to each other 
   directly. Instead, Agent A produces Documents -> Runner writes to Store X 
   -> Runner reads from Store X -> Agent B processes. This "Air-Gapped" 
   communication allows for easy debugging and human intervention.

3. AGENT AS PURE LABORER:
   The Agent is a "Reasoning Factory." It consumes input Documents and tools 
   to produce output Documents. It has no knowledge of whether it is the 
   first or last step in a chain. 

4. TEMPORAL GROUNDING (THE 2024 BIAS):
   LLMs are "frozen in time" based on their training data (often 2024). 
   To solve this, the 'get_context' tool provides the "Project Reality" 
   (April 2026). This allows the agent to recognize that its internal 
   knowledge is "Historical" and search results are "Current."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict, Set, Optional
from datetime import datetime, timezone
import json, subprocess, random, string, heapq, re

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI

# --------------------------
# Models
# --------------------------

@dataclass
class Document:
    """
    The fundamental unit of information exchange.
    Design Insight: Separating raw content from metadata allows the 
    framework to track 'updated_at' and 'id' without polluting the 
    text the LLM needs to summarize.
    """
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

# --------------------------
# Store (Communication Hub)
# --------------------------

class Store:
    """
    A managed repository for inter-agent handoffs.
    Design Insight: Uses 'Sidecar Metadata' (.filename.json) to store 
    system state. This keeps the primary files raw and human-readable.
    """
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, id: str) -> Document:
        txt = (self.root / id).read_text()
        m = self.root / f".{id}.json"
        return Document(id, txt, json.loads(m.read_text()) if m.exists() else {"id": id})

    def write(self, doc: Document):
        """Persists the document and updates its chronological metadata."""
        (self.root / doc.id).write_text(doc.content)
        meta = {**doc.metadata, "id": doc.id, "updated_at": datetime.now(timezone.utc).isoformat()}
        (self.root / f".{doc.id}.json").write_text(json.dumps(meta, indent=2))

    def all(self) -> List[Document]:
        return [self.read(p.name) for p in self.root.iterdir() 
                if p.is_file() and not p.name.startswith(".")]

# --------------------------
# Agent (The Data Processor)
# --------------------------

class Agent:
    """
    Stateless reasoning unit. 
    Design Insight: Uses a 'Document Factory' approach. It can return multiple 
    documents using a specific syntax, which the framework then dispatches.
    """
    def __init__(self, cfg: dict, trace: Trace):
        self.client = OpenAI(base_url=cfg["openai_base_url"], api_key="dummy")
        self.cfg, self.trace = cfg, trace

    def run(self, docs: List[Document], conf: dict, tid: str, tools: dict) -> List[Document]:
        sys, user = conf["system"], "\n\n".join(f"## {d.id}\n{d.content}" for d in docs)
        self.trace.log_call(tid, sys, user)
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        
        for _ in range(12):
            resp = self.client.chat.completions.create(
                model=self.cfg["default_model"], messages=msgs,
                tools=[{"type": "function", "function": {"name": k}} for k in tools]
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                msgs.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    res = tools[tc.function.name](args)
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)})
                continue
            
            # Logic: Multi-document parsing (The 'Gatherer' pattern)
            content = msg.content or ""
            doc_pattern = r"---DOC:([\w\.-]+)---\n(.*?)(?=\n---DOC:|$)"
            found = re.findall(doc_pattern, content, re.DOTALL)
            
            if found:
                return [Document(name.strip(), body.strip()) for name, body in found]
            
            return [Document(conf.get("output_id", f"{tid}.md"), content)]
        return []

# --------------------------
# Runner (The Pattern Kernel)
# --------------------------

class Runner:
    """
    The Orchestration Engine. 
    Responsible for:
    1. Managing the Blackboard (in-memory document state).
    2. Executing Patterns (Tree-Reduce, Map, etc.).
    3. Ensuring the Communication Stores exist.
    """
    def __init__(self, store_root: Path, cfg: dict, trace: Trace):
        self.store_root = store_root
        self.cfg = cfg
        self.trace = trace
        self.blackboard: Dict[str, List[Document]] = {}
        self.agent = Agent(cfg, trace)
        
        # Initialize existing stores
        self.stores: Dict[str, Store] = {
            d.name: Store(d) for d in store_root.iterdir() if d.is_dir()
        }

    def get_store(self, name: str) -> Store:
        """Retrieves or dynamically initializes a communication store."""
        if name not in self.stores:
            self.stores[name] = Store(self.store_root / name)
        return self.stores[name]

    def run(self, wf: dict):
        for s in wf.get("steps", []):
            stype = s["type"]
            
            # Tools injected into the Agent's reasoning loop
            tools = {
                "get_context": lambda a: json.dumps({
                    "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "project": self.cfg.get("project_name", "Orchestra")
                }),
                "duckduckgo_search": lambda a: json.dumps(list(DDGS().text(a.get("query") or a.get("q", ""), max_results=5))),
                "fetch_url": lambda a: BeautifulSoup(httpx.get(a.get("url","")).text, "html.parser").get_text()[:8000]
            }

            if stype == "read_documents":
                self.blackboard[s["output"]] = self.get_store(s["store"]).all()

            elif stype == "run_agent":
                inputs = self.blackboard.get(s["input"], [])
                self.blackboard[s["output"]] = self.agent.run(inputs, s["agent_config"], s["id"], tools)

            elif stype == "tree_reduce":
                """PATTERN: Recursive Pairwise Merging."""
                layer = self.blackboard.get(s["input"], [])
                depth = 0
                while len(layer) > 1:
                    next_layer = []
                    for i in range(0, len(layer), 2):
                        pair = layer[i:i+2]
                        tid = f"{s['id']}_d{depth}_p{i//2}"
                        next_layer.extend(self.agent.run(pair, s["agent_config"], tid, tools))
                    layer = next_layer
                    depth += 1
                self.blackboard[s["output"]] = layer

            elif stype == "write_documents":
                # Framework-led handoff to communication store
                store = self.get_store(s["store"])
                for d in self.blackboard.get(s["input"], []):
                    store.write(d)

# --------------------------
# Main
# --------------------------

class Trace:
    def __init__(self, root: Path, rid: str):
        self.root = root / rid
        (self.root / "calls").mkdir(parents=True, exist_ok=True)
    def log_call(self, tid, sys, user):
        d = self.root / "calls" / tid
        d.mkdir(parents=True, exist_ok=True)
        (d / "system.txt").write_text(sys)
        (d / "user.txt").write_text(user)

def main():
    cfg = json.loads(Path(".orchestra.json").read_text())
    rid = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace = Trace(Path(cfg["traces_root"]), rid)
    store_root = Path(cfg["store_root"])
    
    wf = json.loads(Path(cfg["workflow_path"]).read_text())
    
    # Initialize the Runner with the physical store root
    Runner(store_root, cfg, trace).run(wf)
    print(f"DONE: {rid}")

if __name__ == "__main__":
    main()
