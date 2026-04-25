# ORCHESTRA v1.2 (map_agent + tools + traces)

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import json, subprocess, random, string

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI

# --------------------------
# Document
# --------------------------

@dataclass
class Document:
    """
    The fundamental unit of data within the framework. 
    
    Design Insight: By separating 'content' from 'metadata', the framework 
    allows LLMs to process the raw text while the system tracks lineage, 
    IDs, and timestamps independently.
    """
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

# --------------------------
# Store (sidecar metadata)
# --------------------------

class Store:
    """
    A file-system based document repository using a 'sidecar' pattern.
    
    Design Decision: Metadata is stored in hidden files (e.g., .myfile.txt.json).
    This allows the primary content to remain as raw, human-readable files 
    (like .md or .txt) that can be easily edited or inspected by external 
    tools, while still persisting system-level data.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta(self, id): 
        """Returns the path to the hidden sidecar metadata file."""
        return self.root / f".{id}.json"
    
    def _file(self, id): 
        """Returns the path to the actual content file."""
        return self.root / id

    def list_ids(self):
        """Lists all non-hidden files in the store root."""
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_file() and not p.name.startswith("."))

    def read(self, id) -> Document:
        """Loads a document and its associated metadata from disk."""
        txt = self._file(id).read_text()
        m = self._meta(id)
        meta = json.loads(m.read_text()) if m.exists() else {"id": id}
        return Document(id, txt, meta)

    def write(self, doc: Document):
        """
        Persists a document and updates metadata.
        
        Design Insight: It automatically manages 'created_at' and 'updated_at' 
        to provide a basic audit trail for data flowing through the pipeline.
        """
        now = datetime.now(timezone.utc).isoformat()
        meta = dict(doc.metadata)
        meta["id"] = doc.id

        mpath = self._meta(doc.id)
        if mpath.exists():
            old = json.loads(mpath.read_text())
            # Preserve the original creation date if it exists
            meta.setdefault("created_at", old.get("created_at", now))
        else:
            meta["created_at"] = now

        meta["updated_at"] = now

        self._file(doc.id).write_text(doc.content)
        mpath.write_text(json.dumps(meta, indent=2))

    def all(self):
        """Helper to retrieve all documents in a specific store."""
        return [self.read(i) for i in self.list_ids()]

# --------------------------
# Trace
# --------------------------

def run_id():
    """Generates a unique, sortable ID for every execution run (e.g., run_20240425_123456_abcd)."""
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + \
           "".join(random.choices(string.ascii_lowercase+string.digits,k=4))

class Trace:
    """
    Handles observability and debug logging.
    
    Design Decision: Instead of a flat log file, this creates a structured 
    directory for every run. This allows developers to inspect exactly 
    what was sent to the LLM (calls) and how the workflow progressed (steps).
    """
    def __init__(self, root, rid):
        self.root = Path(root)/rid
        (self.root/"steps").mkdir(parents=True)
        (self.root/"calls").mkdir()

    def j(self, path, obj):
        """Utility for writing JSON files."""
        Path(path).write_text(json.dumps(obj, indent=2))

    def manifest(self, data):
        """Writes the high-level summary of the run."""
        self.j(self.root/"manifest.json", data)

    def step(self, i, name, data):
        """Logs the completion and performance data of a workflow step."""
        self.j(self.root/"steps"/f"{i:03d}_{name}.json", data)

    def call(self, step, i, sys, user, docs):
        """
        Logs a specific LLM interaction.
        
        Design Insight: By logging 'system' and 'user' prompts separately, 
        it becomes easy to reconstruct prompts for testing or fine-tuning.
        """
        d = self.root/"calls"/step/f"{i:04d}"
        (d/"rounds").mkdir(parents=True)
        (d/"system.txt").write_text(sys)
        (d/"user.txt").write_text(user)
        self.j(d/"input_docs.json", [d_.__dict__ for d_ in docs])
        return d

# --------------------------
# Tools
# --------------------------

def make_tools(cfg):
    """
    Factory function that defines the toolset available to the Agent.
    
    Design Insight: By wrapping tools in this closure, we inject the 'cfg' 
    (configuration) context once, allowing tools like 'write_file' to 
    safely know their workspace boundaries without the Agent having 
    to pass paths constantly.
    """
    def ddg(a):
        """DuckDuckGo search tool."""
        out=[]
        with DDGS() as d:
            for r in d.text(a["query"], max_results=5):
                out.append(r)
        return json.dumps(out)

    def fetch(a):
        """Fetches a URL and strips HTML to provide clean text to the LLM."""
        html = httpx.get(a["url"]).text
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style"]): t.decompose()
        return soup.get_text()[:10000]

    def write(a):
        """Writes content to the designated workspace root."""
        p = Path(cfg["workspace_root"]) / a["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(a["content"])
        return "ok"

    def read(a):
        """Reads a file from the workspace."""
        return (Path(cfg["workspace_root"]) / a["path"]).read_text()

    def ls(a):
        """Lists files in the workspace to allow the Agent to explore."""
        p = Path(cfg["workspace_root"]) / a.get("path",".")
        return json.dumps([x.name for x in p.iterdir()])

    def sh(a):
        """
        Executes shell commands. 
        Note: This is powerful and potentially dangerous; meant for sandboxed use.
        """
        r = subprocess.run(a["command"], shell=True,
                           cwd=cfg["workspace_root"],
                           capture_output=True, text=True)
        return json.dumps({"stdout":r.stdout,"stderr":r.stderr})

    return {
        "duckduckgo_search": ddg,
        "fetch_url": fetch,
        "write_file": write,
        "read_file": read,
        "list_directory": ls,
        "run_shell_command": sh,
    }

# --------------------------
# Agent (tool loop)
# --------------------------

class Agent:
    """
    A ReAct-style agent wrapper that handles tool-calling loops.
    """
    def __init__(self, cfg, tools, trace):
        # Design Insight: base_url allows this framework to work with 
        # local providers (like Ollama or vLLM) as well as OpenAI.
        self.client = OpenAI(base_url=cfg["openai_base_url"], api_key="dummy")
        self.cfg = cfg
        self.tools = tools
        self.trace = trace
        self.calls = {} # Counter to track calls per step for unique logging

    def run(self, docs: list[Document], conf: dict, step: str) -> list[Document]:
        """
        Executes a conversation loop with the LLM.
        
        Design Insight: This implementation uses a 'tool loop' limited to 6 turns
        to prevent infinite recursions and excessive token spend.
        """
        sys = conf["system"]
        user = "\n\n".join(d.content for d in docs)

        self.calls[step] = self.calls.get(step,0)+1
        cid = self.calls[step]

        # Log initial state
        cdir = self.trace.call(step, cid, sys, user, docs)

        msgs=[{"role":"system","content":sys},
              {"role":"user","content":user}]

        for _ in range(6):
            resp = self.client.chat.completions.create(
                model=self.cfg["default_model"],
                messages=msgs,
                tools=[{
                    "type":"function",
                    "function":{"name":k}
                } for k in self.tools]
            )

            msg = resp.choices[0].message

            # Check if the LLM wants to use a tool
            if getattr(msg,"tool_calls",None):
                msgs.append(msg)
                for tc in msg.tool_calls:
                    fn = self.tools[tc.function.name]
                    args = json.loads(tc.function.arguments or "{}")
                    result = fn(args)

                    # Feed the tool result back into the message history
                    msgs.append({
                        "role":"tool",
                        "tool_call_id":tc.id,
                        "content":result
                    })
                continue

            # If no tool_calls, the agent has finished its task
            content = msg.content or ""
            return [Document(conf.get("output_id","result.md"), content)]

        return [Document("error.md","Agent failed")]

# --------------------------
# Runner (with map_agent)
# --------------------------

class Runner:
    """
    The engine that executes the 'workflow.json' plan.
    
    Design Insight: The Runner maintains an internal state 'B' which maps 
    output keys to lists of Documents. This allows steps to pass data 
    seamlessly without constantly hitting the disk.
    """
    def __init__(self, stores, agent, trace):
        self.stores = stores
        self.agent = agent
        self.trace = trace

    def run(self, wf):
        """Iterates through steps defined in the workflow manifest."""
        B = {} # The execution context/blackboard

        for i,s in enumerate(wf["steps"],1):
            t0 = datetime.now().timestamp()

            if s["type"] == "read_documents":
                # Pulls documents from a physical Store into the execution context
                B[s["output"]] = self.stores[s["store"]].all()

            elif s["type"] == "run_agent":
                # Passes all documents in a group to a single agent call
                B[s["output"]] = self.agent.run(B[s["input"]],
                                               s["agent_config"],
                                               s["id"])

            elif s["type"] == "map_agent":
                """
                Design Insight: The 'map_agent' pattern is essential for scalability.
                It takes a list of documents and processes each one individually. 
                This prevents context window overflow and allows for isolated logic 
                per document.
                """
                out=[]
                for idx,d in enumerate(B[s["input"]]):
                    res = self.agent.run([d],
                                         s["agent_config"],
                                         f"{s['id']}_{idx}")
                    out.extend(res)
                B[s["output"]] = out

            elif s["type"] == "write_documents":
                # Persists documents from the context back to a physical Store
                for d in B[s["input"]]:
                    self.stores[s["store"]].write(d)

            # Record telemetry for the step
            self.trace.step(i, s["id"], {
                "duration": datetime.now().timestamp() - t0
            })

# --------------------------
# Main
# --------------------------

def main():
    """
    Bootstrap process:
    1. Load global config (.orchestra.json)
    2. Initialize Stores, Tools, and Agent
    3. Load workflow and hand off to the Runner
    """
    cfg = json.loads(Path(".orchestra.json").read_text())

    rid = run_id()
    trace = Trace(cfg["traces_root"], rid)

    # Automatically discover and initialize all stores in the store directory
    stores = {
        d.name: Store(d)
        for d in Path(cfg["store_root"]).iterdir()
        if d.is_dir()
    }

    tools = make_tools(cfg)
    agent = Agent(cfg, tools, trace)

    wf = json.loads(Path(cfg["workflow_path"]).read_text())

    # Log the manifest before starting execution
    trace.manifest({"run_id": rid, "workflow": wf["id"]})

    Runner(stores, agent, trace).run(wf)

    print("\nDONE:", rid)

if __name__ == "__main__":
    main()
