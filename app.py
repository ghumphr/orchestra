# (TRUNCATED HEADER COMMENT)
# Full Orchestra v1.1 runtime with:
# - filename-based document IDs
# - sidecar metadata
# - full trace logging
# - streaming responses
# - tool system
# --------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Any, Callable
from datetime import datetime, timezone
import argparse, json, os, re, time, subprocess, random, string

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI

# ----------------------------
# Core Data Model
# ----------------------------

@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

# ----------------------------
# Store Implementation
# ----------------------------

class FolderDocumentStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _content_path(self, doc_id: str) -> Path:
        return self.root / doc_id

    def _metadata_path(self, doc_id: str) -> Path:
        return self.root / f".{doc_id}.json"

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def list_ids(self):
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )

    def read(self, doc_id: str) -> Document:
        content = self._content_path(doc_id).read_text()
        meta_path = self._metadata_path(doc_id)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {"id": doc_id}
        return Document(id=doc_id, content=content, metadata=metadata)

    def write(self, doc: Document):
        content_path = self._content_path(doc.id)
        meta_path = self._metadata_path(doc.id)

        meta = dict(doc.metadata)
        meta["id"] = doc.id
        now = self._now()

        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
                meta.setdefault("created_at", existing.get("created_at", now))
            except:
                meta.setdefault("created_at", now)
        else:
            meta["created_at"] = now

        meta["updated_at"] = now

        content_path.write_text(doc.content)
        meta_path.write_text(json.dumps(meta, indent=2))

    def iter_documents(self):
        return [self.read(i) for i in self.list_ids()]

# ----------------------------
# Trace System
# ----------------------------

def make_run_id():
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_" + \
           "".join(random.choices(string.ascii_lowercase+string.digits,k=4))

class TraceWriter:
    def __init__(self, root, run_id):
        self.root = Path(root) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/"steps").mkdir()
        (self.root/"agent_calls").mkdir()

    def write_json(self, path, obj):
        Path(path).write_text(json.dumps(obj, indent=2))

    def manifest(self, data):
        self.write_json(self.root/"manifest.json", data)

    def step(self, idx, step_id, data):
        self.write_json(self.root/"steps"/f"{idx:03d}_{step_id}.json", data)

    def agent_call(self, step_id, call_idx, system, user, docs):
        p = self.root/"agent_calls"/step_id/f"{call_idx:04d}"
        (p/"rounds").mkdir(parents=True)
        (p/"system.txt").write_text(system)
        (p/"user.txt").write_text(user)
        self.write_json(p/"input_docs.json",[d.__dict__ for d in docs])
        return p

# ----------------------------
# Tools
# ----------------------------

def make_tools(cfg):
    tools = {}

    def register(name, desc, schema, fn):
        tools[name] = {"spec": {
            "type":"function",
            "function":{
                "name":name,
                "description":desc,
                "parameters":schema
            }
        }, "fn":fn}

    def ddg(args):
        out=[]
        with DDGS() as ddgs:
            for r in ddgs.text(args["query"],max_results=5):
                out.append(r)
        return json.dumps(out)

    def fetch(args):
        html=httpx.get(args["url"]).text
        soup=BeautifulSoup(html,"html.parser")
        for t in soup(["script","style"]): t.decompose()
        return soup.get_text()[:12000]

    def write_file(args):
        p=Path(cfg.workspace_root)/args["path"]
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(args["content"])
        return "ok"

    def read_file(args):
        return (Path(cfg.workspace_root)/args["path"]).read_text()

    def list_dir(args):
        p=Path(cfg.workspace_root)/args.get("path",".")
        return json.dumps([x.name for x in p.iterdir()])

    def shell(args):
        r=subprocess.run(args["command"],shell=True,cwd=cfg.workspace_root,
                         capture_output=True,text=True)
        return json.dumps({"stdout":r.stdout,"stderr":r.stderr})

    register("duckduckgo_search","Search web",{"type":"object","properties":{"query":{"type":"string"}}},ddg)
    register("fetch_url","Fetch page",{"type":"object","properties":{"url":{"type":"string"}}},fetch)
    register("write_file","Write file",{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}},write_file)
    register("read_file","Read file",{"type":"object","properties":{"path":{"type":"string"}}},read_file)
    register("list_directory","List dir",{"type":"object","properties":{"path":{"type":"string"}}},list_dir)
    register("run_shell_command","Run shell",{"type":"object","properties":{"command":{"type":"string"}}},shell)

    return tools

# ----------------------------
# Agent
# ----------------------------

class AgentLLM:
    def __init__(self,cfg,tools,trace):
        self.client=OpenAI(base_url=cfg.openai_base_url,api_key="dummy")
        self.cfg=cfg
        self.tools=tools
        self.trace=trace
        self.calls={}

    def run(self,docs,config,step_id):
        sys=config["system"]
        user="\n\n".join(d.content for d in docs)

        self.calls[step_id]=self.calls.get(step_id,0)+1
        call_id=self.calls[step_id]

        call_dir=self.trace.agent_call(step_id,call_id,sys,user,docs)

        messages=[{"role":"system","content":sys},{"role":"user","content":user}]

        stream=self.client.chat.completions.create(
            model=self.cfg.default_model,
            messages=messages,
            stream=True
        )

        out=""
        for c in stream:
            if c.choices:
                delta=c.choices[0].delta
                if delta.content:
                    print(delta.content,end="",flush=True)
                    out+=delta.content

        print()
        return [Document(id=config.get("output_id","result.md"),content=out)]

# ----------------------------
# Workflow Runner
# ----------------------------

class Runner:
    def __init__(self,cfg,stores,agent,trace):
        self.cfg=cfg
        self.stores=stores
        self.agent=agent
        self.trace=trace

    def run(self,wf):
        bindings={}
        for i,step in enumerate(wf["steps"],1):
            sid=step["id"]
            t0=time.time()

            if step["type"]=="read_documents":
                docs=self.stores[step["store"]].iter_documents()
                bindings[step["output"]]=docs

            elif step["type"]=="run_agent":
                docs=bindings[step["input"]]
                out=self.agent.run(docs,step["agent_config"],sid)
                bindings[step["output"]]=out

            elif step["type"]=="write_documents":
                for d in bindings[step["input"]]:
                    self.stores[step["store"]].write(d)

            self.trace.step(i,sid,{
                "step":sid,
                "duration":time.time()-t0
            })

# ----------------------------
# Main
# ----------------------------

def main():
    cfg=type("Cfg",(),json.loads(Path(".orchestra.json").read_text()))()

    run_id=make_run_id()
    trace=TraceWriter(cfg.traces_root,run_id)

    stores={}
    for d in Path(cfg.store_root).iterdir():
        if d.is_dir():
            stores[d.name]=FolderDocumentStore(d)

    tools=make_tools(cfg)
    agent=AgentLLM(cfg,tools,trace)

    wf=json.loads(Path(cfg.workflow_path).read_text())

    trace.manifest({"run_id":run_id,"workflow":wf["id"]})

    Runner(cfg,stores,agent,trace).run(wf)

    print("\nDONE:",run_id)

if __name__=="__main__":
    main()
