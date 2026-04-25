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
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

# --------------------------
# Store (sidecar metadata)
# --------------------------

class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta(self, id): return self.root / f".{id}.json"
    def _file(self, id): return self.root / id

    def list_ids(self):
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_file() and not p.name.startswith("."))

    def read(self, id):
        txt = self._file(id).read_text()
        m = self._meta(id)
        meta = json.loads(m.read_text()) if m.exists() else {"id": id}
        return Document(id, txt, meta)

    def write(self, doc: Document):
        now = datetime.now(timezone.utc).isoformat()
        meta = dict(doc.metadata)
        meta["id"] = doc.id

        mpath = self._meta(doc.id)
        if mpath.exists():
            old = json.loads(mpath.read_text())
            meta.setdefault("created_at", old.get("created_at", now))
        else:
            meta["created_at"] = now

        meta["updated_at"] = now

        self._file(doc.id).write_text(doc.content)
        mpath.write_text(json.dumps(meta, indent=2))

    def all(self):
        return [self.read(i) for i in self.list_ids()]

# --------------------------
# Trace
# --------------------------

def run_id():
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + \
           "".join(random.choices(string.ascii_lowercase+string.digits,k=4))

class Trace:
    def __init__(self, root, rid):
        self.root = Path(root)/rid
        (self.root/"steps").mkdir(parents=True)
        (self.root/"calls").mkdir()

    def j(self, path, obj):
        Path(path).write_text(json.dumps(obj, indent=2))

    def manifest(self, data):
        self.j(self.root/"manifest.json", data)

    def step(self, i, name, data):
        self.j(self.root/"steps"/f"{i:03d}_{name}.json", data)

    def call(self, step, i, sys, user, docs):
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
    def ddg(a):
        out=[]
        with DDGS() as d:
            for r in d.text(a["query"], max_results=5):
                out.append(r)
        return json.dumps(out)

    def fetch(a):
        html = httpx.get(a["url"]).text
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style"]): t.decompose()
        return soup.get_text()[:10000]

    def write(a):
        p = Path(cfg["workspace_root"]) / a["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(a["content"])
        return "ok"

    def read(a):
        return (Path(cfg["workspace_root"]) / a["path"]).read_text()

    def ls(a):
        p = Path(cfg["workspace_root"]) / a.get("path",".")
        return json.dumps([x.name for x in p.iterdir()])

    def sh(a):
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
    def __init__(self, cfg, tools, trace):
        self.client = OpenAI(base_url=cfg["openai_base_url"], api_key="dummy")
        self.cfg = cfg
        self.tools = tools
        self.trace = trace
        self.calls = {}

    def run(self, docs, conf, step):
        sys = conf["system"]
        user = "\n\n".join(d.content for d in docs)

        self.calls[step] = self.calls.get(step,0)+1
        cid = self.calls[step]

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

            if getattr(msg,"tool_calls",None):
                msgs.append(msg)
                for tc in msg.tool_calls:
                    fn = self.tools[tc.function.name]
                    args = json.loads(tc.function.arguments or "{}")
                    result = fn(args)

                    msgs.append({
                        "role":"tool",
                        "tool_call_id":tc.id,
                        "content":result
                    })
                continue

            content = msg.content or ""
            return [Document(conf.get("output_id","result.md"), content)]

        return [Document("error.md","Agent failed")]

# --------------------------
# Runner (with map_agent)
# --------------------------

class Runner:
    def __init__(self, stores, agent, trace):
        self.stores = stores
        self.agent = agent
        self.trace = trace

    def run(self, wf):
        B = {}

        for i,s in enumerate(wf["steps"],1):
            t0 = datetime.now().timestamp()

            if s["type"] == "read_documents":
                B[s["output"]] = self.stores[s["store"]].all()

            elif s["type"] == "run_agent":
                B[s["output"]] = self.agent.run(B[s["input"]],
                                               s["agent_config"],
                                               s["id"])

            elif s["type"] == "map_agent":
                out=[]
                for idx,d in enumerate(B[s["input"]]):
                    res = self.agent.run([d],
                                         s["agent_config"],
                                         f"{s['id']}_{idx}")
                    out.extend(res)
                B[s["output"]] = out

            elif s["type"] == "write_documents":
                for d in B[s["input"]]:
                    self.stores[s["store"]].write(d)

            self.trace.step(i, s["id"], {
                "duration": datetime.now().timestamp() - t0
            })

# --------------------------
# Main
# --------------------------

def main():
    cfg = json.loads(Path(".orchestra.json").read_text())

    rid = run_id()
    trace = Trace(cfg["traces_root"], rid)

    stores = {
        d.name: Store(d)
        for d in Path(cfg["store_root"]).iterdir()
        if d.is_dir()
    }

    tools = make_tools(cfg)
    agent = Agent(cfg, tools, trace)

    wf = json.loads(Path(cfg["workflow_path"]).read_text())

    trace.manifest({"run_id": rid, "workflow": wf["id"]})

    Runner(stores, agent, trace).run(wf)

    print("\nDONE:", rid)

if __name__ == "__main__":
    main()
