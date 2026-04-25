from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Any, Callable
from datetime import datetime, timezone
import argparse
import json
import os
import re
import time
import subprocess
import random
import string

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentStore(Protocol):
    def list_ids(self) -> list[str]: ...
    def exists(self, doc_id: str) -> bool: ...
    def read(self, doc_id: str) -> Document: ...
    def write(self, document: Document) -> None: ...
    def iter_documents(self) -> list[Document]: ...


class Agent(Protocol):
    def execute(self, documents: list[Document], config: dict[str, Any]) -> list[Document]: ...


@dataclass
class WorkflowStep:
    id: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    id: str
    steps: list[WorkflowStep]


@dataclass
class RunContext:
    bindings: dict[str, list[Document]] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    workflow_path: str = "./workflow.json"
    store_root: str = "./stores"
    prompt_root: str = "./prompts"
    workspace_root: str = "./workspace"
    traces_root: str = "./traces"
    library_root: str = "./library"
    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_api_key: str = "dummy"
    default_model: str = "default"
    config_path: str | None = None
    verbosity: int = 3
    max_tool_rounds: int = 10
    max_duplicate_blocks_before_finalize: int = 2


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"run_{stamp}_{suffix}"


class FolderDocumentStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _content_path(self, doc_id: str) -> Path:
        return self.root / doc_id

    def _metadata_path(self, doc_id: str) -> Path:
        return self.root / f".{doc_id}.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def list_ids(self) -> list[str]:
        ids: list[str] = []
        for p in self.root.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            ids.append(p.name)
        return sorted(ids)

    def exists(self, doc_id: str) -> bool:
        return self._content_path(doc_id).exists()

    def read(self, doc_id: str) -> Document:
        content_path = self._content_path(doc_id)
        metadata_path = self._metadata_path(doc_id)

        if not content_path.exists():
            raise FileNotFoundError(f"Document not found: {content_path}")

        content = content_path.read_text(encoding="utf-8")

        metadata: dict[str, Any]
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {"id": doc_id}

        metadata.setdefault("id", doc_id)
        return Document(id=doc_id, content=content, metadata=metadata)

    def write(self, document: Document) -> None:
        content_path = self._content_path(document.id)
        metadata_path = self._metadata_path(document.id)

        now = self._now_iso()
        metadata = dict(document.metadata)
        metadata["id"] = document.id

        if metadata_path.exists():
            try:
                existing_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}
            metadata.setdefault("created_at", existing_meta.get("created_at", now))
        else:
            metadata.setdefault("created_at", now)

        metadata["updated_at"] = now

        content_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        content_path.write_text(document.content, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def iter_documents(self) -> list[Document]:
        return [self.read(doc_id) for doc_id in self.list_ids()]


class PromptRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, prompt_id: str) -> dict[str, Any]:
        path = self.root / f"{prompt_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt profile not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


class TraceWriter:
    def __init__(self, traces_root: str | Path, run_id: str) -> None:
        self.traces_root = Path(traces_root)
        self.run_id = run_id
        self.run_root = self.traces_root / run_id
        self.steps_root = self.run_root / "steps"
        self.calls_root = self.run_root / "agent_calls"

        self.steps_root.mkdir(parents=True, exist_ok=True)
        self.calls_root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.run_root / "manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_step_record(self, step_index: int, step_id: str, record: dict[str, Any]) -> None:
        path = self.steps_root / f"{step_index:03d}_{step_id}.json"
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def begin_agent_call(
        self,
        step_id: str,
        call_index: int,
        request: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> Path:
        call_dir = self.calls_root / step_id / f"{call_index:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        (call_dir / "request.json").write_text(
            json.dumps(request, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (call_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (call_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")

        return call_dir

    def write_agent_response(self, call_dir: Path, response_text: str) -> None:
        (call_dir / "response.txt").write_text(response_text, encoding="utf-8")

    def write_tool_calls(self, call_dir: Path, tool_calls: list[dict[str, Any]]) -> None:
        (call_dir / "tool_calls.json").write_text(
            json.dumps(tool_calls, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_tool_results(self, call_dir: Path, tool_results: list[dict[str, Any]]) -> None:
        (call_dir / "tool_results.json").write_text(
            json.dumps(tool_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_output_docs(self, call_dir: Path, docs: list[Document]) -> None:
        payload = [
            {
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
            }
            for d in docs
        ]
        (call_dir / "output_docs.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]

    def openai_tools(self, names: list[str]) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for name in names:
            tool = self.get(name)
            specs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return specs


def safe_workspace_path(workspace_root: str, relative_path: str) -> Path:
    root = Path(workspace_root).resolve()
    target = (root / relative_path).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("Path escapes workspace root")
    return target


def make_tools(runtime_config: RuntimeConfig) -> ToolRegistry:
    registry = ToolRegistry()

    def tool_duckduckgo_search(args: dict[str, Any]) -> str:
        query = args["query"]
        max_results = int(args.get("max_results", 8))
        results = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": row.get("title", ""),
                    "url": row.get("href", ""),
                    "snippet": row.get("body", ""),
                })
        return json.dumps(results, ensure_ascii=False)

    def tool_fetch_url(args: dict[str, Any]) -> str:
        url = args["url"]
        timeout_seconds = float(args.get("timeout_seconds", 20))

        headers = {
            "User-Agent": "Mozilla/5.0 Orchestra/0.1",
        }

        with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        content = "\n".join(lines)[:16000]

        return json.dumps({
            "url": url,
            "title": title,
            "content": content,
        }, ensure_ascii=False)

    def tool_list_directory(args: dict[str, Any]) -> str:
        rel = args.get("path", ".")
        path = safe_workspace_path(runtime_config.workspace_root, rel)
        if not path.exists():
            return json.dumps({"error": "not_found", "path": rel}, ensure_ascii=False)
        if not path.is_dir():
            return json.dumps({"error": "not_directory", "path": rel}, ensure_ascii=False)

        entries = []
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            entries.append({
                "name": child.name,
                "type": "directory" if child.is_dir() else "file",
            })
        return json.dumps({"path": rel, "entries": entries}, ensure_ascii=False)

    def tool_read_file(args: dict[str, Any]) -> str:
        rel = args["path"]
        path = safe_workspace_path(runtime_config.workspace_root, rel)
        if not path.exists():
            return json.dumps({"error": "not_found", "path": rel}, ensure_ascii=False)
        if path.is_dir():
            return json.dumps({"error": "is_directory", "path": rel}, ensure_ascii=False)
        content = path.read_text(encoding="utf-8")
        return json.dumps({"path": rel, "content": content}, ensure_ascii=False)

    def tool_write_file(args: dict[str, Any]) -> str:
        rel = args["path"]
        content = args["content"]
        path = safe_workspace_path(runtime_config.workspace_root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing == content:
                return json.dumps({
                    "status": "ok",
                    "path": rel,
                    "message": "File already exists with identical content. No change was needed."
                }, ensure_ascii=False)

        path.write_text(content, encoding="utf-8")
        return json.dumps({
            "status": "ok",
            "path": rel,
            "message": f"File '{rel}' written successfully."
        }, ensure_ascii=False)

    def tool_run_shell_command(args: dict[str, Any]) -> str:
        command = args["command"]
        timeout_seconds = int(args.get("timeout_seconds", 60))

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=runtime_config.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return json.dumps({
                "status": "ok",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            }, ensure_ascii=False)
        except subprocess.TimeoutExpired as e:
            return json.dumps({
                "status": "timeout",
                "exit_code": None,
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "command": command,
            }, ensure_ascii=False)

    registry.register(ToolSpec(
        name="duckduckgo_search",
        description="Search the web using DuckDuckGo and return a compact list of results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20}
            },
            "required": ["query"]
        },
        handler=tool_duckduckgo_search,
    ))

    registry.register(ToolSpec(
        name="fetch_url",
        description="Fetch a web page URL and return extracted readable text content.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_seconds": {"type": "number", "minimum": 1, "maximum": 60}
            },
            "required": ["url"]
        },
        handler=tool_fetch_url,
    ))

    registry.register(ToolSpec(
        name="list_directory",
        description="List files and directories under a workspace-relative path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        },
        handler=tool_list_directory,
    ))

    registry.register(ToolSpec(
        name="read_file",
        description="Read a text file from the workspace using a relative path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        },
        handler=tool_read_file,
    ))

    registry.register(ToolSpec(
        name="write_file",
        description="Write a text file to the workspace using a relative path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        },
        handler=tool_write_file,
    ))

    registry.register(ToolSpec(
        name="run_shell_command",
        description="Run a shell command inside the workspace and return exit code, stdout, and stderr.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300}
            },
            "required": ["command"]
        },
        handler=tool_run_shell_command,
    ))

    return registry


class OpenAIChatAgent:
    def __init__(
        self,
        prompt_registry: PromptRegistry,
        runtime_config: RuntimeConfig,
        tool_registry: ToolRegistry,
        trace_writer: TraceWriter,
    ) -> None:
        self.prompt_registry = prompt_registry
        self.runtime_config = runtime_config
        self.tool_registry = tool_registry
        self.trace_writer = trace_writer
        self.client = OpenAI(
            base_url=runtime_config.openai_base_url,
            api_key=runtime_config.openai_api_key,
        )
        self.call_counters: dict[str, int] = {}

    def _log(self, message: str, level: int = 1) -> None:
        if self.runtime_config.verbosity >= level:
            print(message)

    def _print_block(self, title: str, body: str, level: int = 3, limit: int | None = None) -> None:
        if self.runtime_config.verbosity < level:
            return
        print(f"\n--- {title} ---")
        if limit is not None and len(body) > limit:
            print(body[:limit])
            print("\n[truncated]")
        else:
            print(body)

    def _stream_chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
    ) -> tuple[str, list[dict[str, Any]]]:
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=temperature,
            stream=True,
        )

        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        if self.runtime_config.verbosity >= 3:
            print("\n--- STREAM START ---")

        for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            delta_content = getattr(delta, "content", None)
            if delta_content:
                content_parts.append(delta_content)
                if self.runtime_config.verbosity >= 3:
                    print(delta_content, end="", flush=True)

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc in delta_tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            },
                        }

                    entry = tool_calls_by_index[idx]

                    if getattr(tc, "id", None):
                        entry["id"] = tc.id
                    if getattr(tc, "type", None):
                        entry["type"] = tc.type

                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["function"]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            entry["function"]["arguments"] += fn.arguments

        if self.runtime_config.verbosity >= 3:
            print("\n--- STREAM END ---")

        full_content = "".join(content_parts)
        normalized_tool_calls = [
            tool_calls_by_index[i]
            for i in sorted(tool_calls_by_index.keys())
        ]
        return full_content, normalized_tool_calls

    def execute(self, documents: list[Document], config: dict[str, Any]) -> list[Document]:
        prompt_profile_id = config["prompt_profile"]
        prompt_profile = self.prompt_registry.load(prompt_profile_id)

        model = config.get("model") or self.runtime_config.default_model
        system_prompt = prompt_profile["system_prompt"]
        user_template = prompt_profile.get("user_template")
        output_mode = prompt_profile.get("output_mode", "single_document")
        temperature = config.get("temperature", prompt_profile.get("temperature", 0.2))
        allowed_tools = config.get("tools", prompt_profile.get("tools", []))
        trace_step_id = config.get("_trace_step_id", "unknown_step")

        joined_input = "\n\n".join(
            f"[Document: {doc.id}]\n{doc.content}" for doc in documents
        )
        user_prompt = user_template.replace("{{input}}", joined_input) if user_template else joined_input

        self._print_block("SYSTEM PROMPT", system_prompt, level=3)
        self._print_block("USER PROMPT", user_prompt, level=3, limit=20000)

        self.call_counters[trace_step_id] = self.call_counters.get(trace_step_id, 0) + 1
        call_index = self.call_counters[trace_step_id]

        request_payload = {
            "model": model,
            "temperature": temperature,
            "tools_enabled": allowed_tools,
            "input_doc_ids": [d.id for d in documents],
        }

        call_dir = self.trace_writer.begin_agent_call(
            step_id=trace_step_id,
            call_index=call_index,
            request=request_payload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tools = self.tool_registry.openai_tools(allowed_tools) if allowed_tools else None

        tool_signature_counts: dict[str, int] = {}
        duplicate_block_count = 0
        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []

        max_identical_call_counts = {
            "duckduckgo_search": int(config.get("max_identical_duckduckgo_search_calls", 2)),
            "fetch_url": int(config.get("max_identical_fetch_url_calls", 2)),
            "list_directory": int(config.get("max_identical_list_directory_calls", 1)),
            "read_file": int(config.get("max_identical_read_file_calls", 2)),
            "write_file": int(config.get("max_identical_write_file_calls", 5)),
            "run_shell_command": int(config.get("max_identical_run_shell_command_calls", 2)),
        }

        final_response_text = ""

        for round_num in range(self.runtime_config.max_tool_rounds):
            self._log(f"\n[tool-round {round_num + 1}] model={model}", 2)

            content, tool_calls = self._stream_chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )

            if content:
                final_response_text += content

            if tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_message)
                all_tool_calls.extend(tool_calls)

                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"] or "{}"

                    self._log(f"\n[TOOL CALL] {tool_name}", 4)
                    self._log(f"ARGS RAW: {raw_args}", 4)

                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {"raw": raw_args}

                    self._log(f"ARGS PARSED: {parsed_args}", 4)

                    signature = f"{tool_name}:{json.dumps(parsed_args, sort_keys=True, ensure_ascii=False)}"
                    tool_signature_counts[signature] = tool_signature_counts.get(signature, 0) + 1
                    count = tool_signature_counts[signature]
                    limit = max_identical_call_counts.get(tool_name, 1)

                    if count > limit:
                        duplicate_block_count += 1
                        result = json.dumps({
                            "error": "tool_unavailable",
                            "reason": "duplicate_call_blocked",
                            "message": (
                                f"The exact tool call '{tool_name}' with those arguments has already been used "
                                f"{count - 1} times in this run. Do not repeat it. "
                                f"If the work is already done, produce your final answer now."
                            )
                        }, ensure_ascii=False)

                        result_record = {
                            "tool_name": tool_name,
                            "arguments": parsed_args,
                            "result": result,
                        }
                        all_tool_results.append(result_record)

                        self._log("[TOOL RESULT PREVIEW]", 4)
                        self._log(result[:4000], 4)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })

                        if duplicate_block_count >= self.runtime_config.max_duplicate_blocks_before_finalize:
                            self._log("[duplicate-call threshold reached] forcing final answer without tools", 2)
                            messages.append({
                                "role": "system",
                                "content": (
                                    "You have already completed or attempted the relevant tool operations. "
                                    "No more tool calls are allowed. Produce your final answer now summarizing what you created, tested, or changed."
                                )
                            })

                            forced_content, _ = self._stream_chat_completion(
                                model=model,
                                messages=messages,
                                tools=None,
                                temperature=temperature,
                            )
                            final_response_text += forced_content
                            docs = self._documents_from_output_mode(forced_content, output_mode, documents, config)

                            self.trace_writer.write_agent_response(call_dir, final_response_text)
                            self.trace_writer.write_tool_calls(call_dir, all_tool_calls)
                            self.trace_writer.write_tool_results(call_dir, all_tool_results)
                            self.trace_writer.write_output_docs(call_dir, docs)

                            self._log("\n--- FINAL FORCED RETURNED DOCUMENTS ---", 2)
                            for d in docs:
                                self._log(f"[{d.id}]", 2)
                                self._log(d.content[:1500], 2)
                                self._log("", 2)
                            return docs

                        continue

                    tool = self.tool_registry.get(tool_name)
                    try:
                        result = tool.handler(parsed_args)
                    except Exception as e:
                        result = json.dumps({
                            "error": "tool_execution_failed",
                            "message": str(e),
                        }, ensure_ascii=False)

                    result_record = {
                        "tool_name": tool_name,
                        "arguments": parsed_args,
                        "result": result,
                    }
                    all_tool_results.append(result_record)

                    self._log("[TOOL RESULT PREVIEW]", 4)
                    self._log(result[:4000], 4)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                continue

            docs = self._documents_from_output_mode(content, output_mode, documents, config)
            self.trace_writer.write_agent_response(call_dir, final_response_text)
            self.trace_writer.write_tool_calls(call_dir, all_tool_calls)
            self.trace_writer.write_tool_results(call_dir, all_tool_results)
            self.trace_writer.write_output_docs(call_dir, docs)

            self._log("\n--- RETURNED DOCUMENTS ---", 2)
            for d in docs:
                self._log(f"[{d.id}]", 2)
                self._log(d.content[:1500], 2)
                self._log("", 2)
            return docs

        self._log("[tool-round limit reached] forcing final answer without tools", 2)
        messages.append({
            "role": "system",
            "content": "No more tool calls are available in this run. Produce your best final answer now."
        })

        forced_content, _ = self._stream_chat_completion(
            model=model,
            messages=messages,
            tools=None,
            temperature=temperature,
        )
        final_response_text += forced_content
        docs = self._documents_from_output_mode(forced_content, output_mode, documents, config)

        self.trace_writer.write_agent_response(call_dir, final_response_text)
        self.trace_writer.write_tool_calls(call_dir, all_tool_calls)
        self.trace_writer.write_tool_results(call_dir, all_tool_results)
        self.trace_writer.write_output_docs(call_dir, docs)

        self._log("\n--- FINAL FORCED RETURNED DOCUMENTS ---", 2)
        for d in docs:
            self._log(f"[{d.id}]", 2)
            self._log(d.content[:1500], 2)
            self._log("", 2)
        return docs

    def _documents_from_output_mode(
        self,
        content: str,
        output_mode: str,
        documents: list[Document],
        config: dict[str, Any],
    ) -> list[Document]:
        if output_mode == "single_document":
            output_id = config.get("output_id", documents[0].id if documents else "llm_output.txt")
            return [Document(id=output_id, content=content)]

        if output_mode == "json_documents":
            parsed = self._parse_json_array(content)
            out: list[Document] = []
            for item in parsed:
                out.append(
                    Document(
                        id=item["id"],
                        content=item["content"],
                        metadata=item.get("metadata", {}),
                    )
                )
            return out

        raise ValueError(f"Unknown output_mode: {output_mode}")

    def _parse_json_array(self, text: str) -> list[dict[str, Any]]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


class WorkflowRunner:
    def __init__(
        self,
        stores: dict[str, DocumentStore],
        agents: dict[str, Agent],
        runtime_config: RuntimeConfig,
        trace_writer: TraceWriter,
        run_id: str,
    ) -> None:
        self.stores = stores
        self.agents = agents
        self.runtime_config = runtime_config
        self.trace_writer = trace_writer
        self.run_id = run_id

    def log(self, message: str, level: int = 1) -> None:
        if self.runtime_config.verbosity >= level:
            print(message)

    def run(self, workflow: Workflow, context: RunContext | None = None) -> RunContext:
        context = context or RunContext()

        for step_index, step in enumerate(workflow.steps, start=1):
            started = datetime.now(timezone.utc).isoformat()
            wall_start = time.time()

            self.log(f"\n== Step: {step.id} ({step.type}) ==", 1)

            step_record: dict[str, Any] = {
                "step_index": step_index,
                "step_id": step.id,
                "step_type": step.type,
                "started_at": started,
                "status": "ok",
            }

            try:
                if step.type == "read_documents":
                    store_name = step.config["store"]
                    output = step.config["output"]
                    docs = self.stores[store_name].iter_documents()
                    context.bindings[output] = docs
                    self.log(f"Read {len(docs)} documents from '{store_name}' into '{output}'", 1)
                    if self.runtime_config.verbosity >= 2:
                        for d in docs:
                            self.log(f"  - {d.id}", 2)

                    step_record["input_store"] = store_name
                    step_record["binding"] = output
                    step_record["processed_ids"] = [d.id for d in docs]

                elif step.type == "write_documents":
                    input_name = step.config["input"]
                    store_name = step.config["store"]
                    docs = context.bindings[input_name]
                    for doc in docs:
                        self.stores[store_name].write(doc)
                    self.log(f"Wrote {len(docs)} documents from '{input_name}' to '{store_name}'", 1)
                    if self.runtime_config.verbosity >= 2:
                        for d in docs:
                            self.log(f"  - {d.id}", 2)

                    step_record["output_store"] = store_name
                    step_record["processed_ids"] = [d.id for d in docs]

                elif step.type == "run_agent":
                    agent_name = step.config["agent"]
                    input_name = step.config["input"]
                    output = step.config["output"]
                    agent_config = dict(step.config.get("agent_config", {}))
                    agent_config["_trace_step_id"] = step.id
                    docs = context.bindings[input_name]
                    self.log(f"Running agent '{agent_name}' on binding '{input_name}' with {len(docs)} docs", 1)
                    if self.runtime_config.verbosity >= 2:
                        for d in docs:
                            self.log(f"  input doc: {d.id}", 2)
                    result = self.agents[agent_name].execute(docs, agent_config)
                    context.bindings[output] = result
                    self.log(f"Agent '{agent_name}' produced {len(result)} docs into '{output}'", 1)
                    if self.runtime_config.verbosity >= 2:
                        for d in result:
                            self.log(f"  output doc: {d.id}", 2)

                    step_record["input_binding"] = input_name
                    step_record["output_binding"] = output
                    step_record["processed_ids"] = [d.id for d in docs]
                    step_record["produced_ids"] = [d.id for d in result]

                elif step.type == "reduce_documents":
                    agent_name = step.config["agent"]
                    input_name = step.config["input"]
                    output = step.config["output"]
                    agent_config = dict(step.config.get("agent_config", {}))
                    agent_config["_trace_step_id"] = step.id
                    docs = context.bindings[input_name]
                    self.log(f"Reducing {len(docs)} docs from '{input_name}' via '{agent_name}'", 1)
                    result = self.agents[agent_name].execute(docs, agent_config)
                    context.bindings[output] = result
                    self.log(f"Reduce agent '{agent_name}' produced {len(result)} docs into '{output}'", 1)
                    if self.runtime_config.verbosity >= 2:
                        for d in result:
                            self.log(f"  output doc: {d.id}", 2)

                    step_record["input_binding"] = input_name
                    step_record["output_binding"] = output
                    step_record["processed_ids"] = [d.id for d in docs]
                    step_record["produced_ids"] = [d.id for d in result]

                else:
                    raise ValueError(f"Unknown step type: {step.type}")

            except Exception as e:
                step_record["status"] = "error"
                step_record["error"] = str(e)
                step_record["ended_at"] = datetime.now(timezone.utc).isoformat()
                self.trace_writer.write_step_record(step_index, step.id, step_record)
                raise

            elapsed = time.time() - wall_start
            self.log(f"[step time] {elapsed:.2f}s", 1)

            step_record["ended_at"] = datetime.now(timezone.utc).isoformat()
            step_record["elapsed_seconds"] = elapsed
            self.trace_writer.write_step_record(step_index, step.id, step_record)

        return context


def load_workflow(path: str | Path) -> Workflow:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Workflow(
        id=data["id"],
        steps=[WorkflowStep(**step) for step in data["steps"]],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", dest="workflow_path")
    parser.add_argument("--store-root")
    parser.add_argument("--prompt-root")
    parser.add_argument("--workspace-root")
    parser.add_argument("--traces-root")
    parser.add_argument("--library-root")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--openai-api-key")
    parser.add_argument("--model", dest="default_model")
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--verbose", type=int, default=None)
    return parser.parse_args()


def load_dotfile(path: str | None) -> dict[str, Any]:
    candidates = [Path(path)] if path else [Path(".orchestra.json")]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def load_runtime_config() -> RuntimeConfig:
    args = parse_args()
    dotfile_data = load_dotfile(args.config_path)

    def resolve(name: str, env_name: str, default: Any) -> Any:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        if env_name in os.environ:
            return os.environ[env_name]
        if name in dotfile_data:
            return dotfile_data[name]
        return default

    verbosity = args.verbose if args.verbose is not None else dotfile_data.get("verbosity", 3)

    return RuntimeConfig(
        workflow_path=resolve("workflow_path", "ORCHESTRA_WORKFLOW", "./workflow.json"),
        store_root=resolve("store_root", "ORCHESTRA_STORE_ROOT", "./stores"),
        prompt_root=resolve("prompt_root", "ORCHESTRA_PROMPT_ROOT", "./prompts"),
        workspace_root=resolve("workspace_root", "ORCHESTRA_WORKSPACE_ROOT", "./workspace"),
        traces_root=resolve("traces_root", "ORCHESTRA_TRACES_ROOT", "./traces"),
        library_root=resolve("library_root", "ORCHESTRA_LIBRARY_ROOT", "./library"),
        openai_base_url=resolve("openai_base_url", "OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        openai_api_key=resolve("openai_api_key", "OPENAI_API_KEY", "dummy"),
        default_model=resolve("default_model", "ORCHESTRA_DEFAULT_MODEL", "default"),
        config_path=resolve("config_path", "ORCHESTRA_CONFIG", None),
        verbosity=int(verbosity),
    )


def build_store_map(store_root: str) -> dict[str, FolderDocumentStore]:
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)

    stores: dict[str, FolderDocumentStore] = {}
    for child in root.iterdir():
        if child.is_dir():
            stores[child.name] = FolderDocumentStore(child)
    return stores


def main() -> None:
    runtime_config = load_runtime_config()

    Path(runtime_config.workspace_root).mkdir(parents=True, exist_ok=True)
    Path(runtime_config.traces_root).mkdir(parents=True, exist_ok=True)
    Path(runtime_config.library_root).mkdir(parents=True, exist_ok=True)

    run_id = make_run_id()
    trace_writer = TraceWriter(runtime_config.traces_root, run_id)

    print("Runtime config:")
    print(f"  workflow_path={runtime_config.workflow_path}")
    print(f"  store_root={runtime_config.store_root}")
    print(f"  prompt_root={runtime_config.prompt_root}")
    print(f"  workspace_root={runtime_config.workspace_root}")
    print(f"  traces_root={runtime_config.traces_root}")
    print(f"  library_root={runtime_config.library_root}")
    print(f"  openai_base_url={runtime_config.openai_base_url}")
    print(f"  default_model={runtime_config.default_model}")
    print(f"  verbosity={runtime_config.verbosity}")
    print(f"  run_id={run_id}")

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "workflow_path": runtime_config.workflow_path,
        "store_root": runtime_config.store_root,
        "prompt_root": runtime_config.prompt_root,
        "workspace_root": runtime_config.workspace_root,
        "traces_root": runtime_config.traces_root,
        "library_root": runtime_config.library_root,
        "openai_base_url": runtime_config.openai_base_url,
        "default_model": runtime_config.default_model,
        "verbosity": runtime_config.verbosity,
    }
    trace_writer.write_manifest(manifest)

    prompt_registry = PromptRegistry(runtime_config.prompt_root)
    stores = build_store_map(runtime_config.store_root)
    tool_registry = make_tools(runtime_config)

    agents = {
        "llm_agent": OpenAIChatAgent(prompt_registry, runtime_config, tool_registry, trace_writer),
    }

    workflow = load_workflow(runtime_config.workflow_path)
    manifest["workflow_id"] = workflow.id
    trace_writer.write_manifest(manifest)

    runner = WorkflowRunner(
        stores=stores,
        agents=agents,
        runtime_config=runtime_config,
        trace_writer=trace_writer,
        run_id=run_id,
    )
    context = runner.run(workflow)

    manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
    trace_writer.write_manifest(manifest)

    print(f"\nWorkflow complete: {workflow.id}")
    print("Bindings:")
    for name, docs in context.bindings.items():
        print(f"  {name}: {[doc.id for doc in docs]}")


if __name__ == "__main__":
    main()
