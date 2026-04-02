"""Async client for managing models across Ollama fleet nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anyio
import httpx


@dataclass
class RunningModel:
    name: str
    size: int  # VRAM bytes
    digest: str
    expires_at: str

    @property
    def size_gb(self) -> float:
        return self.size / (1024**3)


@dataclass
class AvailableModel:
    name: str
    size: int  # disk bytes
    parameter_size: str
    quantization: str

    @property
    def size_gb(self) -> float:
        return self.size / (1024**3)


@dataclass
class NodeSnapshot:
    name: str
    host: str
    online: bool
    latency_ms: float = 0.0
    running: list[RunningModel] = field(default_factory=list)
    available: list[AvailableModel] = field(default_factory=list)
    error: str | None = None

    @property
    def vram_used_gb(self) -> float:
        return sum(m.size_gb for m in self.running)

    @property
    def running_names(self) -> set[str]:
        return {m.name for m in self.running}


class OllamaNode:
    """Client for a single Ollama instance."""

    def __init__(self, name: str, host: str, timeout: float = 15.0):
        self.name = name
        self.host = host.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def close(self):
        await self.client.aclose()

    async def ping(self) -> tuple[bool, float]:
        start = time.monotonic()
        try:
            resp = await self.client.get("/api/version")
            resp.raise_for_status()
            return True, (time.monotonic() - start) * 1000
        except Exception:
            return False, (time.monotonic() - start) * 1000

    async def list_running(self) -> list[RunningModel]:
        resp = await self.client.get("/api/ps")
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            models.append(RunningModel(
                name=m.get("name", ""),
                size=m.get("size", 0),
                digest=m.get("digest", "")[:12],
                expires_at=m.get("expires_at", ""),
            ))
        return models

    async def list_available(self) -> list[AvailableModel]:
        resp = await self.client.get("/api/tags")
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            details = m.get("details", {})
            models.append(AvailableModel(
                name=m["name"],
                size=m.get("size", 0),
                parameter_size=details.get("parameter_size", "?"),
                quantization=details.get("quantization_level", "?"),
            ))
        return models

    async def load_model(self, model: str, keep_alive: str = "-1") -> bool:
        """Load a model into VRAM. keep_alive=-1 means stay forever."""
        try:
            resp = await self.client.post(
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": keep_alive},
                timeout=300.0,
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False

    async def unload_model(self, model: str) -> bool:
        """Unload a model from VRAM by setting keep_alive=0."""
        try:
            resp = await self.client.post(
                "/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30.0,
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False

    async def snapshot(self) -> NodeSnapshot:
        online, latency = await self.ping()
        snap = NodeSnapshot(name=self.name, host=self.host, online=online, latency_ms=latency)
        if not online:
            snap.error = "unreachable"
            return snap
        try:
            snap.running = await self.list_running()
        except Exception as e:
            snap.error = str(e)
        try:
            snap.available = await self.list_available()
        except Exception:
            pass
        return snap


class Fleet:
    """Manages multiple Ollama nodes."""

    def __init__(self, nodes: list[OllamaNode]):
        self.nodes = nodes

    async def close(self):
        for node in self.nodes:
            await node.close()

    async def snapshot_all(self) -> list[NodeSnapshot]:
        results: list[NodeSnapshot] = [None] * len(self.nodes)  # type: ignore

        async def _snap(i: int, node: OllamaNode):
            results[i] = await node.snapshot()

        async with anyio.create_task_group() as tg:
            for i, node in enumerate(self.nodes):
                tg.start_soon(_snap, i, node)

        return results

    def find_node(self, name: str) -> OllamaNode | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None
