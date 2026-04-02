"""Parking engine — warm, cool, status, and optimize operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import anyio

from ollapark.client import Fleet, NodeSnapshot, OllamaNode


@dataclass
class ModelPlacement:
    """A model that should be on a specific node."""
    node_name: str
    model: str
    is_loaded: bool = False
    is_available: bool = True  # model exists on the node


@dataclass
class ParkStatus:
    """Comparison of desired vs actual model placement."""
    placements: list[ModelPlacement] = field(default_factory=list)
    snapshots: list[NodeSnapshot] = field(default_factory=list)
    extra_loaded: list[tuple[str, str]] = field(default_factory=list)  # (node, model) loaded but not in profile

    @property
    def missing(self) -> list[ModelPlacement]:
        return [p for p in self.placements if not p.is_loaded]

    @property
    def warm(self) -> list[ModelPlacement]:
        return [p for p in self.placements if p.is_loaded]

    @property
    def unavailable(self) -> list[ModelPlacement]:
        return [p for p in self.placements if not p.is_available]

    @property
    def all_warm(self) -> bool:
        return all(p.is_loaded for p in self.placements)


@dataclass
class WarmResult:
    model: str
    node: str
    success: bool
    error: str | None = None


@dataclass
class CoolResult:
    model: str
    node: str
    success: bool


@dataclass
class OptimizeSuggestion:
    node_name: str
    vram_gb: float | None
    models: list[str]
    estimated_vram_gb: float


def _model_matches(running_name: str, target: str) -> bool:
    """Check if a running model name matches the target, handling tag defaults."""
    if running_name == target:
        return True
    # "llama3.2:latest" matches "llama3.2"
    if ":" in running_name and running_name.split(":")[0] == target:
        return True
    if ":" not in target and running_name == f"{target}:latest":
        return True
    return False


async def check_status(
    fleet: Fleet,
    profile: dict[str, list[str]],
) -> ParkStatus:
    """Compare desired profile against actual fleet state."""
    snapshots = await fleet.snapshot_all()
    snap_by_name = {s.name: s for s in snapshots}

    placements = []
    for node_name, models in profile.items():
        snap = snap_by_name.get(node_name)
        for model in models:
            p = ModelPlacement(node_name=node_name, model=model)
            if snap and snap.online:
                p.is_loaded = any(_model_matches(r.name, model) for r in snap.running)
                p.is_available = any(_model_matches(a.name, model) for a in snap.available)
            else:
                p.is_loaded = False
                p.is_available = False
            placements.append(p)

    # Find models loaded but not in the profile
    profile_set = {}
    for node_name, models in profile.items():
        profile_set[node_name] = set(models)

    extra = []
    for snap in snapshots:
        if not snap.online:
            continue
        desired = profile_set.get(snap.name, set())
        for r in snap.running:
            if not any(_model_matches(r.name, d) for d in desired):
                extra.append((snap.name, r.name))

    return ParkStatus(placements=placements, snapshots=snapshots, extra_loaded=extra)


async def warm_profile(
    fleet: Fleet,
    profile: dict[str, list[str]],
    on_progress: callable | None = None,
) -> list[WarmResult]:
    """Load all models in a profile onto their designated nodes."""
    # First check what's already loaded
    status = await check_status(fleet, profile)
    results: list[WarmResult] = []

    # Only load what's missing
    to_load = status.missing

    if not to_load:
        return results

    async def _load(placement: ModelPlacement):
        node = fleet.find_node(placement.node_name)
        if not node:
            results.append(WarmResult(
                model=placement.model, node=placement.node_name,
                success=False, error="node not found",
            ))
            return

        if not placement.is_available:
            results.append(WarmResult(
                model=placement.model, node=placement.node_name,
                success=False, error="model not available on node (pull it first)",
            ))
            return

        if on_progress:
            on_progress(placement.node_name, placement.model, "loading")

        ok = await node.load_model(placement.model, keep_alive="-1")
        results.append(WarmResult(
            model=placement.model, node=placement.node_name,
            success=ok, error=None if ok else "load failed",
        ))

        if on_progress:
            on_progress(placement.node_name, placement.model, "done" if ok else "failed")

    # Load models per-node sequentially (VRAM contention), nodes in parallel
    nodes_work: dict[str, list[ModelPlacement]] = {}
    for p in to_load:
        nodes_work.setdefault(p.node_name, []).append(p)

    async def _load_node(placements: list[ModelPlacement]):
        for p in placements:
            await _load(p)

    async with anyio.create_task_group() as tg:
        for node_placements in nodes_work.values():
            tg.start_soon(_load_node, node_placements)

    return results


async def cool_fleet(
    fleet: Fleet,
    node_filter: str | None = None,
) -> list[CoolResult]:
    """Unload all models from fleet (or a specific node)."""
    snapshots = await fleet.snapshot_all()
    results: list[CoolResult] = []

    async def _unload_node(node: OllamaNode, snap: NodeSnapshot):
        for model in snap.running:
            ok = await node.unload_model(model.name)
            results.append(CoolResult(model=model.name, node=snap.name, success=ok))

    async with anyio.create_task_group() as tg:
        for snap, node in zip(snapshots, fleet.nodes):
            if not snap.online or not snap.running:
                continue
            if node_filter and snap.name != node_filter:
                continue
            tg.start_soon(_unload_node, node, snap)

    return results


def optimize_placement(
    snapshots: list[NodeSnapshot],
    node_vram: dict[str, float],
    desired_models: list[str],
) -> list[OptimizeSuggestion]:
    """Simple greedy bin-packing: place models on nodes with most VRAM, largest models first.

    Uses actual VRAM size from running models or estimates from disk size * 1.2.
    """
    # Build model size estimates from what we can see across the fleet
    model_sizes: dict[str, float] = {}
    for snap in snapshots:
        for m in snap.running:
            model_sizes[m.name] = m.size_gb
        for m in snap.available:
            base = m.name
            # Rough estimate: disk size * 1.2 for VRAM overhead
            if base not in model_sizes:
                model_sizes[base] = m.size_gb * 1.2

    # Filter to models we actually want
    to_place = []
    for model in desired_models:
        size = model_sizes.get(model, 0)
        # Try matching without tag
        if size == 0:
            for k, v in model_sizes.items():
                if _model_matches(k, model):
                    size = v
                    break
        to_place.append((model, size))

    # Sort largest first for better packing
    to_place.sort(key=lambda x: x[1], reverse=True)

    # Sort nodes by VRAM descending
    node_names = sorted(node_vram.keys(), key=lambda n: node_vram[n], reverse=True)
    node_remaining = {n: node_vram[n] for n in node_names}
    node_assignments: dict[str, list[str]] = {n: [] for n in node_names}
    node_used: dict[str, float] = {n: 0.0 for n in node_names}

    for model, size in to_place:
        # Find first node that fits
        placed = False
        for node in node_names:
            if node_remaining[node] >= size:
                node_assignments[node].append(model)
                node_remaining[node] -= size
                node_used[node] += size
                placed = True
                break
        if not placed and node_names:
            # Force onto largest node
            node_assignments[node_names[0]].append(model)
            node_used[node_names[0]] += size

    suggestions = []
    for node in node_names:
        if node_assignments[node]:
            suggestions.append(OptimizeSuggestion(
                node_name=node,
                vram_gb=node_vram.get(node),
                models=node_assignments[node],
                estimated_vram_gb=node_used[node],
            ))

    return suggestions
