"""Configuration loader for ollapark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ollapark.client import OllamaNode

CONFIG_SEARCH_PATHS = [
    Path("park.yml"),
    Path.home() / ".config" / "ollapark" / "park.yml",
]

DEFAULT_CONFIG = """\
# ollapark — Model parking configuration
# Define your fleet nodes and parking profiles.
# Profiles specify which models to keep warm on which nodes.

nodes:
  - name: local
    host: http://localhost:11434
    # vram_gb: 24  # optional: used by 'optimize' command

# Profiles define model placement across your fleet.
# Use 'ollapark warm <profile>' to load models.
# Use 'ollapark cool' to unload everything.
profiles:
  default:
    local:
      - llama3.2:latest

  # Example multi-node profile:
  # coding:
  #   workstation:
  #     - qwen2.5-coder:32b
  #     - deepseek-r1:70b
  #   laptop:
  #     - qwen2.5-coder:7b
"""


@dataclass
class NodeDef:
    name: str
    host: str
    vram_gb: float | None = None


@dataclass
class ParkConfig:
    nodes: list[NodeDef] = field(default_factory=list)
    profiles: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def profile_names(self) -> list[str]:
        return list(self.profiles.keys())

    def get_profile(self, name: str) -> dict[str, list[str]]:
        if name not in self.profiles:
            raise KeyError(f"Profile '{name}' not found. Available: {', '.join(self.profile_names())}")
        return self.profiles[name]

    def build_fleet(self) -> tuple[list[OllamaNode], dict[str, NodeDef]]:
        nodes = []
        defs = {}
        for nd in self.nodes:
            nodes.append(OllamaNode(name=nd.name, host=nd.host))
            defs[nd.name] = nd
        return nodes, defs


def find_config(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found: {explicit}")

    for p in CONFIG_SEARCH_PATHS:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No park.yml found. Create one in the current directory or "
        "~/.config/ollapark/park.yml. Run 'ollapark init' to generate one."
    )


def load_config(path: Path) -> ParkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "nodes" not in data:
        raise ValueError(f"Invalid config: {path} must have a 'nodes' list")

    nodes = []
    for entry in data["nodes"]:
        nodes.append(NodeDef(
            name=entry["name"],
            host=entry["host"].rstrip("/"),
            vram_gb=entry.get("vram_gb"),
        ))

    profiles: dict[str, dict[str, list[str]]] = {}
    for pname, pdata in data.get("profiles", {}).items():
        profiles[pname] = {}
        for node_name, models in pdata.items():
            profiles[pname][node_name] = models if isinstance(models, list) else [models]

    return ParkConfig(nodes=nodes, profiles=profiles)
