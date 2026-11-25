# ollapark

Model parking manager for Ollama fleets. Define which models stay warm on which nodes, then warm up or cool down your fleet with a single command.

## Why

When you run multiple Ollama nodes with different VRAM budgets, you end up manually loading and unloading models every session. ollapark automates this with **parking profiles** — named configurations that specify exactly which models should be loaded where.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Generate a starter config
ollapark init

# Edit park.yml with your nodes and profiles
vim park.yml

# Check what's loaded vs what should be
ollapark status

# Warm up your default profile
ollapark warm

# Switch to a different profile
ollapark warm coding

# Done for the day? Free all VRAM
ollapark cool
```

## Configuration

`park.yml` (searched in `./park.yml` then `~/.config/ollapark/park.yml`):

```yaml
nodes:
  - name: corsair
    host: http://192.168.1.100:11434
    vram_gb: 128
  - name: mac-studio
    host: http://192.168.1.101:11434
    vram_gb: 64
  - name: alienware
    host: http://192.168.1.102:11434
    vram_gb: 24

profiles:
  default:
    corsair:
      - deepseek-r1:70b
      - qwen2.5-coder:32b
    mac-studio:
      - llama3.3:70b
    alienware:
      - mistral:7b
      - llava:13b

  coding:
    corsair:
      - qwen2.5-coder:32b
      - deepseek-coder-v2:33b
    mac-studio:
      - qwen2.5-coder:14b
    alienware:
      - qwen2.5-coder:7b

  filmwork:
    corsair:
      - llama3.3:70b
    mac-studio:
      - llava:34b
    alienware:
      - llava:13b
```

## Commands

| Command | Description |
|---------|-------------|
| `ollapark status [profile]` | Compare loaded models against a profile |
| `ollapark warm [profile]` | Load all models in a profile (skips already-loaded) |
| `ollapark cool [--node NAME]` | Unload all models from VRAM |
| `ollapark fleet` | Raw fleet overview (all nodes + loaded models) |
| `ollapark optimize` | Suggest optimal placement based on VRAM budgets |
| `ollapark profiles` | List available profiles |
| `ollapark init` | Generate starter park.yml |

## How It Works

- **warm** sends `keep_alive: -1` to each model (stays in VRAM until explicitly unloaded)
- **cool** sends `keep_alive: 0` to all running models (immediately frees VRAM)
- **optimize** uses greedy bin-packing to suggest model placement given VRAM constraints
- Models on different nodes load in parallel; models on the same node load sequentially to avoid VRAM contention

## Ecosystem

Part of the Ollama fleet toolkit:
- [ollafleet](https://github.com/Obrais-cloud/ollafleet) — Fleet status + management CLI
- ollaroute — Smart model routing proxy
- [ollametrics](https://github.com/Obrais-cloud/ollametrics) — Prometheus exporter

## License

MIT
