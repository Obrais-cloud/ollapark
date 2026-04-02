"""CLI interface for ollapark."""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ollapark.client import Fleet, OllamaNode
from ollapark.config import DEFAULT_CONFIG, ParkConfig, find_config, load_config
from ollapark.engine import (
    check_status,
    cool_fleet,
    optimize_placement,
    warm_profile,
)

console = Console()


def _load(config_path: str | None) -> ParkConfig:
    try:
        path = find_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    try:
        return load_config(path)
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)


def _run(coro):
    return anyio.from_thread.run(coro) if False else anyio.run(coro)


@click.group()
@click.option("-c", "--config", "config_path", default=None, help="Path to park.yml")
@click.pass_context
def cli(ctx, config_path):
    """ollapark — Model parking manager for Ollama fleets."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.command()
@click.argument("profile", default="default")
@click.pass_context
def status(ctx, profile):
    """Show what's loaded vs what should be loaded."""
    cfg = _load(ctx.obj["config_path"])

    try:
        prof = cfg.get_profile(profile)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    nodes, _ = cfg.build_fleet()
    fleet = Fleet(nodes)

    async def _run():
        try:
            return await check_status(fleet, prof)
        finally:
            await fleet.close()

    st = anyio.run(_run)

    # Node overview
    node_table = Table(title=f"Fleet Status (profile: {profile})", show_lines=True)
    node_table.add_column("Node", style="bold")
    node_table.add_column("Status")
    node_table.add_column("Latency")
    node_table.add_column("VRAM Used")
    node_table.add_column("Models Loaded")

    for snap in st.snapshots:
        status_str = "[green]online[/green]" if snap.online else "[red]offline[/red]"
        latency_str = f"{snap.latency_ms:.0f}ms" if snap.online else "-"
        vram_str = f"{snap.vram_used_gb:.1f} GB" if snap.online else "-"
        loaded = ", ".join(m.name for m in snap.running) if snap.running else "[dim]none[/dim]"
        node_table.add_row(snap.name, status_str, latency_str, vram_str, loaded)

    console.print(node_table)
    console.print()

    # Parking status
    park_table = Table(title="Parking Status", show_lines=True)
    park_table.add_column("Node", style="bold")
    park_table.add_column("Model")
    park_table.add_column("Status")

    for p in st.placements:
        if p.is_loaded:
            status_str = "[green]WARM[/green]"
        elif not p.is_available:
            status_str = "[red]NOT FOUND[/red] (pull first)"
        else:
            status_str = "[yellow]COLD[/yellow]"
        park_table.add_row(p.node_name, p.model, status_str)

    console.print(park_table)

    if st.extra_loaded:
        console.print()
        extra_table = Table(title="Extra Models (loaded but not in profile)", show_lines=True)
        extra_table.add_column("Node", style="bold")
        extra_table.add_column("Model")
        for node, model in st.extra_loaded:
            extra_table.add_row(node, model)
        console.print(extra_table)

    if st.all_warm:
        console.print("\n[green bold]All models parked and warm.[/green bold]")
    else:
        missing = len(st.missing)
        console.print(f"\n[yellow]{missing} model(s) need warming.[/yellow] Run: ollapark warm {profile}")


@cli.command()
@click.argument("profile", default="default")
@click.pass_context
def warm(ctx, profile):
    """Load all models in a profile onto their designated nodes."""
    cfg = _load(ctx.obj["config_path"])

    try:
        prof = cfg.get_profile(profile)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    nodes, _ = cfg.build_fleet()
    fleet = Fleet(nodes)

    total = sum(len(models) for models in prof.values())
    console.print(f"[bold]Warming profile '{profile}' — {total} model(s) across {len(prof)} node(s)[/bold]\n")

    def on_progress(node: str, model: str, state: str):
        if state == "loading":
            console.print(f"  [cyan]{node}[/cyan] loading [bold]{model}[/bold]...")
        elif state == "done":
            console.print(f"  [cyan]{node}[/cyan] [green]{model} loaded[/green]")
        elif state == "failed":
            console.print(f"  [cyan]{node}[/cyan] [red]{model} FAILED[/red]")

    async def _run():
        try:
            return await warm_profile(fleet, prof, on_progress=on_progress)
        finally:
            await fleet.close()

    results = anyio.run(_run)

    if not results:
        console.print("[green]All models already warm. Nothing to do.[/green]")
        return

    ok = sum(1 for r in results if r.success)
    fail = sum(1 for r in results if not r.success)

    console.print()
    if fail == 0:
        console.print(f"[green bold]Done. {ok} model(s) loaded successfully.[/green bold]")
    else:
        console.print(f"[yellow]{ok} loaded, {fail} failed.[/yellow]")
        for r in results:
            if not r.success:
                console.print(f"  [red]{r.node}/{r.model}: {r.error}[/red]")


@cli.command()
@click.option("-n", "--node", default=None, help="Only cool a specific node")
@click.pass_context
def cool(ctx, node):
    """Unload all models from VRAM (or a specific node)."""
    cfg = _load(ctx.obj["config_path"])
    nodes, _ = cfg.build_fleet()
    fleet = Fleet(nodes)

    scope = f"node '{node}'" if node else "all nodes"
    console.print(f"[bold]Cooling {scope}...[/bold]\n")

    async def _run():
        try:
            return await cool_fleet(fleet, node_filter=node)
        finally:
            await fleet.close()

    results = anyio.run(_run)

    if not results:
        console.print("[dim]No models were loaded. Nothing to unload.[/dim]")
        return

    for r in results:
        icon = "[green]unloaded[/green]" if r.success else "[red]failed[/red]"
        console.print(f"  [cyan]{r.node}[/cyan] {r.model} — {icon}")

    ok = sum(1 for r in results if r.success)
    console.print(f"\n[green bold]Done. {ok} model(s) unloaded.[/green bold]")


@cli.command()
@click.pass_context
def optimize(ctx):
    """Suggest optimal model placement based on VRAM budgets."""
    cfg = _load(ctx.obj["config_path"])
    nodes, defs = cfg.build_fleet()
    fleet = Fleet(nodes)

    # Collect VRAM budgets
    vram_map = {}
    for nd in cfg.nodes:
        if nd.vram_gb is None:
            console.print(f"[yellow]Warning: no vram_gb set for '{nd.name}'. Set it in park.yml for better suggestions.[/yellow]")
            vram_map[nd.name] = 999.0  # unlimited
        else:
            vram_map[nd.name] = nd.vram_gb

    # Collect all unique models across all profiles
    all_models = set()
    for prof in cfg.profiles.values():
        for models in prof.values():
            all_models.update(models)

    if not all_models:
        console.print("[dim]No profiles defined. Nothing to optimize.[/dim]")
        return

    async def _run():
        try:
            return await fleet.snapshot_all()
        finally:
            await fleet.close()

    snapshots = anyio.run(_run)
    suggestions = optimize_placement(snapshots, vram_map, list(all_models))

    if not suggestions:
        console.print("[dim]No suggestions.[/dim]")
        return

    console.print(Panel("[bold]Optimized Model Placement[/bold]", style="blue"))

    for s in suggestions:
        vram_str = f"{s.vram_gb:.0f} GB" if s.vram_gb and s.vram_gb < 900 else "unlimited"
        console.print(f"\n[bold cyan]{s.node_name}[/bold cyan] (VRAM: {vram_str})")
        console.print(f"  Estimated usage: [yellow]{s.estimated_vram_gb:.1f} GB[/yellow]")
        for m in s.models:
            console.print(f"    - {m}")

    # Generate profile YAML
    console.print("\n[bold]Suggested profile (add to park.yml):[/bold]\n")
    console.print("[dim]  optimized:[/dim]")
    for s in suggestions:
        console.print(f"[dim]    {s.node_name}:[/dim]")
        for m in s.models:
            console.print(f"[dim]      - {m}[/dim]")


@cli.command()
def profiles():
    """List available parking profiles."""
    try:
        path = find_config(None)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    cfg = load_config(path)
    console.print(f"[bold]Profiles in {path}:[/bold]\n")

    for name, prof in cfg.profiles.items():
        total = sum(len(models) for models in prof.values())
        nodes = ", ".join(prof.keys())
        console.print(f"  [cyan]{name}[/cyan] — {total} model(s) on {nodes}")


@cli.command()
@click.option("-o", "--output", default="park.yml", help="Output path")
def init(output):
    """Generate a starter park.yml configuration."""
    path = Path(output)
    if path.exists():
        console.print(f"[yellow]{path} already exists. Not overwriting.[/yellow]")
        sys.exit(1)

    path.write_text(DEFAULT_CONFIG)
    console.print(f"[green]Created {path}[/green]")
    console.print("Edit it with your fleet nodes and profiles, then run:")
    console.print("  [bold]ollapark status[/bold]")


@cli.command()
@click.pass_context
def fleet(ctx):
    """Show raw fleet status (all nodes, all loaded models)."""
    cfg = _load(ctx.obj["config_path"])
    nodes, _ = cfg.build_fleet()
    fleet_obj = Fleet(nodes)

    async def _run():
        try:
            return await fleet_obj.snapshot_all()
        finally:
            await fleet_obj.close()

    snapshots = anyio.run(_run)

    table = Table(title="Fleet Overview", show_lines=True)
    table.add_column("Node", style="bold")
    table.add_column("Status")
    table.add_column("Latency")
    table.add_column("VRAM Used")
    table.add_column("Loaded Models")
    table.add_column("Available Models")

    for snap in snapshots:
        status_str = "[green]online[/green]" if snap.online else "[red]offline[/red]"
        latency = f"{snap.latency_ms:.0f}ms" if snap.online else "-"
        vram = f"{snap.vram_used_gb:.1f} GB" if snap.online else "-"
        loaded = "\n".join(f"{m.name} ({m.size_gb:.1f}GB)" for m in snap.running) or "[dim]none[/dim]"
        available = str(len(snap.available)) if snap.online else "-"
        table.add_row(snap.name, status_str, latency, vram, loaded, available)

    console.print(table)


if __name__ == "__main__":
    cli()
