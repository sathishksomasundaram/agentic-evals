"""Build three charts from existing spike / v0 data.

Outputs go to docs/v0/charts/{name}.png.

Charts:
  1. qwen25_cliff_step  — spike 0008 step function (PASS/FAIL vs buffer ratio)
  2. cross_model_cliff  — spike 0008 + 0009 combined, three models overlaid
  3. v0_verdict_heatmap — 6 models × 6 cells, color-coded

Run with:
    uv run agentic-evals exp-001 charts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = (
    ROOT / "docs" / "experiments" / "exp-001-mlx-kv-compression-toolcalling" / "results" / "charts"
)

# Color palette (consistent across charts)
COLOR_PASS = "#2c8a3e"  # green
COLOR_HALLUCINATE = "#d97c1e"  # dark orange — Qwen-2.5 style "fluent wrong"
COLOR_COLLAPSE = "#c53030"  # red — token garbage
COLOR_LOST_FORMAT = "#e9b941"  # yellow-orange — answered without tool
COLOR_ALT_FORMAT = "#3182ce"  # blue — Gemma's alt schema
COLOR_NEUTRAL = "#7a7a7a"  # grey — error / N/A


# --------------------------------------------------------------------------- #
# Chart 1 — Qwen-2.5 cliff step function (spike 0008)                         #
# --------------------------------------------------------------------------- #
def chart_qwen25_step() -> None:
    spike_data = json.loads(
        (
            ROOT
            / "docs"
            / "experiments"
            / "exp-001-mlx-kv-compression-toolcalling"
            / "spikes"
            / "0008-cliff-fine-sweep"
            / "raw-results.json"
        ).read_text()
    )
    runs = spike_data["runs"]

    ratios = [r["buffer_ratio"] for r in runs]
    verdicts = [r["verdict"] for r in runs]
    times = [r["time_s"] for r in runs]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot each ratio as a colored dot (PASS = green, FAIL = red) at y=1/0
    y_pass_fail = [1 if v == "PASS" else 0 for v in verdicts]
    for x, y, t, v in zip(ratios, y_pass_fail, times, verdicts, strict=True):
        color = COLOR_PASS if v == "PASS" else COLOR_LOST_FORMAT
        ax.scatter(x, y, c=color, s=200, zorder=5, edgecolors="black", linewidths=1.2, label=v)
        ax.annotate(
            f"buf={int(x * 1966)}\n{t:.2f}s",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12 if y == 1 else -28),
            ha="center",
            fontsize=9,
        )

    # Vertical line at the cliff (between 0.5000 and 0.5051)
    cliff = 0.5025
    ax.axvline(cliff, color="black", linestyle="--", alpha=0.5, zorder=1)
    ax.text(
        cliff,
        0.5,
        " cliff at ratio ≈ 0.505",
        rotation=90,
        va="center",
        ha="left",
        color="black",
        fontsize=10,
    )

    ax.set_xlim(0.490, 0.525)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FAIL", "PASS"])
    ax.set_xlabel("buffer_size / prompt_tokens  (1966-token prompt, K3/V2)")
    ax.set_ylabel("tool-call verdict")
    ax.set_title(
        "TurboQuant buffer-cliff on Qwen-2.5-7B-Instruct (MLX) — sharp transition at ratio ≈ 0.505",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)

    # Custom legend
    pass_patch = mpatches.Patch(color=COLOR_PASS, label="PASS — perfect tool call")
    fail_patch = mpatches.Patch(color=COLOR_LOST_FORMAT, label="FAIL — hallucinated / wrong-schema")
    ax.legend(handles=[pass_patch, fail_patch], loc="center right")

    fig.tight_layout()
    out = OUT_DIR / "01_qwen25_cliff_step.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Chart 2 — Cross-model cliff comparison (spikes 0008 + 0009)                 #
# --------------------------------------------------------------------------- #
def chart_cross_model_cliff() -> None:
    # Qwen-2.5 from spike 0008
    qwen = json.loads(
        (
            ROOT
            / "docs"
            / "experiments"
            / "exp-001-mlx-kv-compression-toolcalling"
            / "spikes"
            / "0008-cliff-fine-sweep"
            / "raw-results.json"
        ).read_text()
    )
    qwen_ratios = [r["buffer_ratio"] for r in qwen["runs"]]
    qwen_pass = [1 if r["verdict"] == "PASS" else 0 for r in qwen["runs"]]

    # Llama-3.2 and Phi-4-mini from spike 0009
    spike9_dir = (
        ROOT
        / "docs"
        / "experiments"
        / "exp-001-mlx-kv-compression-toolcalling"
        / "spikes"
        / "0009-per-model-cliff"
    )
    llama_data = json.loads(
        (spike9_dir / "mlx-community-llama-3.2-3b-instruct-4bit" / "raw-results.json").read_text()
    )
    phi_data = json.loads(
        (spike9_dir / "mlx-community-phi-4-mini-instruct-4bit" / "raw-results.json").read_text()
    )

    def extract_long(spike: dict[str, Any]) -> tuple[list[float], list[int]]:
        long_runs = [r for r in spike["rows"] if r["prompt_label"] == "long" and r["ratio"]]
        return (
            [r["ratio"] for r in long_runs],
            [1 if r["verdict"] == "PASS" else 0 for r in long_runs],
        )

    llama_r, llama_p = extract_long(llama_data)
    phi_r, phi_p = extract_long(phi_data)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Draw each model as a step plot with jittered y values for visibility
    def step(x: list[float], y: list[int], label: str, color: str, y_offset: float) -> None:
        y_jitter = [v + y_offset for v in y]
        ax.plot(
            x, y_jitter, marker="o", markersize=10, linewidth=2, color=color, label=label, zorder=3
        )
        # Mark transitions
        for i in range(1, len(y)):
            if y[i] != y[i - 1]:
                mid = (x[i] + x[i - 1]) / 2
                ax.axvspan(x[i - 1], x[i], alpha=0.1, color=color)
                ax.text(mid, y_offset + 1.05, "cliff", fontsize=7, ha="center", color=color)

    step(qwen_ratios, qwen_pass, "Qwen-2.5-7B-Instruct (spike 0008, long prompt)", "#1f6feb", 0.0)
    step(llama_r, llama_p, "Llama-3.2-3B-Instruct (spike 0009, long prompt)", "#9333ea", 0.08)
    step(phi_r, phi_p, "Phi-4-mini-instruct (spike 0009, long prompt)", "#dc2626", 0.16)

    ax.set_xlim(0.48, 0.90)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([0.08, 1.08])
    ax.set_yticklabels(["FAIL", "PASS"])
    ax.set_xlabel("buffer_size / prompt_tokens")
    ax.set_ylabel("tool-call verdict")
    ax.set_title(
        "TurboQuant buffer-cliff is universal: ratio ≈ 0.50-0.55 across three instruct models",
        fontsize=12,
    )
    ax.axvspan(0.50, 0.55, alpha=0.10, color="black", zorder=1)
    ax.text(0.525, -0.25, "universal cliff window", fontsize=9, ha="center", color="black")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "02_cross_model_cliff.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Chart 3 — v0 verdict heatmap (6 models × 6 cells)                           #
# --------------------------------------------------------------------------- #
def chart_v0_heatmap() -> None:
    jsonl_path = (
        ROOT
        / "docs"
        / "experiments"
        / "exp-001-mlx-kv-compression-toolcalling"
        / "results"
        / "raw-results.jsonl"
    )
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

    # Take the LATEST row per (model, prompt, config) since multiple runs accumulated
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r.get("model_id", "?"), r.get("prompt_label", "?"), r.get("config", "?"))
        latest[key] = r

    models = [
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Qwen3-4B-Instruct-2507-6bit",
        "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "mlx-community/Phi-4-mini-instruct-4bit",
        "mlx-community/gemma-3-4b-it-4bit",
    ]
    model_labels = [m.replace("mlx-community/", "").replace("-Instruct", "-Inst") for m in models]

    cells = [
        ("short", "baseline"),
        ("short", "tq_ratio_0.5"),
        ("short", "tq_ratio_1.0"),
        ("long", "baseline"),
        ("long", "tq_ratio_0.5"),
        ("long", "tq_ratio_1.0"),
    ]
    cell_labels = [
        "short\nbaseline",
        "short\nTQ r=0.5",
        "short\nTQ r=1.0",
        "long\nbaseline",
        "long\nTQ r=0.5",
        "long\nTQ r=1.0",
    ]

    # Map verdict to color; also annotate with specific failure mode for the blog
    def verdict_color_and_label(verdict: str, output: str) -> tuple[str, str]:
        if verdict == "PASS":
            return COLOR_PASS, "PASS"
        if "!!!" in output or output.count("!") > 50:
            return COLOR_COLLAPSE, "COLLAPSE"
        if "bó" in output or "EXEX" in output or output.startswith('{"}') or "™\n" in output:
            return COLOR_COLLAPSE, "COLLAPSE"
        if "```tool" in output:
            return COLOR_ALT_FORMAT, "ALT FMT"
        if verdict == "LOST_FORMAT":
            return COLOR_LOST_FORMAT, "LOST FMT"
        if "weather" in output.lower() and '"tool"' not in output:
            return COLOR_HALLUCINATE, "HALLUC"
        return COLOR_NEUTRAL, verdict[:8]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, model in enumerate(models):
        for j, (pl, cfg) in enumerate(cells):
            row = latest.get((model, pl, cfg))
            if row is None:
                color, label = COLOR_NEUTRAL, "—"
            else:
                color, label = verdict_color_and_label(row["verdict"], row.get("output", ""))
            rect = mpatches.Rectangle(
                (j, len(models) - 1 - i), 1, 1, facecolor=color, edgecolor="white", linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                j + 0.5,
                len(models) - 1 - i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    ax.set_xlim(0, len(cells))
    ax.set_ylim(0, len(models))
    ax.set_xticks([x + 0.5 for x in range(len(cells))])
    ax.set_xticklabels(cell_labels, fontsize=9)
    ax.set_yticks([y + 0.5 for y in range(len(models))])
    ax.set_yticklabels(reversed(model_labels), fontsize=9)
    ax.set_title(
        "Open-Model Benchmark v0 — tool-calling verdict heatmap\n"
        "(6 models × 2 prompts × 3 configs · greedy · Apple Silicon · yzamari TQ K3/V2)",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Legend
    legend_items = [
        mpatches.Patch(color=COLOR_PASS, label="PASS — correct tool call"),
        mpatches.Patch(color=COLOR_HALLUCINATE, label="HALLUC — fluent hallucination (dangerous)"),
        mpatches.Patch(color=COLOR_LOST_FORMAT, label="LOST FMT — answered, no tool call"),
        mpatches.Patch(color=COLOR_COLLAPSE, label="COLLAPSE — token garbage (loud failure)"),
        mpatches.Patch(color=COLOR_ALT_FORMAT, label="ALT FMT — model's own tool schema"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    # Hide the spine
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    out = OUT_DIR / "03_v0_verdict_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"==> Building charts → {OUT_DIR}")
    chart_qwen25_step()
    chart_cross_model_cliff()
    chart_v0_heatmap()
    print("\n==> Done. Three charts written.")


if __name__ == "__main__":
    main()
