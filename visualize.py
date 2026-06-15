import json
import os
from pathlib import Path

from leaderboard import load_results, build_leaderboard, leaderboard_by_task, leaderboard_by_language


def plot_overall_leaderboard(results_dir: str = "results", output_dir: str = "plots", fmt: str = "png"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(output_dir, exist_ok=True)
    rows = build_leaderboard(results_dir)
    if not rows:
        print("No results to plot.")
        return
    metric_cols = [k for k in rows[0] if k.endswith("_mean")]
    models = [r["model"] for r in rows]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 6), squeeze=False)
    colors = list(plt.cm.Set2.colors)
    for ax, metric in zip(axes[0], metric_cols):
        values = [r.get(metric, 0) for r in rows]
        bars = ax.barh(models, values, color=[colors[i % len(colors)] for i in range(len(models))])
        ax.set_xlabel(metric.replace("_mean", "").upper())
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    fig.suptitle("Swiss VLM Benchmark - Overall Leaderboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"overall_leaderboard.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_task_comparison(results_dir: str = "results", output_dir: str = "plots", fmt: str = "png"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(output_dir, exist_ok=True)
    task_boards = leaderboard_by_task(results_dir)
    if not task_boards:
        print("No results to plot.")
        return
    all_models = set()
    for rows in task_boards.values():
        for r in rows:
            all_models.add(r["model"])
    models = sorted(all_models)
    tasks = sorted(task_boards.keys())
    metric = "anls_mean"
    fig, ax = plt.subplots(figsize=(12, 6))
    import numpy as np
    x = np.arange(len(tasks))
    width = 0.8 / max(len(models), 1)
    colors = list(plt.cm.Set2.colors)
    for i, model in enumerate(models):
        values = []
        for task in tasks:
            task_rows = task_boards.get(task, [])
            val = next((r.get(metric, 0) for r in task_rows if r["model"] == model), 0)
            values.append(val)
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])
    ax.set_xlabel("Task")
    ax.set_ylabel(metric.replace("_mean", "").upper())
    ax.set_title("Swiss VLM Benchmark - Task Comparison")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, f"task_comparison.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_language_radar(results_dir: str = "results", output_dir: str = "plots", fmt: str = "png"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    lang_boards = leaderboard_by_language(results_dir)
    if not lang_boards:
        print("No results to plot.")
        return
    all_models = set()
    for rows in lang_boards.values():
        for r in rows:
            all_models.add(r["model"])
    models = sorted(all_models)
    languages = sorted(lang_boards.keys())
    metric = "anls_mean"
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(languages), endpoint=False).tolist()
    angles += angles[:1]
    colors = list(plt.cm.Set2.colors)
    for i, model in enumerate(models):
        values = []
        for lang in languages:
            lang_rows = lang_boards.get(lang, [])
            val = next((r.get(metric, 0) for r in lang_rows if r["model"] == model), 0)
            values.append(val)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Performance by Language", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    path = os.path.join(output_dir, f"language_radar.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency_comparison(results_dir: str = "results", output_dir: str = "plots", fmt: str = "png"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(output_dir, exist_ok=True)
    rows = build_leaderboard(results_dir)
    if not rows:
        print("No results to plot.")
        return
    models = [r["model"] for r in rows]
    latencies = [r.get("avg_latency_ms", 0) for r in rows]
    colors = list(plt.cm.Set2.colors)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, latencies, color=[colors[i % len(colors)] for i in range(len(models))])
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Swiss VLM Benchmark - Inference Latency")
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}ms", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"latency_comparison.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_case_heatmap(results_dir: str = "results", output_dir: str = "plots", fmt: str = "png"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    all_results = load_results(results_dir)
    if not all_results:
        print("No results to plot.")
        return
    models = sorted(all_results.keys())
    all_ids = []
    for results in all_results.values():
        for r in results:
            if r["test_id"] not in all_ids:
                all_ids.append(r["test_id"])
    matrix = np.zeros((len(models), len(all_ids)))
    for i, model in enumerate(models):
        for r in all_results[model]:
            if r["test_id"] in all_ids:
                j = all_ids.index(r["test_id"])
                matrix[i, j] = r["metrics"].get("anls", r["metrics"].get("exact_match", 0))
    fig, ax = plt.subplots(figsize=(max(12, len(all_ids) * 0.8), max(4, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_ids)))
    ax.set_xticklabels(all_ids, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    fig.colorbar(im, ax=ax, label="Score")
    ax.set_title("Per-Case Score Heatmap")
    plt.tight_layout()
    path = os.path.join(output_dir, f"per_case_heatmap.{fmt}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_interactive_leaderboard(results_dir: str = "results", output_dir: str = "plots"):
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("plotly not installed. Install with: pip install plotly")
        return
    os.makedirs(output_dir, exist_ok=True)
    rows = build_leaderboard(results_dir)
    if not rows:
        print("No results to plot.")
        return
    metric_cols = [k for k in rows[0] if k.endswith("_mean")]
    models = [r["model"] for r in rows]
    fig = go.Figure()
    for metric in metric_cols:
        values = [r.get(metric, 0) for r in rows]
        fig.add_trace(go.Bar(name=metric.replace("_mean", "").upper(), x=models, y=values))
    fig.update_layout(
        barmode="group",
        title="Swiss VLM Benchmark - Interactive Leaderboard",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        template="plotly_white",
    )
    path = os.path.join(output_dir, "interactive_leaderboard.html")
    fig.write_html(path)
    print(f"Saved: {path}")

    task_boards = leaderboard_by_task(results_dir)
    if task_boards:
        fig2 = go.Figure()
        for task, task_rows in task_boards.items():
            for r in task_rows:
                fig2.add_trace(go.Bar(
                    name=f"{r['model']}",
                    x=[task.replace("_", " ").title()],
                    y=[r.get("anls_mean", 0)],
                    legendgroup=r["model"],
                    showlegend=task == sorted(task_boards.keys())[0],
                ))
        fig2.update_layout(
            barmode="group",
            title="Task-Level Performance",
            yaxis_range=[0, 1.05],
            template="plotly_white",
        )
        path2 = os.path.join(output_dir, "task_performance.html")
        fig2.write_html(path2)
        print(f"Saved: {path2}")


def generate_all_plots(results_dir: str = "results", output_dir: str = "plots", fmt: str = "png"):
    print("Generating plots...")
    plot_overall_leaderboard(results_dir, output_dir, fmt)
    plot_task_comparison(results_dir, output_dir, fmt)
    plot_language_radar(results_dir, output_dir, fmt)
    plot_latency_comparison(results_dir, output_dir, fmt)
    plot_per_case_heatmap(results_dir, output_dir, fmt)
    plot_interactive_leaderboard(results_dir, output_dir)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Swiss VLM Benchmark Results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--plot", choices=["all", "overall", "task", "radar", "latency", "heatmap", "interactive"], default="all")
    args = parser.parse_args()

    plot_map = {
        "overall": lambda: plot_overall_leaderboard(args.results_dir, args.output_dir, args.format),
        "task": lambda: plot_task_comparison(args.results_dir, args.output_dir, args.format),
        "radar": lambda: plot_language_radar(args.results_dir, args.output_dir, args.format),
        "latency": lambda: plot_latency_comparison(args.results_dir, args.output_dir, args.format),
        "heatmap": lambda: plot_per_case_heatmap(args.results_dir, args.output_dir, args.format),
        "interactive": lambda: plot_interactive_leaderboard(args.results_dir, args.output_dir),
    }
    if args.plot == "all":
        generate_all_plots(args.results_dir, args.output_dir, args.format)
    else:
        plot_map[args.plot]()
