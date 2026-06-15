import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark import EvalResult, aggregate_metrics, load_test_cases, TASK_TYPES


def load_results(results_dir: str = "results") -> dict[str, list[dict]]:
    all_results = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return all_results
    for f in sorted(results_path.glob("*_results.json")):
        model_name = f.stem.replace("_results", "")
        with open(f) as fh:
            all_results[model_name] = json.load(fh)
    return all_results


def build_leaderboard(results_dir: str = "results") -> list[dict]:
    all_results = load_results(results_dir)
    rows = []
    for model_name, results in all_results.items():
        metrics_list = [r["metrics"] for r in results if r.get("metrics")]
        agg = aggregate_metrics(metrics_list)
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        row = {"model": model_name, "num_cases": len(results)}
        for metric_name, stats in agg.items():
            row[f"{metric_name}_mean"] = round(stats["mean"], 4)
        if latencies:
            row["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
        rows.append(row)
    rows.sort(key=lambda r: r.get("anls_mean", r.get("exact_match_mean", 0)), reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


def leaderboard_by_task(results_dir: str = "results") -> dict[str, list[dict]]:
    all_results = load_results(results_dir)
    by_task: dict[str, dict[str, list[dict]]] = {}
    for model_name, results in all_results.items():
        for r in results:
            task = r.get("task", "unknown")
            by_task.setdefault(task, {}).setdefault(model_name, []).append(r)

    task_leaderboards = {}
    for task, models in by_task.items():
        rows = []
        for model_name, results in models.items():
            metrics_list = [r["metrics"] for r in results if r.get("metrics")]
            agg = aggregate_metrics(metrics_list)
            row = {"model": model_name, "num_cases": len(results)}
            for metric_name, stats in agg.items():
                row[f"{metric_name}_mean"] = round(stats["mean"], 4)
            rows.append(row)
        rows.sort(key=lambda r: r.get("anls_mean", r.get("exact_match_mean", 0)), reverse=True)
        for i, row in enumerate(rows):
            row["rank"] = i + 1
        task_leaderboards[task] = rows
    return task_leaderboards


def leaderboard_by_language(results_dir: str = "results") -> dict[str, list[dict]]:
    all_results = load_results(results_dir)
    by_lang: dict[str, dict[str, list[dict]]] = {}
    for model_name, results in all_results.items():
        for r in results:
            lang = r.get("language", "unknown")
            by_lang.setdefault(lang, {}).setdefault(model_name, []).append(r)

    lang_leaderboards = {}
    for lang, models in by_lang.items():
        rows = []
        for model_name, results in models.items():
            metrics_list = [r["metrics"] for r in results if r.get("metrics")]
            agg = aggregate_metrics(metrics_list)
            row = {"model": model_name, "num_cases": len(results)}
            for metric_name, stats in agg.items():
                row[f"{metric_name}_mean"] = round(stats["mean"], 4)
            rows.append(row)
        rows.sort(key=lambda r: r.get("anls_mean", r.get("exact_match_mean", 0)), reverse=True)
        for i, row in enumerate(rows):
            row["rank"] = i + 1
        lang_leaderboards[lang] = rows
    return lang_leaderboards


def print_leaderboard(rows: list[dict], title: str = "Leaderboard"):
    if not rows:
        print(f"\n{'='*60}\n  {title}\n  No results found.\n{'='*60}")
        return
    metric_cols = [k for k in rows[0] if k.endswith("_mean") and k != "rank"]
    header = f"{'Rank':<6}{'Model':<30}{'Cases':<8}"
    header += "".join(f"{c:<20}" for c in metric_cols)
    sep = "=" * len(header)
    print(f"\n{sep}\n  {title}\n{sep}")
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['rank']:<6}{row['model']:<30}{row['num_cases']:<8}"
        for col in metric_cols:
            val = row.get(col, "-")
            line += f"{val:<20}"
        print(line)
    print(sep)


def export_leaderboard_csv(rows: list[dict], output_path: str):
    if not rows:
        return
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def export_leaderboard_markdown(rows: list[dict], output_path: str, title: str = "Leaderboard"):
    if not rows:
        return
    metric_cols = [k for k in rows[0] if k.endswith("_mean")]
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| Rank | Model | Cases |")
        for col in metric_cols:
            f.write(f" {col} |")
        f.write("\n|---|---|---|")
        for _ in metric_cols:
            f.write("---|")
        f.write("\n")
        for row in rows:
            f.write(f"| {row['rank']} | {row['model']} | {row['num_cases']} |")
            for col in metric_cols:
                f.write(f" {row.get(col, '-')} |")
            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Swiss VLM Benchmark Leaderboard")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--export-csv", default=None)
    parser.add_argument("--export-md", default=None)
    parser.add_argument("--by-task", action="store_true")
    parser.add_argument("--by-language", action="store_true")
    args = parser.parse_args()

    rows = build_leaderboard(args.results_dir)
    print_leaderboard(rows, "Overall Leaderboard")

    if args.by_task:
        task_boards = leaderboard_by_task(args.results_dir)
        for task, task_rows in task_boards.items():
            print_leaderboard(task_rows, f"Task: {task}")

    if args.by_language:
        lang_boards = leaderboard_by_language(args.results_dir)
        for lang, lang_rows in lang_boards.items():
            print_leaderboard(lang_rows, f"Language: {lang}")

    if args.export_csv:
        export_leaderboard_csv(rows, args.export_csv)
        print(f"\nExported CSV to {args.export_csv}")
    if args.export_md:
        export_leaderboard_markdown(rows, args.export_md)
        print(f"\nExported Markdown to {args.export_md}")
