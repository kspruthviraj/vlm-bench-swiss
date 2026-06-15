"""GPU profiling, batch optimization, and throughput analysis for VLM benchmarks."""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ProfileResult:
    model_name: str
    task: str
    num_samples: int
    total_time_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    peak_memory_mb: float | None = None
    gpu_utilization_pct: float | None = None
    tokens_per_second: float | None = None
    batch_size: int = 1


class LatencyProfiler:
    """Profile inference latency for a VLM model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.latencies: list[float] = []

    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def summary(self) -> dict:
        if not self.latencies:
            return {}
        sorted_lat = sorted(self.latencies)
        n = len(sorted_lat)
        return {
            "count": n,
            "mean_ms": sum(sorted_lat) / n,
            "p50_ms": sorted_lat[int(n * 0.5)],
            "p95_ms": sorted_lat[int(n * 0.95)],
            "p99_ms": sorted_lat[int(n * 0.99)],
            "min_ms": sorted_lat[0],
            "max_ms": sorted_lat[-1],
            "std_ms": (sum((x - sum(sorted_lat) / n) ** 2 for x in sorted_lat) / n) ** 0.5,
        }


class MemoryTracker:
    """Track GPU/CPU memory usage during inference."""

    def __init__(self):
        self.snapshots: list[dict] = []
        self._tracking = False

    def snapshot(self, label: str = ""):
        snap = {"label": label, "timestamp": time.time()}
        try:
            import torch
            if torch.cuda.is_available():
                snap["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                snap["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
                snap["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        snap["cpu_rss_mb"] = mem.rss / 1024 / 1024
        snap["cpu_vms_mb"] = mem.vms / 1024 / 1024
        self.snapshots.append(snap)
        return snap

    def peak_gpu_mb(self) -> float | None:
        gpu_snaps = [s.get("gpu_max_allocated_mb") for s in self.snapshots if "gpu_max_allocated_mb" in s]
        return max(gpu_snaps) if gpu_snaps else None

    def peak_cpu_mb(self) -> float:
        return max((s.get("cpu_rss_mb", 0) for s in self.snapshots), default=0)


class ThroughputAnalyzer:
    """Analyze throughput at different batch sizes."""

    @staticmethod
    def estimate_optimal_batch_size(
        model_name: str,
        sample_latencies: list[float],
        target_gpu_memory_mb: float | None = None,
    ) -> dict:
        if not sample_latencies:
            return {"optimal_batch_size": 1}
        avg_latency = sum(sample_latencies) / len(sample_latencies)
        estimates = {}
        for bs in [1, 2, 4, 8, 16, 32]:
            # Rough estimate: latency increases sub-linearly with batch size
            scaling = bs ** 0.7
            est_latency = avg_latency * scaling
            est_throughput = bs / (est_latency / 1000)
            estimates[bs] = {
                "estimated_latency_ms": round(est_latency, 1),
                "estimated_throughput": round(est_throughput, 2),
            }
        best_bs = max(estimates.keys(), key=lambda b: estimates[b]["estimated_throughput"])
        return {
            "optimal_batch_size": best_bs,
            "estimates": estimates,
        }


def profile_model(
    model,
    test_cases: list,
    batch_sizes: list[int] | None = None,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> dict:
    """Run full profiling on a model: latency, memory, throughput."""
    from benchmark import evaluate_model, load_test_cases

    if batch_sizes is None:
        batch_sizes = [1]

    profiler = LatencyProfiler(model.name)
    mem_tracker = MemoryTracker()

    # Warmup
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        for tc in test_cases[:2]:
            try:
                start = time.perf_counter()
                model.predict(tc.image_path, tc.question)
                profiler.record((time.perf_counter() - start) * 1000)
            except Exception:
                pass

    # Reset profiler after warmup
    profiler = LatencyProfiler(model.name)

    # Profile
    print(f"  Profiling ({num_runs} runs on {len(test_cases)} cases)...")
    mem_tracker.snapshot("start")
    all_latencies = []
    for run_idx in range(num_runs):
        for tc in test_cases:
            try:
                start = time.perf_counter()
                prediction = model.predict(tc.image_path, tc.question)
                latency = (time.perf_counter() - start) * 1000
                profiler.record(latency)
                all_latencies.append(latency)
            except Exception:
                pass
        mem_tracker.snapshot(f"run_{run_idx}")

    mem_tracker.snapshot("end")

    summary = profiler.summary()
    batch_analysis = ThroughputAnalyzer.estimate_optimal_batch_size(model.name, all_latencies)

    total_time = sum(all_latencies) / 1000
    throughput = len(all_latencies) / total_time if total_time > 0 else 0

    result = ProfileResult(
        model_name=model.name,
        task="all",
        num_samples=len(all_latencies),
        total_time_s=round(total_time, 2),
        avg_latency_ms=round(summary.get("mean_ms", 0), 1),
        p50_latency_ms=round(summary.get("p50_ms", 0), 1),
        p95_latency_ms=round(summary.get("p95_ms", 0), 1),
        p99_latency_ms=round(summary.get("p99_ms", 0), 1),
        throughput_samples_per_sec=round(throughput, 3),
        peak_memory_mb=mem_tracker.peak_gpu_mb() or mem_tracker.peak_cpu_mb(),
    )

    return {
        "profile": asdict(result),
        "latency_stats": summary,
        "memory": {
            "peak_gpu_mb": mem_tracker.peak_gpu_mb(),
            "peak_cpu_mb": round(mem_tracker.peak_cpu_mb(), 1),
            "snapshots": len(mem_tracker.snapshots),
        },
        "batch_analysis": batch_analysis,
    }


def compare_model_profiles(profiles: dict[str, dict]) -> dict:
    """Compare profiling results across multiple models."""
    comparison = {}
    for model_name, profile_data in profiles.items():
        p = profile_data.get("profile", {})
        comparison[model_name] = {
            "avg_latency_ms": p.get("avg_latency_ms", 0),
            "p95_latency_ms": p.get("p95_latency_ms", 0),
            "throughput": p.get("throughput_samples_per_sec", 0),
            "peak_memory_mb": p.get("peak_memory_mb", 0),
            "num_samples": p.get("num_samples", 0),
        }

    if not comparison:
        return {}

    fastest = min(comparison.keys(), key=lambda m: comparison[m]["avg_latency_ms"])
    highest_throughput = max(comparison.keys(), key=lambda m: comparison[m]["throughput"])

    return {
        "models": comparison,
        "fastest_model": fastest,
        "highest_throughput_model": highest_throughput,
    }


def save_profile(profile_data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile_data, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile VLM models")
    parser.add_argument("--model", default="mock", choices=["mock", "mimo", "openai", "anthropic"])
    parser.add_argument("--data-dir", default="sample_data")
    parser.add_argument("--output", default="results/profile.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    from benchmark import MockVLM, XiaomiMiMoVLM, OpenAIVLM, load_test_cases

    model_map = {
        "mock": lambda: MockVLM(),
        "mimo": lambda: XiaomiMiMoVLM(),
        "openai": lambda: OpenAIVLM(),
    }
    model = model_map.get(args.model, lambda: MockVLM())()
    cases = load_test_cases(args.data_dir)

    print(f"Profiling {model.name} on {len(cases)} test cases...")
    result = profile_model(model, cases, num_warmup=args.warmup, num_runs=args.runs)
    save_profile(result, args.output)
    print(json.dumps(result["profile"], indent=2))
    print(f"\nSaved to {args.output}")
