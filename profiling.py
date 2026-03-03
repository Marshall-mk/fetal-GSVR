"""
Lightweight profiling for each pipeline stage.

Tracks wall time, peak GPU memory, and GPU memory delta per stage.
Usage:
    profiler = PipelineProfiler(device)
    with profiler.stage("Load stacks"):
        obs = load_stacks(...)
    profiler.summary()
"""

import time
from contextlib import contextmanager

import torch


class PipelineProfiler:
    """Profiles wall time and GPU memory for each pipeline stage."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        self.records: list[dict] = []

    @contextmanager
    def stage(self, name: str):
        """Context manager to profile a named stage.

        Args:
            name: Human-readable stage name.
        """
        if self.use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)

        t_start = time.perf_counter()

        yield

        if self.use_cuda:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t_start

        record = {"name": name, "time_s": elapsed}

        if self.use_cuda:
            mem_after = torch.cuda.memory_allocated(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            record["mem_before_mb"] = mem_before / 1e6
            record["mem_after_mb"] = mem_after / 1e6
            record["mem_peak_mb"] = peak / 1e6
            record["mem_delta_mb"] = (mem_after - mem_before) / 1e6

        self.records.append(record)

        # Print immediately so the user sees progress
        self._print_record(record)

    def _print_record(self, r: dict):
        msg = f"  [{r['name']}] {r['time_s']:.1f}s"
        if "mem_peak_mb" in r:
            msg += (
                f" | GPU peak: {r['mem_peak_mb']:.0f} MB"
                f", delta: {r['mem_delta_mb']:+.0f} MB"
                f", current: {r['mem_after_mb']:.0f} MB"
            )
        print(msg)

    def summary(self):
        """Print a summary table of all profiled stages."""
        total_time = sum(r["time_s"] for r in self.records)

        print("\n" + "=" * 72)
        print("Pipeline Profile Summary")
        print("=" * 72)

        if self.use_cuda:
            header = f"{'Stage':<30} {'Time':>8} {'%':>5} {'Peak GPU':>10} {'Delta':>10}"
        else:
            header = f"{'Stage':<30} {'Time':>8} {'%':>5}"
        print(header)
        print("-" * len(header))

        for r in self.records:
            pct = (r["time_s"] / total_time * 100) if total_time > 0 else 0
            line = f"{r['name']:<30} {r['time_s']:>7.1f}s {pct:>4.0f}%"
            if "mem_peak_mb" in r:
                line += f" {r['mem_peak_mb']:>8.0f} MB {r['mem_delta_mb']:>+8.0f} MB"
            print(line)

        print("-" * len(header))
        line = f"{'TOTAL':<30} {total_time:>7.1f}s  100%"
        if self.use_cuda:
            overall_peak = max(r.get("mem_peak_mb", 0) for r in self.records)
            line += f" {overall_peak:>8.0f} MB{'':>10}"
            print(line)
            print(f"\nOverall peak GPU memory: {overall_peak:.0f} MB")
        else:
            print(line)
        print("=" * 72)
