#!/usr/bin/env python
"""Profile SAM 3 image inference latency over 100 runs.

We reuse 3 sample images, cycle through them, exclude model load time and
warm-up passes, and synchronize the device before each timing measurement.
Reports per-stage and end-to-end latency plus peak device memory.

Usage:
    python m1_test/profile_sam3_inference.py [--num-runs 100] [--warmup 5]
"""

import argparse
import statistics
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

from sam3.model.device_utils import get_default_device
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_CASES = [
    ("truck.jpg", "truck"),
    ("groceries.jpg", "grocery bag"),
    ("test_image.jpg", "child"),
]


def device_sync(device: torch.device) -> None:
    """Block until all in-flight kernels on ``device`` are done."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def mem_snapshot(device: torch.device) -> dict:
    """Return current / peak allocated bytes on ``device``.

    Both CUDA and MPS expose ``memory_allocated`` (live tensor footprint) and a
    peak counter that we can reset before timing. CPU has nothing equivalent —
    return zeros and a note.
    """
    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    if device.type == "mps":
        return {
            "allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
            "reserved_mb": torch.mps.driver_allocated_memory() / 1024**2,
            "peak_allocated_mb": _MPS_PEAK[0] / 1024**2,
        }
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}


# MPS doesn't expose a peak counter, so we maintain our own.
_MPS_PEAK = [0]


def update_mps_peak() -> None:
    if torch.backends.mps.is_available():
        cur = torch.mps.current_allocated_memory()
        if cur > _MPS_PEAK[0]:
            _MPS_PEAK[0] = cur


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        _MPS_PEAK[0] = torch.mps.current_allocated_memory()


def summarize(label: str, samples_ms: list[float]) -> str:
    n = len(samples_ms)
    mean = statistics.fmean(samples_ms)
    stdev = statistics.stdev(samples_ms) if n > 1 else 0.0
    s = sorted(samples_ms)
    p50 = s[n // 2]
    p95 = s[min(int(0.95 * n), n - 1)]
    return (
        f"  {label:<20s} mean={mean:7.1f} ms  std={stdev:6.1f}  "
        f"p50={p50:7.1f}  p95={p95:7.1f}  min={s[0]:7.1f}  max={s[-1]:7.1f}  "
        f"fps={1000.0 / mean:5.2f}"
    )


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for kept detections.",
    )
    args = parser.parse_args()

    device = get_default_device()
    print(f"device: {device}")
    print(
        f"torch: {torch.__version__}  cuda: {torch.cuda.is_available()}  "
        f"mps: {torch.backends.mps.is_available()}"
    )

    # ── Model load (NOT counted in profile) ─────────────────────────
    t0 = time.perf_counter()
    print("loading model (excluded from timing)...")
    model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(
        model, confidence_threshold=args.confidence_threshold
    )
    device_sync(device)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s; model on {next(model.parameters()).device}")

    # Pre-load PIL images (so we don't measure JPEG decode either)
    images = []
    for name, prompt in TEST_CASES:
        p = REPO_ROOT / "assets" / "images" / name
        if not p.exists():
            raise SystemExit(f"missing test image: {p}")
        images.append((Image.open(p).convert("RGB"), prompt, name))
    print(f"prepared {len(images)} source image(s); cycling for {args.num_runs} runs")

    # ── Warm-up ─────────────────────────────────────────────────────
    print(f"warming up ({args.warmup} runs)...")
    with torch.inference_mode():
        for i in range(args.warmup):
            img, prompt, _ = images[i % len(images)]
            state = processor.set_image(img)
            _ = processor.set_text_prompt(state=state, prompt=prompt)
    device_sync(device)

    # ── Snapshot memory baseline + reset peak ──────────────────────
    reset_peak(device)
    baseline = mem_snapshot(device)
    print(
        f"memory baseline: allocated={baseline['allocated_mb']:.1f} MB  "
        f"reserved/driver={baseline['reserved_mb']:.1f} MB"
    )

    # ── Timed runs ─────────────────────────────────────────────────
    set_image_ms: list[float] = []
    set_prompt_ms: list[float] = []
    total_ms: list[float] = []
    n_detections: list[int] = []
    per_image_total: dict[str, list[float]] = {n: [] for _, _, n in images}

    print(f"profiling {args.num_runs} runs...")
    with torch.inference_mode():
        for i in range(args.num_runs):
            img, prompt, name = images[i % len(images)]

            device_sync(device)
            t_start = time.perf_counter()

            t_a = time.perf_counter()
            state = processor.set_image(img)
            device_sync(device)
            t_b = time.perf_counter()
            output = processor.set_text_prompt(state=state, prompt=prompt)
            device_sync(device)
            t_c = time.perf_counter()

            set_image_ms.append((t_b - t_a) * 1000)
            set_prompt_ms.append((t_c - t_b) * 1000)
            total_ms.append((t_c - t_start) * 1000)
            per_image_total[name].append((t_c - t_start) * 1000)
            n_detections.append(
                0 if output["scores"] is None else int(output["scores"].numel())
            )
            update_mps_peak()

            if (i + 1) % 10 == 0:
                print(f"  {i + 1:3d}/{args.num_runs}  last total={total_ms[-1]:6.1f} ms")

    # ── Report ─────────────────────────────────────────────────────
    final = mem_snapshot(device)
    print()
    print("=== Latency (ms per image) ===")
    print(summarize("set_image (encoder)", set_image_ms))
    print(summarize("set_text_prompt", set_prompt_ms))
    print(summarize("total / image", total_ms))
    print()
    print("=== Per-image breakdown (total ms) ===")
    for name in per_image_total:
        if per_image_total[name]:
            print(summarize(name, per_image_total[name]))
    print()
    print("=== Memory ===")
    print(
        f"  allocated (live)     baseline={baseline['allocated_mb']:7.1f} MB  "
        f"after={final['allocated_mb']:7.1f} MB"
    )
    print(
        f"  reserved / driver    baseline={baseline['reserved_mb']:7.1f} MB  "
        f"after={final['reserved_mb']:7.1f} MB"
    )
    print(f"  peak allocated       {final['peak_allocated_mb']:7.1f} MB")
    print()
    avg_dets = statistics.fmean(n_detections)
    print(
        f"avg detections/image: {avg_dets:.2f}  "
        f"(model load: {load_s:.1f}s, excluded from timing)"
    )


if __name__ == "__main__":
    main()
