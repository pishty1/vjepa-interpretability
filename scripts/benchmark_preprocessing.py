from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the shared runtime wrapper against direct Hugging Face AutoVideoProcessor calls."
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing extracted RGB frames for one clip, such as clip_a/.",
    )
    parser.add_argument(
        "--hf-model",
        default="facebook/vjepa2-vitg-fpc64-384",
        help="Hugging Face model id used to load AutoVideoProcessor.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations to exclude from timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed iterations per preprocessor.",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _time_callable(fn, repeats: int, warmup: int):
    durations_ms: list[float] = []
    last_output = None
    for iteration in range(warmup + repeats):
        start = time.perf_counter()
        last_output = fn()
        end = time.perf_counter()
        if iteration >= warmup:
            durations_ms.append((end - start) * 1000.0)
    return last_output, durations_ms


def main() -> int:
    args = parse_args()

    from transformers import AutoVideoProcessor

    from scripts.vjepa21_pipeline.runtime import (
        import_runtime_dependencies,
        normalize_video_processor_output,
        preprocess_clip,
        read_rgb_frames,
    )

    cv2, np, torch, _, _, _ = import_runtime_dependencies()

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

    frames, frame_files = read_rgb_frames(cv2, frames_dir)
    frames_np = np.stack(frames, axis=0)
    frames_tchw = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

    processor = AutoVideoProcessor.from_pretrained(args.hf_model)

    def run_runtime_wrapper():
        return preprocess_clip(processor, torch, frames_np)

    def run_hf_direct():
        result = processor(frames_tchw, return_tensors="pt")
        return result["pixel_values_videos"]

    runtime_output, runtime_ms = _time_callable(run_runtime_wrapper, repeats=args.repeats, warmup=args.warmup)
    hf_output, hf_ms = _time_callable(run_hf_direct, repeats=args.repeats, warmup=args.warmup)

    hf_output = normalize_video_processor_output(torch, hf_output)

    difference = (runtime_output - hf_output).detach().abs()
    summary = {
        "frames_dir": str(frames_dir),
        "hf_model": args.hf_model,
        "input": {
            "frame_count": len(frame_files),
            "frame_shape_hwc": list(frames_np.shape[1:]),
        },
        "runtime_wrapper": {
            "shape": list(runtime_output.shape),
            "mean_ms": round(_mean(runtime_ms), 3),
            "median_ms": round(_median(runtime_ms), 3),
            "stddev_ms": round(_stddev(runtime_ms), 3),
            "samples_ms": [round(value, 3) for value in runtime_ms],
        },
        "huggingface_direct": {
            "shape": list(hf_output.shape),
            "mean_ms": round(_mean(hf_ms), 3),
            "median_ms": round(_median(hf_ms), 3),
            "stddev_ms": round(_stddev(hf_ms), 3),
            "samples_ms": [round(value, 3) for value in hf_ms],
        },
        "alignment": {
            "runtime_wrapper_shape": list(runtime_output.shape),
            "max_abs_diff": round(float(difference.max().item()), 8),
            "mean_abs_diff": round(float(difference.mean().item()), 8),
            "allclose_atol_1e-4_rtol_1e-4": bool(torch.allclose(runtime_output, hf_output, atol=1e-4, rtol=1e-4)),
        },
        "speedup": {
            "hf_direct_div_runtime_mean": round(_mean(hf_ms) / max(_mean(runtime_ms), 1e-12), 3),
            "runtime_wrapper_faster": _mean(runtime_ms) < _mean(hf_ms),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())