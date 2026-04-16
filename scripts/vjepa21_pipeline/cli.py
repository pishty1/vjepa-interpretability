from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .config import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_ROOT, DEFAULT_VIDEO_DIR
from .extract_stage import command_extract
from .heatmaps_stage import command_heatmaps
from .io_utils import latest_run, run_root
from .model_stage import command_run_model


def add_run_pipeline_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "run-pipeline",
        help="Run the full temporal analysis pipeline.",
    )
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory containing videos.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for window sampling.")
    parser.add_argument(
        "--frames",
        dest="clip_num_frames",
        type=int,
        required=True,
        help="Number of frames in each window. Must be an even number.",
    )
    parser.add_argument(
        "--shift",
        dest="window_shift_frames",
        type=int,
        default=2,
        help="Frame shift between the two windows. Must be an even number. Defaults to 2.",
    )
    parser.add_argument(
        "--experiments",
        dest="num_experiments",
        type=int,
        required=True,
        help="Total number of random sliding-window comparisons to run across all videos.",
    )
    parser.add_argument("--save-format", default="jpg", choices=["jpg", "png"], help="Image format for saved frames.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to the local V-JEPA 2.1 checkpoint.")
    parser.add_argument(
        "--hf-model",
        default="facebook/vjepa2-vitg-fpc64-384",
        help="Hugging Face model id used to load AutoVideoProcessor.",
    )
    parser.add_argument("--repo-dir", default=None, help="Optional local clone of `facebookresearch/vjepa2`.")
    parser.add_argument("--device", default="mps", choices=["auto", "cpu", "cuda", "mps"], help="Torch device. Defaults to mps.")
    parser.add_argument("--crop-size", type=int, default=384, help="Input crop size.")
    parser.add_argument("--patch-size", type=int, default=16, help="Spatial patch size.")
    parser.add_argument("--tubelet-size", type=int, default=2, help="Temporal tubelet size.")
    parser.set_defaults(func=command_run_pipeline)


def command_run_pipeline(args) -> int:
    extract_args = SimpleNamespace(
        video_dir=args.video_dir,
        output_root=args.output_root,
        seed=args.seed,
        num_experiments=args.num_experiments,
        clip_num_frames=args.clip_num_frames,
        sampling_stride=1,
        window_shift_frames=args.window_shift_frames,
        save_format=args.save_format,
    )
    command_extract(extract_args)

    output_root = Path(args.output_root).expanduser().resolve()
    extract_run_dir = latest_run(run_root(output_root, "extract"))

    model_args = SimpleNamespace(
        output_root=args.output_root,
        extract_run_id=extract_run_dir.name,
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        repo_dir=args.repo_dir,
        device=args.device,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
    )
    command_run_model(model_args)

    model_run_dir = latest_run(
        run_root(output_root, "model"),
        predicate=lambda metadata: metadata.get("source_extract_run_id") == extract_run_dir.name,
    )
    heatmap_args = SimpleNamespace(
        output_root=args.output_root,
        extract_run_id=extract_run_dir.name,
        model_run_id=model_run_dir.name,
    )
    command_heatmaps(heatmap_args)

    heatmap_run_dir = latest_run(
        run_root(output_root, "heatmap"),
        predicate=lambda metadata: metadata.get("source_model_run_id") == model_run_dir.name,
    )
    print(
        json.dumps(
            {
                "stage": "run-pipeline",
                "extract_run_id": extract_run_dir.name,
                "model_run_id": model_run_dir.name,
                "heatmap_run_id": heatmap_run_dir.name,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the V-JEPA 2.1 pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_pipeline_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
