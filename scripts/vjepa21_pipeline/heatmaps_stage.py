from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_OUTPUT_ROOT
from .io_utils import ensure_dir, load_metadata, make_run_id, resolve_model_run, run_root, utc_now_iso, write_json
from .runtime import import_runtime_dependencies, load_window_outputs, write_heatmap_png


def add_heatmaps_parser(subparsers) -> None:
    parser = subparsers.add_parser("heatmaps", help="Render edge-slice heatmaps for each extracted window.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--extract-run-id", default=None, help="Extraction run ID. Defaults to the latest extraction run.")
    parser.add_argument("--model-run-id", default=None, help="Model run ID. Defaults to the latest model run for the selected extraction.")
    parser.set_defaults(func=command_heatmaps)


def command_heatmaps(args) -> int:
    cv2, np, _, _, _, tqdm = import_runtime_dependencies()

    output_root = Path(args.output_root).expanduser().resolve()
    model_run_dir = resolve_model_run(output_root, args.model_run_id, args.extract_run_id)
    model_metadata = load_metadata(model_run_dir)
    outputs = load_window_outputs(np, model_run_dir, model_metadata)

    run_id = make_run_id("heatmap")
    run_dir = ensure_dir(run_root(output_root, "heatmap") / run_id)
    metadata = {
        "run_id": run_id,
        "stage": "heatmap",
        "created_at": utc_now_iso(),
        "source_model_run_id": model_run_dir.name,
        "source_extract_run_id": model_metadata.get("source_extract_run_id"),
        "windows": [],
    }

    for item in tqdm(outputs, desc="Heatmaps"):
        window = item["window"]
        heatmap = item["heatmap"]
        target_dir = ensure_dir(run_dir / window["video_slug"] / Path(window["relative_window_dir"]).name)
        shared_start_path = target_dir / "shared_start.png"
        shared_end_path = target_dir / "shared_end.png"
        write_heatmap_png(cv2, np, heatmap[0], shared_start_path, "shared_start")
        write_heatmap_png(cv2, np, heatmap[-1], shared_end_path, "shared_end")
        metadata["windows"].append(
            {
                "window_id": window["window_id"],
                "video": window["video"],
                "shared_start_png": str(shared_start_path.relative_to(run_dir)),
                "shared_end_png": str(shared_end_path.relative_to(run_dir)),
            }
        )

    write_json(run_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "stage": "heatmaps",
                "run_id": run_id,
                "source_model_run_id": model_run_dir.name,
                "rendered_windows": len(metadata["windows"]),
                "output_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return 0
