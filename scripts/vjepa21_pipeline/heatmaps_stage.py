from __future__ import annotations

import json
from pathlib import Path

from .io_utils import ensure_dir, load_metadata, make_run_id, resolve_model_run, run_root, utc_now_iso, write_json
from .runtime import import_runtime_dependencies, load_window_outputs, write_stacked_latent_comparison_jpg


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
        "skipped": [],
    }

    latent_shift = int(model_metadata.get("config", {}).get("latent_shift", 0) or 0)

    for item in tqdm(outputs, desc="Heatmaps"):
        window = item["window"]
        boundary_latent_diffs = item.get("boundary_latent_diffs")
        if boundary_latent_diffs is None:
            metadata["skipped"].append(
                {
                    "window_id": window["window_id"],
                    "reason": "missing_boundary_latent_diffs",
                }
            )
            continue
        target_dir = ensure_dir(run_dir / window["video_slug"] / Path(window["relative_window_dir"]).name)
        comparison_path = target_dir / "latent_comparison.jpg"
        plot_info = write_stacked_latent_comparison_jpg(
            cv2,
            np,
            boundary_latent_diffs[0],
            boundary_latent_diffs[-1],
            comparison_path,
        )
        overlap_slices = int(window.get("overlap_slices", 0))
        compared_slices = {
            "start": {
                "overlap_slice_index": 0,
                "left_time_index": latent_shift,
                "right_time_index": 0,
            },
            "end": {
                "overlap_slice_index": max(overlap_slices - 1, 0),
                "left_time_index": max(latent_shift + overlap_slices - 1, 0),
                "right_time_index": max(overlap_slices - 1, 0),
            },
        }
        window_metadata = {
            "window_id": window["window_id"],
            "video": window["video"],
            "comparison_jpg": str(comparison_path.relative_to(run_dir)),
            "plot": {
                "token_count": int(plot_info["matrix_shape"][0]),
                "embedding_dim": int(plot_info["matrix_shape"][1]),
                "grid_size": window.get("grid_size"),
                "overlap_latent_steps": overlap_slices,
                "compared_slices": compared_slices,
                "color_limit": float(plot_info["color_limit"]),
            },
        }
        write_json(target_dir / "window_heatmap_metadata.json", window_metadata)
        metadata["windows"].append(window_metadata)

    if not metadata["windows"]:
        raise RuntimeError(
            "No latent comparison heatmaps were rendered. "
            "This model run does not include boundary latent differences; re-run `run-model` with the updated pipeline first."
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
