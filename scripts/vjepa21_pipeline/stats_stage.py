from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_OUTPUT_ROOT, STAT_ALIASES, SUPPORTED_STATS
from .io_utils import ensure_dir, load_metadata, make_run_id, resolve_model_run, run_root, utc_now_iso, write_csv, write_json
from .runtime import benjamini_hochberg, import_runtime_dependencies, load_window_outputs, paired_t_from_stack, write_heatmap_png


def expand_stats(requested_stats: list[str]) -> list[str]:
    expanded = set()
    for item in requested_stats:
        token = item.strip()
        if not token:
            continue
        if token in STAT_ALIASES:
            expanded.update(STAT_ALIASES[token])
            continue
        if token not in SUPPORTED_STATS:
            supported = ", ".join(SUPPORTED_STATS + sorted(STAT_ALIASES.keys()))
            raise ValueError(f"Unsupported statistic '{token}'. Supported values: {supported}")
        expanded.add(token)
    if not expanded:
        expanded.update(STAT_ALIASES["all"])
    return sorted(expanded)


def add_stats_parser(subparsers) -> None:
    parser = subparsers.add_parser("stats", help="Compute selected statistics from a model run.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--extract-run-id", default=None, help="Extraction run ID. Defaults to the latest extraction run.")
    parser.add_argument("--model-run-id", default=None, help="Model run ID. Defaults to the latest model run for the selected extraction.")
    parser.add_argument(
        "--stats",
        default="all",
        help="Comma-separated stats: all, summary, slice_curve, dimension_rankings, dimension_rankings_all, dimension_rankings_boundary_top, pca, pca_dim_loadings, pca_slice_dim_loadings, spatial_heatmaps.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Rows to keep in the boundary ranking summary.")
    parser.add_argument("--pca-components", type=int, default=3, help="Number of PCA components to estimate.")
    parser.set_defaults(func=command_stats)


def command_stats(args) -> int:
    cv2, np, _, _, student_t, _ = import_runtime_dependencies()
    from sklearn.decomposition import PCA

    output_root = Path(args.output_root).expanduser().resolve()
    model_run_dir = resolve_model_run(output_root, args.model_run_id, args.extract_run_id)
    model_metadata = load_metadata(model_run_dir)
    outputs = load_window_outputs(np, model_run_dir, model_metadata)
    requested_stats = [item.strip() for item in args.stats.split(",")]
    selected_stats = expand_stats(requested_stats)

    spatial_mean_abs_stack = np.stack([item["spatial_mean_abs"] for item in outputs], axis=0)
    spatial_mean_signed_stack = np.stack([item["spatial_mean_signed"] for item in outputs], axis=0)
    slice_magnitudes_stack = np.stack([item["slice_magnitudes"] for item in outputs], axis=0)
    heatmap_stack = np.stack([item["heatmap"] for item in outputs], axis=0)
    motion_scores = np.asarray([item["motion_score"] for item in outputs], dtype=np.float64)

    processed_samples, overlap_slices, embedding_dim = spatial_mean_abs_stack.shape
    mean_abs = spatial_mean_abs_stack.mean(axis=0)
    std_abs = spatial_mean_abs_stack.std(axis=0, ddof=1) if processed_samples > 1 else np.zeros_like(mean_abs)
    snr = np.divide(mean_abs, std_abs, out=np.zeros_like(mean_abs), where=std_abs > 0)
    signed_mean = spatial_mean_signed_stack.mean(axis=0)
    t_stat, p_value = paired_t_from_stack(np, student_t, spatial_mean_signed_stack)
    q_value = np.stack([benjamini_hochberg(np, p_value[slice_idx]) for slice_idx in range(overlap_slices)], axis=0)
    significant = q_value < 0.001
    mean_slice_curve = mean_abs.mean(axis=1)
    std_slice_curve = std_abs.mean(axis=1)
    heatmap_mean = heatmap_stack.mean(axis=0)
    overall_heatmap = heatmap_mean.mean(axis=0)
    centered_motion = motion_scores - motion_scores.mean() if processed_samples > 0 else motion_scores
    centered_slice = slice_magnitudes_stack - slice_magnitudes_stack.mean(axis=0, keepdims=True)
    motion_denom = np.sqrt((centered_motion**2).sum() * (centered_slice**2).sum(axis=0))
    motion_corr = np.divide(
        (centered_motion[:, None] * centered_slice).sum(axis=0),
        motion_denom,
        out=np.zeros(overlap_slices),
        where=motion_denom > 0,
    )

    run_id = make_run_id("stats")
    run_dir = ensure_dir(run_root(output_root, "stats") / run_id)
    metadata = {
        "run_id": run_id,
        "stage": "stats",
        "created_at": utc_now_iso(),
        "source_model_run_id": model_run_dir.name,
        "source_extract_run_id": model_metadata.get("source_extract_run_id"),
        "requested_stats": requested_stats,
        "resolved_stats": selected_stats,
        "processed_samples": processed_samples,
        "overlap_slices": overlap_slices,
        "embedding_dim": embedding_dim,
        "generated_files": [],
    }

    if "summary" in selected_stats:
        summary = {
            "config": model_metadata.get("config", {}),
            "checkpoint": model_metadata.get("checkpoint"),
            "repo_dir": model_metadata.get("repo_dir"),
            "device": model_metadata.get("device"),
            "processed_samples": processed_samples,
            "source_model_run_id": model_run_dir.name,
            "source_extract_run_id": model_metadata.get("source_extract_run_id"),
            "overlap_slices": overlap_slices,
            "grid_size": model_metadata.get("windows", [{}])[0].get("grid_size"),
            "embedding_dim": int(embedding_dim),
            "slice_curve_mean_abs": mean_slice_curve.tolist(),
            "slice_curve_std_abs": std_slice_curve.tolist(),
            "motion_correlation_per_slice": motion_corr.tolist(),
            "window_count": len(model_metadata.get("windows", [])),
            "skipped": model_metadata.get("skipped", []),
        }
        write_json(run_dir / "summary.json", summary)
        metadata["generated_files"].append("summary.json")

    if "slice_curve" in selected_stats:
        rows = []
        for slice_idx in range(overlap_slices):
            rows.append(
                {
                    "slice_idx": slice_idx,
                    "mean_abs_diff": float(mean_slice_curve[slice_idx]),
                    "std_abs_diff": float(std_slice_curve[slice_idx]),
                    "motion_correlation": float(motion_corr[slice_idx]),
                }
            )
        write_csv(run_dir / "slice_curve.csv", rows, ["slice_idx", "mean_abs_diff", "std_abs_diff", "motion_correlation"])
        metadata["generated_files"].append("slice_curve.csv")

    ranking_rows = []
    if "dimension_rankings_all" in selected_stats or "dimension_rankings_boundary_top" in selected_stats:
        for slice_idx in range(overlap_slices):
            for dim in range(embedding_dim):
                ranking_rows.append(
                    {
                        "slice_idx": slice_idx,
                        "dim": dim,
                        "mean_abs_diff": float(mean_abs[slice_idx, dim]),
                        "std_abs_diff": float(std_abs[slice_idx, dim]),
                        "snr": float(snr[slice_idx, dim]),
                        "mean_signed_diff": float(signed_mean[slice_idx, dim]),
                        "t_stat": float(t_stat[slice_idx, dim]),
                        "p_value": float(p_value[slice_idx, dim]),
                        "fdr_q_value": float(q_value[slice_idx, dim]),
                        "significant_fdr_q_lt_0_001": bool(significant[slice_idx, dim]),
                    }
                )
        ranking_rows.sort(key=lambda row: (row["slice_idx"], -row["snr"]))

    if "dimension_rankings_all" in selected_stats:
        write_csv(
            run_dir / "dimension_rankings_all.csv",
            ranking_rows,
            [
                "slice_idx",
                "dim",
                "mean_abs_diff",
                "std_abs_diff",
                "snr",
                "mean_signed_diff",
                "t_stat",
                "p_value",
                "fdr_q_value",
                "significant_fdr_q_lt_0_001",
            ],
        )
        metadata["generated_files"].append("dimension_rankings_all.csv")

    if "dimension_rankings_boundary_top" in selected_stats:
        boundary_rows = [row for row in ranking_rows if row["slice_idx"] in {0, overlap_slices - 1}]
        boundary_rows.sort(key=lambda row: (row["slice_idx"], -row["snr"]))
        write_csv(
            run_dir / "dimension_rankings_boundary_top.csv",
            boundary_rows[: 2 * args.top_k],
            [
                "slice_idx",
                "dim",
                "mean_abs_diff",
                "std_abs_diff",
                "snr",
                "mean_signed_diff",
                "t_stat",
                "p_value",
                "fdr_q_value",
                "significant_fdr_q_lt_0_001",
            ],
        )
        metadata["generated_files"].append("dimension_rankings_boundary_top.csv")

    if "pca_dim_loadings" in selected_stats or "pca_slice_dim_loadings" in selected_stats:
        matrix = spatial_mean_abs_stack.reshape(processed_samples, -1)
        max_components = min(args.pca_components, matrix.shape[0], matrix.shape[1])
        if max_components >= 1:
            pca = PCA(n_components=max_components)
            pca.fit(matrix)
            pc1 = pca.components_[0].reshape(overlap_slices, embedding_dim)
            if "pca_dim_loadings" in selected_stats:
                dim_strength = np.abs(pc1).sum(axis=0)
                best_slice = np.argmax(np.abs(pc1), axis=0)
                pca_dim_rows = [
                    {
                        "dim": dim,
                        "pc1_abs_loading_sum": float(dim_strength[dim]),
                        "pc1_abs_loading_mean": float(np.abs(pc1[:, dim]).mean()),
                        "pc1_best_slice": int(best_slice[dim]),
                        "pc1_loading_at_best_slice": float(pc1[best_slice[dim], dim]),
                    }
                    for dim in range(embedding_dim)
                ]
                pca_dim_rows.sort(key=lambda row: -row["pc1_abs_loading_sum"])
                write_csv(
                    run_dir / "pca_dim_loadings.csv",
                    pca_dim_rows,
                    ["dim", "pc1_abs_loading_sum", "pc1_abs_loading_mean", "pc1_best_slice", "pc1_loading_at_best_slice"],
                )
                metadata["generated_files"].append("pca_dim_loadings.csv")
            if "pca_slice_dim_loadings" in selected_stats:
                pca_slice_rows = [
                    {"slice_idx": slice_idx, "dim": dim, "pc1_loading": float(pc1[slice_idx, dim])}
                    for slice_idx in range(overlap_slices)
                    for dim in range(embedding_dim)
                ]
                write_csv(run_dir / "pca_slice_dim_loadings.csv", pca_slice_rows, ["slice_idx", "dim", "pc1_loading"])
                metadata["generated_files"].append("pca_slice_dim_loadings.csv")
        else:
            (run_dir / "pca_skipped.txt").write_text("PCA skipped: not enough samples or dimensions.\n")
            metadata["generated_files"].append("pca_skipped.txt")

    if "spatial_heatmaps" in selected_stats:
        np.savez_compressed(run_dir / "spatial_heatmaps.npz", per_slice=heatmap_mean, overall=overall_heatmap)
        metadata["generated_files"].append("spatial_heatmaps.npz")
        overall_png = run_dir / "overall_heatmap.png"
        write_heatmap_png(cv2, np, overall_heatmap, overall_png, "overall")
        metadata["generated_files"].append("overall_heatmap.png")

    write_json(run_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "stage": "stats",
                "run_id": run_id,
                "source_model_run_id": model_run_dir.name,
                "generated_files": metadata["generated_files"],
                "output_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return 0
