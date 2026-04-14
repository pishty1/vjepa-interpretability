from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
DEFAULT_CHECKPOINT = "/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"
DEFAULT_VIDEO_DIR = "/Users/pishty/ws/vjepa2.1/videos"
DEFAULT_OUTPUT_DIR = "/Users/pishty/ws/vjepa2.1/outputs/vjepa21_temporal_analysis"


@dataclass
class RunningMoments:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        import numpy as np

        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, value) -> None:
        import numpy as np

        array = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = array - self.mean
        self.mean += delta / self.count
        delta2 = array - self.mean
        self.m2 += delta * delta2

    def variance(self, ddof: int = 1):
        import numpy as np

        if self.count <= ddof:
            return np.zeros_like(self.mean)
        return self.m2 / (self.count - ddof)

    def std(self, ddof: int = 1):
        import numpy as np

        return np.sqrt(self.variance(ddof=ddof))


@dataclass
class RunningPearsonVector:
    size: int

    def __post_init__(self) -> None:
        import numpy as np

        self.count = 0
        self.mean_x = 0.0
        self.mean_y = np.zeros(self.size, dtype=np.float64)
        self.c_xy = np.zeros(self.size, dtype=np.float64)
        self.m2_x = 0.0
        self.m2_y = np.zeros(self.size, dtype=np.float64)

    def update(self, x: float, y) -> None:
        import numpy as np

        y = np.asarray(y, dtype=np.float64)
        self.count += 1
        delta_x = x - self.mean_x
        self.mean_x += delta_x / self.count
        delta_y = y - self.mean_y
        self.mean_y += delta_y / self.count
        self.c_xy += delta_x * (y - self.mean_y)
        self.m2_x += delta_x * (x - self.mean_x)
        self.m2_y += delta_y * (y - self.mean_y)

    def correlation(self):
        import numpy as np

        denom = np.sqrt(self.m2_x * self.m2_y)
        return np.divide(self.c_xy, denom, out=np.zeros_like(self.c_xy), where=denom > 0)


class IncrementalPCABuffer:
    def __init__(self, n_components: int, batch_size: int) -> None:
        from sklearn.decomposition import IncrementalPCA

        self.n_components = n_components
        self.batch_size = batch_size
        self.buffer = []
        self.pca = IncrementalPCA(n_components=n_components)
        self.total_rows = 0
        self.was_fitted = False

    def update(self, row) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush(force=False)

    def flush(self, force: bool) -> None:
        import numpy as np

        if not self.buffer:
            return
        batch = np.stack(self.buffer, axis=0)
        if batch.shape[0] < self.n_components and not (force and self.total_rows + batch.shape[0] >= self.n_components):
            return
        self.pca.partial_fit(batch)
        self.total_rows += batch.shape[0]
        self.was_fitted = True
        self.buffer.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run temporal-context shift analysis with V-JEPA 2.1 ViT-B/16 (384)."
    )
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory containing videos.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to the local V-JEPA 2.1 checkpoint.")
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="Optional local clone of `facebookresearch/vjepa2`. If omitted, the script uses `torch.hub` online.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for analysis outputs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit the number of videos. `0` means all videos.")
    parser.add_argument("--clips-per-video", type=int, default=1, help="Random windows sampled per video.")
    parser.add_argument("--clip-num-frames", type=int, default=40, help="Frames fed into the encoder per window.")
    parser.add_argument(
        "--sampling-stride",
        type=int,
        default=1,
        help="Stride in decoded frames when building one model clip.",
    )
    parser.add_argument(
        "--window-shift-frames",
        type=int,
        default=2,
        help="Raw-frame offset between Window A and Window B. Default `2` yields 19 latent overlaps for tubelet size 2.",
    )
    parser.add_argument("--crop-size", type=int, default=384, help="Input crop size.")
    parser.add_argument("--patch-size", type=int, default=16, help="Spatial patch size.")
    parser.add_argument("--tubelet-size", type=int, default=2, help="Temporal tubelet size.")
    parser.add_argument("--pca-components", type=int, default=3, help="Number of PCA components to estimate.")
    parser.add_argument("--pca-batch-size", type=int, default=16, help="Batch size for incremental PCA.")
    parser.add_argument("--top-k", type=int, default=25, help="Rows to keep in ranked CSV summaries.")
    return parser


def import_runtime_dependencies():
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import t as student_t
    from tqdm.auto import tqdm

    return cv2, np, torch, F, student_t, tqdm


def choose_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_videos(video_dir: Path) -> list[Path]:
    return sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def load_model(torch, checkpoint_path: Path, repo_dir: str | None, clip_num_frames: int, crop_size: int):
    hub_source = repo_dir if repo_dir else "facebookresearch/vjepa2"
    hub_kwargs = {
        "pretrained": False,
        "num_frames": clip_num_frames,
    }

    if crop_size != 384:
        raise ValueError(
            "`vjepa2_1_vit_base_384` expects a 384x384 crop. "
            f"Received crop_size={crop_size}."
        )

    if repo_dir:
        model_bundle = torch.hub.load(repo_or_dir=hub_source, model="vjepa2_1_vit_base_384", source="local", **hub_kwargs)
    else:
        model_bundle = torch.hub.load(repo_or_dir=hub_source, model="vjepa2_1_vit_base_384", **hub_kwargs)

    encoder = model_bundle[0] if isinstance(model_bundle, (tuple, list)) else model_bundle
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = None
    for key in ("ema_encoder", "encoder", "target_encoder"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        raise KeyError(f"Could not find an encoder state dict in checkpoint keys: {list(checkpoint.keys())}")
    cleaned_state = {
        key.replace("module.", "").replace("backbone.", ""): value for key, value in state_dict.items()
    }
    load_msg = encoder.load_state_dict(cleaned_state, strict=False)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, checkpoint, str(load_msg)


def decode_video(cv2, video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frames = []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def resize_center_crop(torch, F, video, crop_size: int):
    short_side = int(round((256.0 / 224.0) * crop_size))
    _, _, height, width = video.shape
    scale = short_side / min(height, width)
    resized_height = max(crop_size, int(round(height * scale)))
    resized_width = max(crop_size, int(round(width * scale)))
    video = F.interpolate(video, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    top = max((resized_height - crop_size) // 2, 0)
    left = max((resized_width - crop_size) // 2, 0)
    return video[:, :, top : top + crop_size, left : left + crop_size]


def preprocess_clip(torch, F, frames, crop_size: int):
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    video = resize_center_crop(torch, F, video, crop_size=crop_size)
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=video.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=video.dtype).view(1, 3, 1, 1)
    video = (video - mean) / std
    return video.permute(1, 0, 2, 3).contiguous()


def to_numpy_features(np, tensor):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 5:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unexpected encoder output shape: {array.shape}")
    return array


def reshape_tokens(np, tokens, grid_size: int):
    if tokens.ndim == 5:
        return tokens
    batch, num_tokens, embed_dim = tokens.shape
    tokens_per_slice = grid_size * grid_size
    if num_tokens % tokens_per_slice != 0:
        raise ValueError(
            f"Token count {num_tokens} is not divisible by grid size {grid_size}x{grid_size}."
        )
    temporal_slices = num_tokens // tokens_per_slice
    return tokens.reshape(batch, temporal_slices, grid_size, grid_size, embed_dim)


def compute_motion_score(np, frames) -> float:
    if len(frames) < 2:
        return 0.0
    sample = np.asarray(frames, dtype=np.float32)
    diffs = np.abs(sample[1:] - sample[:-1])
    return float(diffs.mean())


def benjamini_hochberg(np, p_values):
    flat = np.asarray(p_values, dtype=np.float64).reshape(-1)
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.reshape(p_values.shape)


def paired_t_from_diff(np, student_t, diff_stats: RunningMoments):
    mean_diff = diff_stats.mean
    sample_std = diff_stats.std(ddof=1)
    n = diff_stats.count
    if n <= 1:
        shape = mean_diff.shape
        return np.zeros(shape), np.ones(shape)
    standard_error = sample_std / math.sqrt(n)
    t_stat = np.divide(mean_diff, standard_error, out=np.zeros_like(mean_diff), where=standard_error > 0)
    zero_se_nonzero_mean = (standard_error == 0) & (mean_diff != 0)
    t_stat[zero_se_nonzero_mean] = np.sign(mean_diff[zero_se_nonzero_mean]) * np.inf
    p_value = student_t.sf(np.abs(t_stat), df=n - 1) * 2.0
    p_value = np.where((standard_error == 0) & (mean_diff == 0), 1.0, p_value)
    p_value = np.where(zero_se_nonzero_mean, 0.0, p_value)
    return t_stat, p_value


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cv2, np, torch, F, student_t, tqdm = import_runtime_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    video_dir = Path(args.video_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    videos = discover_videos(video_dir)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_dir}")

    device = choose_device(torch, args.device)
    encoder, checkpoint_payload, load_msg = load_model(
        torch=torch,
        checkpoint_path=checkpoint_path,
        repo_dir=args.repo_dir,
        clip_num_frames=args.clip_num_frames,
        crop_size=args.crop_size,
    )
    encoder.to(device)

    grid_size = args.crop_size // args.patch_size
    if args.crop_size % args.patch_size != 0:
        raise ValueError("`crop-size` must be divisible by `patch-size`.")

    clip_span = (args.clip_num_frames - 1) * args.sampling_stride + 1
    if args.window_shift_frames < 0:
        raise ValueError("`window-shift-frames` must be non-negative.")
    latent_shift = max(1, int(round(args.window_shift_frames / args.tubelet_size))) if args.window_shift_frames > 0 else 0

    temporal_abs_stats = None
    temporal_signed_stats = None
    slice_motion_corr = None
    heatmap_sum = None
    pca = None
    overlap_slices = None
    processed_samples = 0
    processed_videos = 0
    skipped = []

    with torch.inference_mode():
        for video_path in tqdm(videos, desc="Videos"):
            try:
                frames, fps = decode_video(cv2, video_path)
            except Exception as exc:
                skipped.append({"video": str(video_path), "reason": f"decode_failed: {exc}"})
                continue

            total_frames = len(frames)
            max_start = total_frames - clip_span - args.window_shift_frames
            if max_start < 0:
                skipped.append({"video": str(video_path), "reason": f"too_short:{total_frames}"})
                continue

            processed_videos += 1
            candidate_starts = list(range(max_start + 1))
            if len(candidate_starts) > args.clips_per_video:
                starts = random.sample(candidate_starts, args.clips_per_video)
            else:
                starts = candidate_starts

            for start in starts:
                window_a = start + np.arange(args.clip_num_frames) * args.sampling_stride
                window_b = start + args.window_shift_frames + np.arange(args.clip_num_frames) * args.sampling_stride

                clip_a = np.stack([frames[index] for index in window_a], axis=0)
                clip_b = np.stack([frames[index] for index in window_b], axis=0)
                union_start = min(window_a[0], window_b[0])
                union_end = max(window_a[-1], window_b[-1])
                motion_frames = np.stack(frames[union_start : union_end + 1], axis=0)
                motion_score = compute_motion_score(np, motion_frames)

                batch_a = preprocess_clip(torch, F, clip_a, crop_size=args.crop_size).unsqueeze(0).to(device)
                batch_b = preprocess_clip(torch, F, clip_b, crop_size=args.crop_size).unsqueeze(0).to(device)

                features_a = to_numpy_features(np, encoder(batch_a))
                features_b = to_numpy_features(np, encoder(batch_b))
                features_a = reshape_tokens(np, features_a, grid_size=grid_size)
                features_b = reshape_tokens(np, features_b, grid_size=grid_size)

                if latent_shift > 0:
                    overlap_a = features_a[:, latent_shift:, :, :, :]
                    overlap_b = features_b[:, :-latent_shift, :, :, :]
                else:
                    overlap_a = features_a
                    overlap_b = features_b

                if overlap_a.shape[1] == 0 or overlap_b.shape[1] == 0:
                    skipped.append({"video": str(video_path), "reason": "zero_overlap"})
                    continue

                overlap_len = min(overlap_a.shape[1], overlap_b.shape[1])
                overlap_a = overlap_a[:, :overlap_len]
                overlap_b = overlap_b[:, :overlap_len]

                if overlap_slices is None:
                    overlap_slices = overlap_len
                    temporal_abs_stats = RunningMoments((overlap_slices, features_a.shape[-1]))
                    temporal_signed_stats = RunningMoments((overlap_slices, features_a.shape[-1]))
                    slice_motion_corr = RunningPearsonVector(overlap_slices)
                    heatmap_sum = np.zeros((overlap_slices, grid_size, grid_size), dtype=np.float64)
                    pca = IncrementalPCABuffer(args.pca_components, args.pca_batch_size)
                elif overlap_len != overlap_slices:
                    skipped.append(
                        {
                            "video": str(video_path),
                            "reason": f"overlap_changed:{overlap_len}!=expected:{overlap_slices}",
                        }
                    )
                    continue

                delta = overlap_a - overlap_b
                abs_delta = np.abs(delta[0])
                signed_delta = delta[0]
                spatial_mean_abs = abs_delta.mean(axis=(1, 2))
                spatial_mean_signed = signed_delta.mean(axis=(1, 2))
                slice_magnitudes = spatial_mean_abs.mean(axis=1)
                heatmap_sum += abs_delta.mean(axis=-1)

                temporal_abs_stats.update(spatial_mean_abs)
                temporal_signed_stats.update(spatial_mean_signed)
                slice_motion_corr.update(motion_score, slice_magnitudes)
                pca.update(spatial_mean_abs.reshape(-1))
                processed_samples += 1

    if processed_samples == 0 or overlap_slices is None:
        raise RuntimeError("No valid samples were processed. Check the video length and CLI settings.")

    pca.flush(force=True)

    mean_abs = temporal_abs_stats.mean
    std_abs = temporal_abs_stats.std(ddof=1)
    snr = np.divide(mean_abs, std_abs, out=np.zeros_like(mean_abs), where=std_abs > 0)
    t_stat, p_value = paired_t_from_diff(np, student_t, temporal_signed_stats)
    q_value = np.stack([benjamini_hochberg(np, p_value[slice_idx]) for slice_idx in range(overlap_slices)], axis=0)
    significant = q_value < 0.001
    mean_slice_curve = mean_abs.mean(axis=1)
    std_slice_curve = std_abs.mean(axis=1)
    motion_corr = slice_motion_corr.correlation()
    heatmap_mean = heatmap_sum / processed_samples
    overall_heatmap = heatmap_mean.mean(axis=0)

    summary = {
        "config": {
            "video_dir": str(video_dir),
            "checkpoint": str(checkpoint_path),
            "repo_dir": args.repo_dir,
            "device": str(device),
            "clip_num_frames": args.clip_num_frames,
            "sampling_stride": args.sampling_stride,
            "window_shift_frames": args.window_shift_frames,
            "tubelet_size": args.tubelet_size,
            "latent_shift": latent_shift,
            "crop_size": args.crop_size,
            "patch_size": args.patch_size,
        },
        "checkpoint_keys": sorted(checkpoint_payload.keys()),
        "load_state_dict": load_msg,
        "processed_videos": processed_videos,
        "processed_samples": processed_samples,
        "overlap_slices": overlap_slices,
        "grid_size": [grid_size, grid_size],
        "embedding_dim": int(mean_abs.shape[1]),
        "slice_curve_mean_abs": mean_slice_curve.tolist(),
        "slice_curve_std_abs": std_slice_curve.tolist(),
        "motion_correlation_per_slice": motion_corr.tolist(),
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    slice_rows = []
    for slice_idx in range(overlap_slices):
        slice_rows.append(
            {
                "slice_idx": slice_idx,
                "mean_abs_diff": float(mean_slice_curve[slice_idx]),
                "std_abs_diff": float(std_slice_curve[slice_idx]),
                "motion_correlation": float(motion_corr[slice_idx]),
            }
        )
    write_csv(output_dir / "slice_curve.csv", slice_rows, ["slice_idx", "mean_abs_diff", "std_abs_diff", "motion_correlation"])

    ranking_rows = []
    for slice_idx in range(overlap_slices):
        for dim in range(mean_abs.shape[1]):
            ranking_rows.append(
                {
                    "slice_idx": slice_idx,
                    "dim": dim,
                    "mean_abs_diff": float(mean_abs[slice_idx, dim]),
                    "std_abs_diff": float(std_abs[slice_idx, dim]),
                    "snr": float(snr[slice_idx, dim]),
                    "mean_signed_diff": float(temporal_signed_stats.mean[slice_idx, dim]),
                    "t_stat": float(t_stat[slice_idx, dim]),
                    "p_value": float(p_value[slice_idx, dim]),
                    "fdr_q_value": float(q_value[slice_idx, dim]),
                    "significant_fdr_q_lt_0_001": bool(significant[slice_idx, dim]),
                }
            )
    ranking_rows.sort(key=lambda row: (row["slice_idx"], -row["snr"]))
    write_csv(
        output_dir / "dimension_rankings_all.csv",
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

    boundary_rows = [
        row for row in ranking_rows if row["slice_idx"] in {0, overlap_slices - 1}
    ]
    boundary_rows.sort(key=lambda row: (row["slice_idx"], -row["snr"]))
    write_csv(
        output_dir / "dimension_rankings_boundary_top.csv",
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

    if pca.was_fitted:
        pc1 = pca.pca.components_[0].reshape(overlap_slices, mean_abs.shape[1])
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
            for dim in range(mean_abs.shape[1])
        ]
        pca_dim_rows.sort(key=lambda row: -row["pc1_abs_loading_sum"])
        write_csv(
            output_dir / "pca_dim_loadings.csv",
            pca_dim_rows,
            ["dim", "pc1_abs_loading_sum", "pc1_abs_loading_mean", "pc1_best_slice", "pc1_loading_at_best_slice"],
        )
        pca_slice_rows = [
            {"slice_idx": slice_idx, "dim": dim, "pc1_loading": float(pc1[slice_idx, dim])}
            for slice_idx in range(overlap_slices)
            for dim in range(mean_abs.shape[1])
        ]
        write_csv(output_dir / "pca_slice_dim_loadings.csv", pca_slice_rows, ["slice_idx", "dim", "pc1_loading"])
    else:
        (output_dir / "pca_dim_loadings.csv").write_text("PCA skipped: not enough samples for IncrementalPCA.\n")

    np.savez_compressed(
        output_dir / "spatial_heatmaps.npz",
        per_slice=heatmap_mean,
        overall=overall_heatmap,
    )

    print(json.dumps({
        "processed_samples": processed_samples,
        "processed_videos": processed_videos,
        "overlap_slices": overlap_slices,
        "output_dir": str(output_dir),
        "device": str(device),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
