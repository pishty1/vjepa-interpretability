from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from .config import FRAME_EXTENSIONS


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


@lru_cache(maxsize=None)
def load_video_processor(model_name: str):
    from transformers import AutoVideoProcessor

    return AutoVideoProcessor.from_pretrained(model_name)


def _extract_square_size(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        height = value.get("height")
        width = value.get("width")
        if height is not None or width is not None:
            if height != width:
                raise ValueError(f"Expected a square video crop, received crop_size={value!r}.")
            return int(height) if height is not None else None
        shortest_edge = value.get("shortest_edge")
        if shortest_edge is not None:
            return int(shortest_edge)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if value[0] != value[1]:
            raise ValueError(f"Expected a square video crop, received crop_size={value!r}.")
        return int(value[0])
    height = getattr(value, "height", None)
    width = getattr(value, "width", None)
    if height is not None or width is not None:
        if height != width:
            raise ValueError(f"Expected a square video crop, received crop_size={value!r}.")
        return int(height) if height is not None else None
    shortest_edge = getattr(value, "shortest_edge", None)
    if shortest_edge is not None:
        return int(shortest_edge)
    return int(value)


def get_video_processor_crop_size(processor) -> int | None:
    crop_size = getattr(processor, "crop_size", None)
    if crop_size is not None:
        return _extract_square_size(crop_size)
    size = getattr(processor, "size", None)
    if size is not None:
        return _extract_square_size(size)
    return None


def normalize_video_processor_output(torch, tensor):
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 5D video tensor, received shape {tuple(tensor.shape)}.")
    if tensor.shape[1] == 3:
        return tensor.contiguous()
    if tensor.shape[2] == 3:
        return tensor.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"Unable to infer channel axis from output shape {tuple(tensor.shape)}.")


def preprocess_clip(processor, torch, frames):
    video = frames if isinstance(frames, torch.Tensor) else torch.from_numpy(frames)
    if video.ndim != 4:
        raise ValueError(f"Expected a 4D video input, received shape {tuple(video.shape)}.")
    if video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)
    elif video.shape[1] != 3:
        raise ValueError(f"Expected channel-first or channel-last RGB video, received shape {tuple(video.shape)}.")
    video = video.contiguous()
    result = processor(video, return_tensors="pt")
    return normalize_video_processor_output(torch, result["pixel_values_videos"])


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


def paired_t_from_stack(np, student_t, signed_stack):
    mean_diff = signed_stack.mean(axis=0)
    if signed_stack.shape[0] <= 1:
        return np.zeros_like(mean_diff), np.ones_like(mean_diff)
    sample_std = signed_stack.std(axis=0, ddof=1)
    n = signed_stack.shape[0]
    standard_error = sample_std / math.sqrt(n)
    t_stat = np.divide(mean_diff, standard_error, out=np.zeros_like(mean_diff), where=standard_error > 0)
    zero_se_nonzero_mean = (standard_error == 0) & (mean_diff != 0)
    t_stat[zero_se_nonzero_mean] = np.sign(mean_diff[zero_se_nonzero_mean]) * np.inf
    p_value = student_t.sf(np.abs(t_stat), df=n - 1) * 2.0
    p_value = np.where((standard_error == 0) & (mean_diff == 0), 1.0, p_value)
    p_value = np.where(zero_se_nonzero_mean, 0.0, p_value)
    return t_stat, p_value


def save_rgb_frames(cv2, frames, target_dir: Path, image_format: str) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if image_format == "png" else ".jpg"
    saved = []
    for frame_idx, frame in enumerate(frames):
        frame_path = target_dir / f"frame_{frame_idx:04d}{suffix}"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(frame_path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to save frame: {frame_path}")
        saved.append(str(frame_path.name))
    return saved


def read_rgb_frames(cv2, frame_dir: Path):
    files = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in FRAME_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    frames = []
    for frame_path in files:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames, files


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


def write_heatmap_png(cv2, np, heatmap, output_path: Path, label: str) -> None:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    normalized = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    scale = max(1, 384 // max(colored.shape[0], colored.shape[1]))
    colored = cv2.resize(
        colored,
        (colored.shape[1] * scale, colored.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.putText(colored, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ok = cv2.imwrite(str(output_path), colored)
    if not ok:
        raise RuntimeError(f"Failed to write heatmap image: {output_path}")


def _signed_matrix_to_bgr(np, matrix, color_limit: float):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D latent comparison matrix, received shape {matrix.shape}.")
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if color_limit <= 0:
        color_limit = 1.0
    normalized = np.clip(matrix / color_limit, -1.0, 1.0)
    intensity = np.rint(np.abs(normalized) * 255.0).astype(np.uint8)
    colored = np.full(matrix.shape + (3,), 255, dtype=np.uint8)
    positive = normalized > 0
    negative = normalized < 0
    colored[..., 0] = np.where(positive, 255 - intensity, colored[..., 0])
    colored[..., 1] = np.where(positive, 255 - intensity, colored[..., 1])
    colored[..., 1] = np.where(negative, 255 - intensity, colored[..., 1])
    colored[..., 2] = np.where(negative, 255 - intensity, colored[..., 2])
    return colored


def _build_labeled_panel(cv2, np, matrix, label: str, color_limit: float):
    image = _signed_matrix_to_bgr(np, matrix, color_limit)
    label_band = np.full((36, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(label_band, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([label_band, image], axis=0)


def write_stacked_latent_comparison_jpg(cv2, np, first_matrix, last_matrix, output_path: Path):
    first_matrix = np.asarray(first_matrix, dtype=np.float32)
    last_matrix = np.asarray(last_matrix, dtype=np.float32)
    if first_matrix.shape != last_matrix.shape:
        raise ValueError(
            "First and last latent comparison matrices must share the same shape, "
            f"received {first_matrix.shape} and {last_matrix.shape}."
        )
    color_limit = float(max(np.max(np.abs(first_matrix)), np.max(np.abs(last_matrix)), 1e-8))
    first_panel = _build_labeled_panel(
        cv2,
        np,
        first_matrix,
        f"first overlap diff | {first_matrix.shape[0]} tokens x {first_matrix.shape[1]} dims",
        color_limit,
    )
    last_panel = _build_labeled_panel(
        cv2,
        np,
        last_matrix,
        f"last overlap diff | {last_matrix.shape[0]} tokens x {last_matrix.shape[1]} dims",
        color_limit,
    )
    separator = np.full((10, first_panel.shape[1], 3), 40, dtype=np.uint8)
    composite = np.concatenate([first_panel, separator, last_panel], axis=0)
    ok = cv2.imwrite(str(output_path), composite, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to write latent comparison image: {output_path}")
    return {
        "matrix_shape": [int(first_matrix.shape[0]), int(first_matrix.shape[1])],
        "color_limit": color_limit,
    }


def load_window_outputs(np, model_run_dir: Path, metadata: dict):
    outputs = []
    for window in metadata.get("windows", []):
        npz_path = model_run_dir / window["output_npz"]
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            outputs.append(
                {
                    "window": window,
                    "spatial_mean_abs": data["spatial_mean_abs"],
                    "spatial_mean_signed": data["spatial_mean_signed"],
                    "slice_magnitudes": data["slice_magnitudes"],
                    "heatmap": data["heatmap"],
                    "boundary_latent_diffs": data["boundary_latent_diffs"] if "boundary_latent_diffs" in data else None,
                    "motion_score": float(data["motion_score"]),
                }
            )
    if not outputs:
        raise RuntimeError(f"No model outputs found in {model_run_dir}")
    return outputs
