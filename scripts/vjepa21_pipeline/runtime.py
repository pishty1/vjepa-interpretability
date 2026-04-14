from __future__ import annotations

import math
from pathlib import Path

from .config import FRAME_EXTENSIONS, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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


def load_window_outputs(np, model_run_dir: Path, metadata: dict):
    outputs = []
    for window in metadata.get("windows", []):
        npz_path = model_run_dir / window["output_npz"]
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        outputs.append(
            {
                "window": window,
                "spatial_mean_abs": data["spatial_mean_abs"],
                "spatial_mean_signed": data["spatial_mean_signed"],
                "slice_magnitudes": data["slice_magnitudes"],
                "heatmap": data["heatmap"],
                "motion_score": float(data["motion_score"]),
            }
        )
    if not outputs:
        raise RuntimeError(f"No model outputs found in {model_run_dir}")
    return outputs
