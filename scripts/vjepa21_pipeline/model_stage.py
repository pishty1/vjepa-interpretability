from __future__ import annotations

import json
from pathlib import Path

from .io_utils import ensure_dir, load_metadata, make_run_id, resolve_extract_run, run_root, utc_now_iso, write_json
from .runtime import (
    choose_device,
    compute_motion_score,
    get_video_processor_crop_size,
    import_runtime_dependencies,
    load_model,
    load_video_processor,
    preprocess_clip,
    read_rgb_frames,
    reshape_tokens,
    to_numpy_features,
)


def command_run_model(args) -> int:
    cv2, np, torch, _, _, tqdm = import_runtime_dependencies()

    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    extract_run_dir = resolve_extract_run(output_root, args.extract_run_id)
    extract_metadata = load_metadata(extract_run_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.crop_size % args.patch_size != 0:
        raise ValueError("`crop-size` must be divisible by `patch-size`.")

    video_processor = load_video_processor(args.hf_model)
    processor_crop_size = get_video_processor_crop_size(video_processor)
    if processor_crop_size is not None and int(args.crop_size) != processor_crop_size:
        raise ValueError(
            "`crop-size` must match the Hugging Face AutoVideoProcessor crop size. "
            f"Received crop_size={args.crop_size}, processor crop_size={processor_crop_size}."
        )

    clip_num_frames = int(extract_metadata["config"]["clip_num_frames"])
    latent_shift = 0
    if int(args.tubelet_size) > 0 and int(extract_metadata["config"]["window_shift_frames"]) > 0:
        latent_shift = max(
            1,
            int(round(int(extract_metadata["config"]["window_shift_frames"]) / int(args.tubelet_size))),
        )
    device = choose_device(torch, args.device)
    encoder, checkpoint_payload, load_msg = load_model(
        torch=torch,
        checkpoint_path=checkpoint_path,
        repo_dir=args.repo_dir,
        clip_num_frames=clip_num_frames,
        crop_size=args.crop_size,
    )
    encoder.to(device)
    grid_size = args.crop_size // args.patch_size

    run_id = make_run_id("model")
    run_dir = ensure_dir(run_root(output_root, "model") / run_id)
    metadata = {
        "run_id": run_id,
        "stage": "model",
        "created_at": utc_now_iso(),
        "source_extract_run_id": extract_run_dir.name,
        "checkpoint": str(checkpoint_path),
        "hf_model": args.hf_model,
        "repo_dir": args.repo_dir,
        "device": str(device),
        "config": {
            "crop_size": args.crop_size,
            "processor_crop_size": processor_crop_size,
            "patch_size": args.patch_size,
            "tubelet_size": args.tubelet_size,
            "clip_num_frames": clip_num_frames,
            "window_shift_frames": int(extract_metadata["config"]["window_shift_frames"]),
            "latent_shift": latent_shift,
        },
        "checkpoint_keys": sorted(checkpoint_payload.keys()),
        "load_state_dict": load_msg,
        "windows": [],
        "skipped": [],
    }

    overlap_slices_expected = None
    embedding_dim_expected = None
    with torch.inference_mode():
        for window in tqdm(extract_metadata.get("windows", []), desc="Model windows"):
            window_dir = extract_run_dir / window["relative_window_dir"]
            try:
                clip_a_frames, _ = read_rgb_frames(cv2, window_dir / "clip_a")
                clip_b_frames, _ = read_rgb_frames(cv2, window_dir / "clip_b")
                motion_frames, _ = read_rgb_frames(cv2, window_dir / "motion_context")
            except Exception as exc:
                metadata["skipped"].append({"window_id": window["window_id"], "reason": f"frame_load_failed:{exc}"})
                continue

            batch_a = preprocess_clip(video_processor, torch, np.stack(clip_a_frames, axis=0)).to(device)
            batch_b = preprocess_clip(video_processor, torch, np.stack(clip_b_frames, axis=0)).to(device)
            features_a = reshape_tokens(np, to_numpy_features(np, encoder(batch_a)), grid_size=grid_size)
            features_b = reshape_tokens(np, to_numpy_features(np, encoder(batch_b)), grid_size=grid_size)

            if latent_shift > 0:
                overlap_a = features_a[:, latent_shift:, :, :, :]
                overlap_b = features_b[:, :-latent_shift, :, :, :]
            else:
                overlap_a = features_a
                overlap_b = features_b

            if overlap_a.shape[1] == 0 or overlap_b.shape[1] == 0:
                metadata["skipped"].append({"window_id": window["window_id"], "reason": "zero_overlap"})
                continue

            overlap_len = min(overlap_a.shape[1], overlap_b.shape[1])
            overlap_a = overlap_a[:, :overlap_len]
            overlap_b = overlap_b[:, :overlap_len]

            if overlap_slices_expected is None:
                overlap_slices_expected = overlap_len
                embedding_dim_expected = int(features_a.shape[-1])
            elif overlap_len != overlap_slices_expected:
                metadata["skipped"].append(
                    {
                        "window_id": window["window_id"],
                        "reason": f"overlap_changed:{overlap_len}!=expected:{overlap_slices_expected}",
                    }
                )
                continue

            delta = overlap_a - overlap_b
            abs_delta = np.abs(delta[0])
            signed_delta = delta[0]
            boundary_latent_diffs = np.stack(
                [
                    signed_delta[0].reshape(-1, signed_delta.shape[-1]),
                    signed_delta[-1].reshape(-1, signed_delta.shape[-1]),
                ],
                axis=0,
            )
            spatial_mean_abs = abs_delta.mean(axis=(1, 2))
            spatial_mean_signed = signed_delta.mean(axis=(1, 2))
            slice_magnitudes = spatial_mean_abs.mean(axis=1)
            heatmap = abs_delta.mean(axis=-1)
            motion_score = compute_motion_score(np, motion_frames)

            target_dir = ensure_dir(run_dir / window["video_slug"] / Path(window["relative_window_dir"]).name)
            npz_name = target_dir / "window_output.npz"
            np.savez_compressed(
                npz_name,
                spatial_mean_abs=spatial_mean_abs,
                spatial_mean_signed=spatial_mean_signed,
                slice_magnitudes=slice_magnitudes,
                heatmap=heatmap,
                boundary_latent_diffs=boundary_latent_diffs,
                motion_score=np.array(motion_score, dtype=np.float32),
            )

            window_record = {
                "window_id": window["window_id"],
                "video": window["video"],
                "video_slug": window["video_slug"],
                "relative_window_dir": str(target_dir.relative_to(run_dir)),
                "output_npz": str(npz_name.relative_to(run_dir)),
                "overlap_slices": overlap_len,
                "grid_size": [grid_size, grid_size],
                "embedding_dim": int(features_a.shape[-1]),
                "boundary_latent_matrix_shape": [
                    int(boundary_latent_diffs.shape[1]),
                    int(boundary_latent_diffs.shape[2]),
                ],
                "motion_score": motion_score,
            }
            write_json(target_dir / "window_output_metadata.json", window_record)
            metadata["windows"].append(window_record)

    metadata["processed_samples"] = len(metadata["windows"])
    metadata["overlap_slices"] = overlap_slices_expected
    metadata["embedding_dim"] = embedding_dim_expected
    write_json(run_dir / "metadata.json", metadata)

    print(
        json.dumps(
            {
                "stage": "run-model",
                "run_id": run_id,
                "source_extract_run_id": extract_run_dir.name,
                "processed_samples": len(metadata["windows"]),
                "overlap_slices": overlap_slices_expected,
                "output_dir": str(run_dir),
                "device": str(device),
            },
            indent=2,
        )
    )
    return 0
