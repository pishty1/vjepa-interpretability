from __future__ import annotations

import json
import random
from pathlib import Path

from .config import DEFAULT_OUTPUT_ROOT, DEFAULT_VIDEO_DIR
from .io_utils import discover_videos, ensure_dir, make_run_id, slugify_path, utc_now_iso, write_json, run_root
from .runtime import decode_video, import_runtime_dependencies, save_rgb_frames


def add_extract_parser(subparsers) -> None:
    parser = subparsers.add_parser("extract", help="Extract paired 40-frame sliding windows from videos.")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory containing videos.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit the number of videos. `0` means all videos.")
    parser.add_argument("--windows-per-video", type=int, default=10, help="Paired sliding windows sampled per video.")
    parser.add_argument("--clip-num-frames", type=int, default=40, help="Frames in each extracted window.")
    parser.add_argument("--sampling-stride", type=int, default=1, help="Stride in decoded frames within each window.")
    parser.add_argument("--window-shift-frames", type=int, default=2, help="Raw-frame offset between clip A and clip B.")
    parser.add_argument("--save-format", default="jpg", choices=["jpg", "png"], help="Image format for saved frames.")
    parser.set_defaults(func=command_extract)


def command_extract(args) -> int:
    cv2, np, _, _, _, tqdm = import_runtime_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    videos = discover_videos(video_dir)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_dir}")
    if args.window_shift_frames < 0:
        raise ValueError("`window-shift-frames` must be non-negative.")

    run_id = make_run_id("extract")
    run_dir = ensure_dir(run_root(output_root, "extract") / run_id)
    clip_span = (args.clip_num_frames - 1) * args.sampling_stride + 1
    metadata = {
        "run_id": run_id,
        "stage": "extract",
        "created_at": utc_now_iso(),
        "video_dir": str(video_dir),
        "config": {
            "seed": args.seed,
            "windows_per_video": args.windows_per_video,
            "clip_num_frames": args.clip_num_frames,
            "sampling_stride": args.sampling_stride,
            "window_shift_frames": args.window_shift_frames,
            "clip_span": clip_span,
            "save_format": args.save_format,
        },
        "windows": [],
        "skipped": [],
    }

    total_windows = 0
    processed_videos = 0
    for video_index, video_path in enumerate(tqdm(videos, desc="Extract videos")):
        try:
            frames, fps = decode_video(cv2, video_path)
        except Exception as exc:
            metadata["skipped"].append({"video": str(video_path), "reason": f"decode_failed:{exc}"})
            continue

        total_frames = len(frames)
        max_start = total_frames - clip_span - args.window_shift_frames
        if max_start < 0:
            metadata["skipped"].append({"video": str(video_path), "reason": f"too_short:{total_frames}"})
            continue

        processed_videos += 1
        candidate_starts = list(range(max_start + 1))
        if len(candidate_starts) > args.windows_per_video:
            starts = sorted(random.sample(candidate_starts, args.windows_per_video))
        else:
            starts = candidate_starts

        relative_video = video_path.relative_to(video_dir)
        video_slug = slugify_path(str(relative_video.with_suffix("")))
        video_dir_out = ensure_dir(run_dir / video_slug)

        for local_index, start in enumerate(starts):
            window_id = f"{video_slug}_window_{local_index:04d}"
            window_dir = ensure_dir(video_dir_out / f"window_{local_index:04d}")
            clip_a_indices = (start + np.arange(args.clip_num_frames) * args.sampling_stride).tolist()
            clip_b_indices = (start + args.window_shift_frames + np.arange(args.clip_num_frames) * args.sampling_stride).tolist()
            union_start = min(clip_a_indices[0], clip_b_indices[0])
            union_end = max(clip_a_indices[-1], clip_b_indices[-1])
            motion_indices = list(range(union_start, union_end + 1))

            clip_a_frames = [frames[index] for index in clip_a_indices]
            clip_b_frames = [frames[index] for index in clip_b_indices]
            motion_frames = [frames[index] for index in motion_indices]

            clip_a_dir = ensure_dir(window_dir / "clip_a")
            clip_b_dir = ensure_dir(window_dir / "clip_b")
            motion_dir = ensure_dir(window_dir / "motion_context")
            save_rgb_frames(cv2, clip_a_frames, clip_a_dir, args.save_format)
            save_rgb_frames(cv2, clip_b_frames, clip_b_dir, args.save_format)
            save_rgb_frames(cv2, motion_frames, motion_dir, args.save_format)

            window_metadata = {
                "window_id": window_id,
                "video_index": video_index,
                "video": str(video_path),
                "video_slug": video_slug,
                "relative_video": str(relative_video),
                "fps": fps,
                "total_frames": total_frames,
                "start_frame": start,
                "window_shift_frames": args.window_shift_frames,
                "clip_num_frames": args.clip_num_frames,
                "sampling_stride": args.sampling_stride,
                "clip_a_indices": clip_a_indices,
                "clip_b_indices": clip_b_indices,
                "motion_indices": motion_indices,
                "relative_window_dir": str(window_dir.relative_to(run_dir)),
                "clip_a_dir": str(clip_a_dir.relative_to(run_dir)),
                "clip_b_dir": str(clip_b_dir.relative_to(run_dir)),
                "motion_context_dir": str(motion_dir.relative_to(run_dir)),
            }
            write_json(window_dir / "window_metadata.json", window_metadata)
            metadata["windows"].append(window_metadata)
            total_windows += 1

    metadata["processed_videos"] = processed_videos
    metadata["extracted_windows"] = total_windows
    write_json(run_dir / "metadata.json", metadata)

    print(
        json.dumps(
            {
                "stage": "extract",
                "run_id": run_id,
                "processed_videos": processed_videos,
                "extracted_windows": total_windows,
                "output_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return 0
