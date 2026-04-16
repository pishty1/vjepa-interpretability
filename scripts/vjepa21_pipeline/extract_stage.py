from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from .io_utils import discover_videos, ensure_dir, make_run_id, run_root, slugify_path, utc_now_iso, write_json
from .runtime import decode_video, import_runtime_dependencies, save_rgb_frames


def command_extract(args) -> int:
    cv2, np, _, _, _, tqdm = import_runtime_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.clip_num_frames <= 0:
        raise ValueError("`frames` must be positive.")
    if args.clip_num_frames % 2 != 0:
        raise ValueError("`frames` must be an even number.")
    if args.sampling_stride <= 0:
        raise ValueError("`sampling-stride` must be positive.")
    if args.window_shift_frames <= 0:
        raise ValueError("`shift` must be positive.")
    if args.window_shift_frames % 2 != 0:
        raise ValueError("`shift` must be an even number.")
    if args.num_experiments <= 0:
        raise ValueError("`experiments` must be positive.")

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    videos = discover_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_dir}")

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
            "num_experiments": args.num_experiments,
            "clip_num_frames": args.clip_num_frames,
            "sampling_stride": args.sampling_stride,
            "window_shift_frames": args.window_shift_frames,
            "clip_span": clip_span,
            "save_format": args.save_format,
        },
        "windows": [],
        "skipped": [],
    }

    eligible_videos = []
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

        relative_video = video_path.relative_to(video_dir)
        video_slug = slugify_path(str(relative_video.with_suffix("")))
        eligible_videos.append(
            {
                "video_index": video_index,
                "video_path": video_path,
                "relative_video": relative_video,
                "video_slug": video_slug,
                "fps": fps,
                "frames": frames,
                "total_frames": total_frames,
                "candidate_count": max_start + 1,
            }
        )

    if not eligible_videos:
        raise RuntimeError("No videos are long enough for the requested frame count and shift.")

    total_candidates = sum(item["candidate_count"] for item in eligible_videos)
    selected_count = min(args.num_experiments, total_candidates)
    selected_offsets = sorted(random.sample(range(total_candidates), selected_count))
    cumulative_counts = []
    running_total = 0
    for item in eligible_videos:
        running_total += item["candidate_count"]
        cumulative_counts.append(running_total)

    selected_by_video = defaultdict(list)
    for global_offset in selected_offsets:
        video_bucket = bisect_right(cumulative_counts, global_offset)
        previous_total = 0 if video_bucket == 0 else cumulative_counts[video_bucket - 1]
        start_frame = global_offset - previous_total
        selected_by_video[video_bucket].append(start_frame)

    total_windows = 0
    processed_video_indexes = set()
    for video_bucket, starts in selected_by_video.items():
        item = eligible_videos[video_bucket]
        video_index = item["video_index"]
        video_path = item["video_path"]
        relative_video = item["relative_video"]
        video_slug = item["video_slug"]
        fps = item["fps"]
        frames = item["frames"]
        total_frames = item["total_frames"]
        processed_video_indexes.add(video_index)
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

    metadata["processed_videos"] = len(processed_video_indexes)
    metadata["extracted_windows"] = total_windows
    write_json(run_dir / "metadata.json", metadata)

    print(
        json.dumps(
            {
                "stage": "extract",
                "run_id": run_id,
                "processed_videos": len(processed_video_indexes),
                "extracted_windows": total_windows,
                "requested_experiments": args.num_experiments,
                "output_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return 0
