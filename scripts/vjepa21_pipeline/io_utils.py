from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import RUN_FOLDERS, VIDEO_EXTENSIONS


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def slugify_path(path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path)
    return slug.strip("._-") or "item"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_root(output_root: Path, stage: str) -> Path:
    return ensure_dir(output_root / RUN_FOLDERS[stage])


def discover_videos(video_dir: Path) -> list[Path]:
    return sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def find_run_by_id(base_dir: Path, run_id: str) -> Path:
    candidate = base_dir / run_id
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Run not found: {candidate}")


def latest_run(base_dir: Path, predicate=None) -> Path:
    candidates = [path for path in sorted(base_dir.iterdir()) if path.is_dir() and (path / "metadata.json").exists()]
    if predicate is not None:
        filtered = []
        for candidate in candidates:
            try:
                metadata = load_metadata(candidate)
            except Exception:
                continue
            if predicate(metadata):
                filtered.append(candidate)
        candidates = filtered
    if not candidates:
        raise FileNotFoundError(f"No runs found in {base_dir}")
    return candidates[-1]


def resolve_extract_run(output_root: Path, extract_run_id: str | None) -> Path:
    base_dir = run_root(output_root, "extract")
    if extract_run_id:
        return find_run_by_id(base_dir, extract_run_id)
    return latest_run(base_dir)


def resolve_model_run(output_root: Path, model_run_id: str | None, extract_run_id: str | None) -> Path:
    base_dir = run_root(output_root, "model")
    if model_run_id:
        return find_run_by_id(base_dir, model_run_id)
    extract_run_dir = resolve_extract_run(output_root, extract_run_id)
    extract_run = extract_run_dir.name
    return latest_run(base_dir, predicate=lambda metadata: metadata.get("source_extract_run_id") == extract_run)
