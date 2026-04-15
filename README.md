# V-JEPA 2.1 Temporal Analysis Pipeline

This project runs a four-stage analysis pipeline on videos using V-JEPA 2.1 latent features:

1. extract paired temporal windows from source videos
2. run the V-JEPA encoder on both clips in each window
3. render boundary latent-difference heatmaps
4. compute summary statistics across all processed windows

The implementation lives in `scripts/vjepa21_pipeline/` and the CLI entrypoint is `scripts/vjepa21_temporal_analysis.py`.

## What this project measures

The pipeline is built to compare two nearby clips from the same source video.

- `clip_a`: the first temporal window
- `clip_b`: the second temporal window, shifted forward by a configurable number of raw frames
- `delta`: the aligned latent difference computed as `clip_a_overlap - clip_b_overlap`

For each processed window, the model stage stores several derived tensors that the stats stage aggregates:

- `spatial_mean_abs`: mean absolute latent difference per overlap slice and embedding dimension
- `spatial_mean_signed`: mean signed latent difference per overlap slice and embedding dimension
- `slice_magnitudes`: mean absolute latent difference per overlap slice
- `heatmap`: mean absolute latent difference per overlap slice and spatial token
- `motion_score`: average absolute RGB frame-to-frame change over the saved `motion_context/` frames

## Repository layout

- `scripts/vjepa21_temporal_analysis.py`: CLI entrypoint
- `scripts/vjepa21_pipeline/config.py`: default paths and stat aliases
- `scripts/vjepa21_pipeline/cli.py`: subcommand registration
- `scripts/vjepa21_pipeline/extract_stage.py`: frame extraction stage
- `scripts/vjepa21_pipeline/model_stage.py`: V-JEPA inference and per-window tensor export
- `scripts/vjepa21_pipeline/heatmaps_stage.py`: boundary latent heatmap rendering
- `scripts/vjepa21_pipeline/stats_stage.py`: statistical aggregation and export
- `scripts/vjepa21_pipeline/runtime.py`: shared numeric helpers and image utilities
- `tests/test_heatmaps_runtime.py`: heatmap-focused tests
- `outputs/`: generated extraction, model, heatmap, and stats runs
- `videos/`: source videos

## Defaults

The code ships with local defaults for this workspace:

- `--video-dir`: `/Users/pishty/ws/vjepa2.1/videos`
- `--output-root`: `/Users/pishty/ws/vjepa2.1/outputs`
- `--checkpoint`: `/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`

## Install

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run the pipeline

All commands below assume you are in the workspace root with the virtual environment activated.

### Show help

```zsh
python scripts/vjepa21_temporal_analysis.py --help
python scripts/vjepa21_temporal_analysis.py extract --help
python scripts/vjepa21_temporal_analysis.py run-model --help
python scripts/vjepa21_temporal_analysis.py heatmaps --help
python scripts/vjepa21_temporal_analysis.py stats --help
```

### Quick start

Run the full pipeline against the default paths:

```zsh
python scripts/vjepa21_temporal_analysis.py extract --max-videos 1
python scripts/vjepa21_temporal_analysis.py run-model --device mps
python scripts/vjepa21_temporal_analysis.py heatmaps
python scripts/vjepa21_temporal_analysis.py stats --stats all
```

### Stage-by-stage commands

#### 1. Extract frame windows

Use the latest videos under `videos/` and write a new extraction run:

```zsh
python scripts/vjepa21_temporal_analysis.py extract --max-videos 2
```

Use a different video directory or output root:

```zsh
python scripts/vjepa21_temporal_analysis.py extract \
	--video-dir /absolute/path/to/videos \
	--output-root /absolute/path/to/outputs
```

#### 2. Run the model

Run the model on the latest extraction run:

```zsh
python scripts/vjepa21_temporal_analysis.py run-model --device mps
```

Run the model on a specific extraction run:

```zsh
python scripts/vjepa21_temporal_analysis.py run-model \
	--extract-run-id extract_YYYYMMDD_HHMMSS \
	--device mps
```

Use a local V-JEPA repo clone or a different checkpoint:

```zsh
python scripts/vjepa21_temporal_analysis.py run-model \
	--repo-dir /absolute/path/to/vjepa2 \
	--checkpoint /absolute/path/to/checkpoint.pt \
	--device mps
```

#### 3. Render heatmaps

Render heatmaps from the latest model run:

```zsh
python scripts/vjepa21_temporal_analysis.py heatmaps
```

Render heatmaps from a specific model run:

```zsh
python scripts/vjepa21_temporal_analysis.py heatmaps \
	--model-run-id model_YYYYMMDD_HHMMSS
```

#### 4. Compute stats

Run all available stats on the latest model run:

```zsh
python scripts/vjepa21_temporal_analysis.py stats --stats all
```

Run only selected stats:

```zsh
python scripts/vjepa21_temporal_analysis.py stats --stats summary,slice_curve
python scripts/vjepa21_temporal_analysis.py stats --stats dimension_rankings
python scripts/vjepa21_temporal_analysis.py stats --stats pca --pca-components 3
```

Run stats for a specific model run:

```zsh
python scripts/vjepa21_temporal_analysis.py stats \
	--model-run-id model_YYYYMMDD_HHMMSS \
	--stats all
```

### Run IDs and outputs

Each stage writes a timestamped run directory under `outputs/`:

- `outputs/extractions/<extract_run_id>/`
- `outputs/model_runs/<model_run_id>/`
- `outputs/heatmaps/<heatmap_run_id>/`
- `outputs/stats_runs/<stats_run_id>/`

If you omit upstream IDs:

- `run-model` uses the latest extraction run
- `heatmaps` uses the latest model run
- `stats` uses the latest model run

If you want to chain explicit runs together:

```zsh
python scripts/vjepa21_temporal_analysis.py run-model --extract-run-id extract_YYYYMMDD_HHMMSS
python scripts/vjepa21_temporal_analysis.py heatmaps --model-run-id model_YYYYMMDD_HHMMSS
python scripts/vjepa21_temporal_analysis.py stats --model-run-id model_YYYYMMDD_HHMMSS --stats all
```

## Pipeline overview

### 1. `extract`

This stage samples windows from videos and writes frame folders that later stages consume.

Key behavior:

- picks up to `10` random valid window starts per video
- saves `clip_a/` and `clip_b/` frame sequences
- saves `motion_context/` for simple motion estimation
- writes per-window metadata plus extraction-run metadata

Output root:

- `outputs/extractions/<extract_run_id>/`

Typical window contents:

- `clip_a/`: first clip frames
- `clip_b/`: shifted clip frames
- `motion_context/`: raw context frames spanning the analyzed motion region
- `window_metadata.json`: frame indices and source metadata

### 2. `run-model`

This stage loads the V-JEPA encoder and computes latent differences between aligned clip slices.

Important details:

- clips are center-cropped and normalized with ImageNet mean/std
- latent overlap is aligned by `latent_shift ~= window_shift_frames / tubelet_size`
- only overlapping latent slices are compared
- per-window outputs are stored as `window_output.npz`

For each valid window, the model stage exports:

- `spatial_mean_abs[slice_idx, dim]`: average of `abs(delta)` over spatial tokens
- `spatial_mean_signed[slice_idx, dim]`: average of signed `delta` over spatial tokens
- `slice_magnitudes[slice_idx]`: average of `spatial_mean_abs` across dimensions
- `heatmap[slice_idx, y, x]`: average of `abs(delta)` across embedding dimensions
- `boundary_latent_diffs[0 or -1, token, dim]`: first and last overlap matrices used by the heatmap stage
- `motion_score`: scalar motion estimate from raw frame differences

Output root:

- `outputs/model_runs/<model_run_id>/`

### 3. `heatmaps`

This stage renders JPEG visualizations from the first and last latent-difference matrices in each window.

Important details:

- uses the signed subtraction `clip_a_overlap - clip_b_overlap`
- renders the first overlap slice and last overlap slice as stacked panels
- colors positive and negative values differently so directional change is visible

Output root:

- `outputs/heatmaps/<heatmap_run_id>/`

### 4. `stats`

This stage aggregates all processed window outputs from a model run and writes analysis artifacts.

For detailed stats documentation, see `readmestats.md`.

Output root:

- `outputs/stats_runs/<stats_run_id>/`

## Notes for macOS

- video decoding is done with OpenCV instead of `decord`
- V-JEPA model code is fetched through `torch.hub` by default
- for offline use, pass `--repo-dir /absolute/path/to/vjepa2`
