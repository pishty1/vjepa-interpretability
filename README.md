# V-JEPA 2.1 Temporal Shift Analyzer

This workspace includes `scripts/vjepa21_temporal_analysis.py`, now split into four independently runnable stages that match `todo.txt`:

1. extract paired sliding-window frames
2. run the V-JEPA model on an extraction run
3. render edge-slice heatmaps for each window
4. run selected statistics on a model run

The local checkpoint default is still `/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`.

Built-in defaults now cover the most common paths:

- `--video-dir` defaults to `/Users/pishty/ws/vjepa2.1/videos`
- `--output-root` defaults to `/Users/pishty/ws/vjepa2.1/outputs`
- `--checkpoint` defaults to `/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`

## Code layout

- `scripts/vjepa21_temporal_analysis.py`: thin CLI entrypoint
- `scripts/vjepa21_pipeline/config.py`: defaults and static configuration
- `scripts/vjepa21_pipeline/io_utils.py`: run discovery and file helpers
- `scripts/vjepa21_pipeline/runtime.py`: video, model, tensor, and plotting helpers
- `scripts/vjepa21_pipeline/extract_stage.py`: extraction stage
- `scripts/vjepa21_pipeline/model_stage.py`: model execution stage
- `scripts/vjepa21_pipeline/heatmaps_stage.py`: edge heatmap stage
- `scripts/vjepa21_pipeline/stats_stage.py`: selectable statistics stage
- `scripts/vjepa21_pipeline/cli.py`: parser assembly

## Pipeline stages

### 1) Extract sliding windows

- Chooses up to `10` random valid window starts per video.
- Saves `40` frames for `clip_a` and another `40` frames for `clip_b` offset by `2` raw frames by default.
- Also saves a `motion_context/` strip and a `metadata.json` manifest for the full extraction run.

Output root:

- `outputs/extractions/<extract_run_id>/`

### 2) Run the model

- Uses a specific extraction run via `--extract-run-id`, or defaults to the latest extraction run.
- Loads the V-JEPA encoder from `torch.hub` or a local repo clone.
- Saves per-window model outputs as compressed `.npz` files plus run metadata.

Output root:

- `outputs/model_runs/<model_run_id>/`

### 3) Render edge heatmaps

- Uses a specific `--model-run-id`, or defaults to the latest model run for the chosen extraction run.
- Writes heatmaps for the shared first latent slice and shared last latent slice for every window.

Output root:

- `outputs/heatmaps/<heatmap_run_id>/`

### 4) Run selected statistics

- Uses a specific `--model-run-id`, or defaults to the latest model run for the chosen extraction run.
- Lets you choose which statistics to compute with `--stats`.
- Saves every stats run with its own `metadata.json`.

Output root:

- `outputs/stats_runs/<stats_run_id>/`

## Notes for macOS

The upstream V-JEPA repo uses `decord`, but this workflow intentionally decodes videos with OpenCV instead, which is much friendlier on macOS.

The script still needs the V-JEPA model-definition code. By default it fetches that with `torch.hub` from `facebookresearch/vjepa2`. If you want an offline/local-code path, clone that repo somewhere and pass `--repo-dir /path/to/vjepa2`.

## Install

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quick start

This default setup uses `40` input frames and a `2`-frame window shift, which typically gives `19` overlapping latent slices with `tubelet_size=2`.

### Extract frames

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python scripts/vjepa21_temporal_analysis.py extract \
  --max-videos 1
```

### Run the model on the latest extraction

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python scripts/vjepa21_temporal_analysis.py run-model \
  --device mps
```

### Render edge heatmaps for the latest model run

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python scripts/vjepa21_temporal_analysis.py heatmaps \
  --extract-run-id extract_YYYYMMDD_HHMMSS
```

### Run selected statistics on the latest model run

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python scripts/vjepa21_temporal_analysis.py stats \
  --stats summary,slice_curve,dimension_rankings,pca,spatial_heatmaps
```

## Selecting a specific run

Every stage writes a timestamped run directory. You can pass those IDs forward explicitly.

```zsh
python scripts/vjepa21_temporal_analysis.py run-model --extract-run-id extract_YYYYMMDD_HHMMSS
python scripts/vjepa21_temporal_analysis.py heatmaps --model-run-id model_YYYYMMDD_HHMMSS
python scripts/vjepa21_temporal_analysis.py stats --extract-run-id extract_YYYYMMDD_HHMMSS
```

If `--extract-run-id` is omitted for `run-model` or `stats`, the latest extraction run is used.
If `--model-run-id` is omitted for `heatmaps` or `stats`, the latest model run for the selected extraction is used.

## Useful options

```zsh
python scripts/vjepa21_temporal_analysis.py --help
python scripts/vjepa21_temporal_analysis.py extract --help
python scripts/vjepa21_temporal_analysis.py run-model --device mps
python scripts/vjepa21_temporal_analysis.py run-model --repo-dir /absolute/path/to/facebookresearch-vjepa2-clone
python scripts/vjepa21_temporal_analysis.py extract --video-dir /some/other/video/folder
python scripts/vjepa21_temporal_analysis.py run-model --checkpoint /some/other/checkpoint.pt
python scripts/vjepa21_temporal_analysis.py stats --stats all
python scripts/vjepa21_temporal_analysis.py stats --stats summary,slice_curve
```

## Statistics outputs

The `stats` stage can generate any subset of:

- `summary.json`
- `slice_curve.csv`
- `dimension_rankings_all.csv`
- `dimension_rankings_boundary_top.csv`
- `pca_dim_loadings.csv`
- `pca_slice_dim_loadings.csv`
- `spatial_heatmaps.npz`
- `overall_heatmap.png`

Aliases accepted by `--stats`:

- `all`
- `dimension_rankings`
- `pca`

## Extraction outputs

Each extraction window saves:

- `clip_a/`: the first `40` frames
- `clip_b/`: the second `40` frames offset by the sliding window shift
- `motion_context/`: contiguous raw frames spanning both windows
- `window_metadata.json`: frame indices and source-video metadata

The extraction run root also saves a `metadata.json` manifest describing the full extraction run.
