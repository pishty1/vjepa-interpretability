# V-JEPA 2.1 Pipeline

This project now runs through one public command: `run-pipeline`.

Internally, the pipeline:

1. samples random sliding-window pairs from videos in `videos/`
2. runs the V-JEPA encoder on each sampled pair
3. writes latent-difference heatmaps to `outputs/`

The user does not need to run `extract`, `run-model`, `heatmaps`, or `stats` separately.

## What you provide

To run the pipeline, provide these inputs:

- `--frames`: number of frames in each window, must be even
- `--shift`: frame offset between the two windows, must be even
- `--experiments`: total number of random window comparisons to run

If you run with `--experiments 10`, the pipeline samples `10` random valid sliding-window comparisons across the videos under `videos/`, using random videos and random valid starting frames.

## Defaults

- `--shift` defaults to `2`
- `--device` defaults to `mps`
- `--video-dir` defaults to `/Users/pishty/ws/vjepa2.1/videos`
- `--output-root` defaults to `/Users/pishty/ws/vjepa2.1/outputs`
- `--checkpoint` defaults to `/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`

## Install

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

Basic example:

```zsh
python scripts/vjepa21_temporal_analysis.py run-pipeline \
  --frames 40 \
  --shift 2 \
  --experiments 10
```

Another example:

```zsh
python scripts/vjepa21_temporal_analysis.py run-pipeline \
  --frames 48 \
  --shift 4 \
  --experiments 20 \
  --device mps
```

## Help

```zsh
python scripts/vjepa21_temporal_analysis.py --help
python scripts/vjepa21_temporal_analysis.py run-pipeline --help
```

## Outputs

Each run writes timestamped folders under `outputs/`:

- `outputs/extractions/<extract_run_id>/`
- `outputs/model_runs/<model_run_id>/`
- `outputs/heatmaps/<heatmap_run_id>/`

The final `run-pipeline` summary prints the three run IDs so you can trace the generated artifacts.
