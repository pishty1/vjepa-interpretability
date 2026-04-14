# V-JEPA 2.1 Temporal Shift Analyzer

This workspace now includes `scripts/vjepa21_temporal_analysis.py`, a standalone analysis script for the local V-JEPA 2.1 ViT-B/16 checkpoint at `/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`.

## What it does

- Loads the V-JEPA 2.1 ViT-B/16 encoder architecture through `torch.hub` or an optional local repo clone.
- Loads the local checkpoint weights from disk instead of downloading model weights.
- Samples overlapping windows from videos in `videos/`.
- Computes spatially averaged absolute deltas per latent time slice and embedding dimension.
- Aggregates running statistics for:
  - SNR-style temporal specialist ranking
  - paired t-statistics and FDR-adjusted q-values
  - incremental PCA over flattened temporal-difference vectors
  - slice-wise temporal decay curves
  - spatial heatmaps for attention-sink analysis
  - motion-vs-delta correlation by slice

## Files

- `scripts/vjepa21_temporal_analysis.py`: main runner
- `requirements.txt`: minimal dependencies for this script
- `outputs/vjepa21_temporal_analysis/`: default output folder

## Notes for macOS

The upstream V-JEPA repo uses `decord`, but this script intentionally avoids it and decodes videos with OpenCV instead, which is much friendlier on macOS.

The script still needs the V-JEPA model-definition code. By default it fetches that with `torch.hub` from `facebookresearch/vjepa2`. If you want an offline/local-code path, clone that repo somewhere and pass `--repo-dir /path/to/vjepa2`.

## Install

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quick run

This default setup uses `40` input frames and a `2`-frame window shift, which gives `19` overlapping latent slices with the usual `tubelet_size=2` setup.

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python scripts/vjepa21_temporal_analysis.py \
  --video-dir /Users/pishty/ws/vjepa2.1/videos \
  --checkpoint /Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt \
  --output-dir /Users/pishty/ws/vjepa2.1/outputs/vjepa21_temporal_analysis
```

## Match your literal window example

If you want a raw-frame offset closer to `1-40` vs `5-44`, use `--window-shift-frames 4`.
That changes the latent overlap count accordingly.

```zsh
cd /Users/pishty/ws/vjepa2.1
source .venv/bin/activate
python scripts/vjepa21_temporal_analysis.py --window-shift-frames 4
```

## Useful options

```zsh
python scripts/vjepa21_temporal_analysis.py --help
python scripts/vjepa21_temporal_analysis.py --max-videos 2 --clips-per-video 1
python scripts/vjepa21_temporal_analysis.py --device mps
python scripts/vjepa21_temporal_analysis.py --repo-dir /absolute/path/to/facebookresearch-vjepa2-clone
```

## Outputs

The script writes:

- `summary.json`: run config, counts, overlap size, and skipped files
- `slice_curve.csv`: mean temporal decay curve and motion correlation per slice
- `dimension_rankings_all.csv`: SNR, t-stat, p-value, and FDR q-value for every slice/dim pair
- `dimension_rankings_boundary_top.csv`: top-ranked boundary dimensions
- `pca_dim_loadings.csv`: PC1-based temporal-dimension ranking
- `pca_slice_dim_loadings.csv`: raw PC1 loadings by slice and dimension
- `spatial_heatmaps.npz`: per-slice and overall spatial attention-sink heatmaps
