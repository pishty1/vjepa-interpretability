# Stats Reference

This file collects the full statistics documentation for the V-JEPA 2.1 temporal analysis pipeline.

## CLI usage

Show the stats command help:

```zsh
python scripts/vjepa21_temporal_analysis.py stats --help
```

Useful stats examples:

```zsh
python scripts/vjepa21_temporal_analysis.py stats --stats all
python scripts/vjepa21_temporal_analysis.py stats --stats summary,slice_curve
python scripts/vjepa21_temporal_analysis.py stats --stats dimension_rankings
python scripts/vjepa21_temporal_analysis.py stats --stats pca --pca-components 3
```

## Supported stats and aliases

The `stats` stage accepts a comma-separated `--stats` list.

Primitive outputs:

- `summary`
- `slice_curve`
- `dimension_rankings_all`
- `dimension_rankings_boundary_top`
- `pca_dim_loadings`
- `pca_slice_dim_loadings`
- `spatial_heatmaps`

Aliases:

- `all` = every primitive output above
- `dimension_rankings` = `dimension_rankings_all` + `dimension_rankings_boundary_top`
- `pca` = `pca_dim_loadings` + `pca_slice_dim_loadings`

## Statistical methodology

All statistics are computed after stacking window-level outputs from the chosen model run.

Let:

- `N` = number of processed windows
- `S` = number of overlapping latent slices per window
- `D` = embedding dimension

The stats stage builds these core arrays:

- `spatial_mean_abs_stack` with shape `[N, S, D]`
- `spatial_mean_signed_stack` with shape `[N, S, D]`
- `slice_magnitudes_stack` with shape `[N, S]`
- `heatmap_stack` with shape `[N, S, H, W]`
- `motion_scores` with shape `[N]`

From there, it computes the following analyses.

### 1. Mean absolute difference

Computed as:

- `mean_abs = mean(spatial_mean_abs_stack, axis=0)`

Interpretation:

- for each overlap slice and embedding dimension, this is the average magnitude of latent change across all processed windows
- larger values indicate dimensions that change more strongly between the two clips

Used in:

- `summary.json`
- `dimension_rankings_all.csv`
- `dimension_rankings_boundary_top.csv`

### 2. Standard deviation of absolute difference

Computed as:

- `std_abs = std(spatial_mean_abs_stack, axis=0, ddof=1)` when `N > 1`
- otherwise a zero array is used

Interpretation:

- measures how stable each absolute-difference signal is across windows
- low variance with high mean implies a more repeatable signal

Used in:

- `summary.json` through the slice-level averages
- both dimension ranking CSVs

### 3. Signal-to-noise ratio (SNR)

Computed as:

- `snr = mean_abs / std_abs`, with safe division where zero standard deviation yields zero unless handled by downstream values

Interpretation:

- highlights embedding dimensions with consistently strong absolute differences
- used as the main sorting key in the ranking outputs

Used in:

- `dimension_rankings_all.csv`
- `dimension_rankings_boundary_top.csv`

### 4. Mean signed difference

Computed as:

- `signed_mean = mean(spatial_mean_signed_stack, axis=0)`

Interpretation:

- preserves direction, not just magnitude
- positive values mean `clip_a` tends to exceed `clip_b` in that latent channel after spatial averaging
- negative values mean the reverse

Used in:

- both dimension ranking CSVs

### 5. Paired t-test per slice and dimension

The stats stage runs a paired t-test over windows for every `[slice_idx, dim]` cell using `spatial_mean_signed_stack`.

Computed as:

- `mean_diff = mean(spatial_mean_signed_stack, axis=0)`
- `sample_std = std(spatial_mean_signed_stack, axis=0, ddof=1)`
- `standard_error = sample_std / sqrt(N)`
- `t_stat = mean_diff / standard_error`
- two-sided `p_value` from Student's t distribution with `df = N - 1`

Edge handling in the implementation:

- if `N <= 1`, `t_stat` is all zeros and `p_value` is all ones
- if standard error is zero and mean difference is also zero, `p_value` becomes `1.0`
- if standard error is zero but mean difference is non-zero, `t_stat` becomes signed infinity and `p_value` becomes `0.0`

Interpretation:

- tests whether the signed latent shift differs from zero consistently across windows

Used in:

- `dimension_rankings_all.csv`
- `dimension_rankings_boundary_top.csv`

### 6. Benjamini-Hochberg false discovery rate correction

The code applies Benjamini-Hochberg correction independently within each overlap slice across all embedding dimensions.

Computed as:

- `q_value[slice_idx, :] = benjamini_hochberg(p_value[slice_idx, :])`

It also derives:

- `significant = q_value < 0.001`

Interpretation:

- controls the expected false discovery rate among per-dimension significance calls inside each slice
- the project uses a strict threshold of `0.001`

Used in:

- `dimension_rankings_all.csv`
- `dimension_rankings_boundary_top.csv`

### 7. Slice curve statistics

These collapse the embedding dimension axis to a per-slice curve.

Computed as:

- `mean_slice_curve = mean(mean_abs, axis=1)`
- `std_slice_curve = mean(std_abs, axis=1)`

Interpretation:

- summarizes how much latent change appears at each overlap slice overall
- useful for seeing whether early, middle, or late overlap slices carry stronger temporal differences

Written to:

- `summary.json` as `slice_curve_mean_abs` and `slice_curve_std_abs`
- `slice_curve.csv` with one row per `slice_idx`

### 8. Motion correlation per slice

This relates simple RGB-space motion to latent-space change magnitude.

Computed as Pearson-style correlation between:

- centered `motion_scores`
- centered `slice_magnitudes_stack[:, slice_idx]`

Implementation details:

- `centered_motion = motion_scores - motion_scores.mean()`
- `centered_slice = slice_magnitudes_stack - slice_magnitudes_stack.mean(axis=0, keepdims=True)`
- denominator is the product of Euclidean norms
- if the denominator is zero, correlation defaults to `0.0`

Interpretation:

- positive values mean windows with more raw-frame motion also tend to show stronger latent change in that slice
- negative values mean more motion is associated with lower latent difference for that slice

Written to:

- `summary.json` as `motion_correlation_per_slice`
- `slice_curve.csv` as `motion_correlation`

### 9. Mean spatial heatmaps

These preserve spatial token layout instead of collapsing everything into dimensions.

Computed as:

- `heatmap_mean = mean(heatmap_stack, axis=0)` with shape `[S, H, W]`
- `overall_heatmap = mean(heatmap_mean, axis=0)` with shape `[H, W]`

Interpretation:

- shows where, spatially, the model sees the largest absolute latent change
- `overall_heatmap` averages across slices to expose persistent hotspots

Written to:

- `spatial_heatmaps.npz`
  - `per_slice`: average heatmap per overlap slice
  - `overall`: average heatmap across all slices
- `overall_heatmap.png`: rendered image of the overall heatmap

### 10. PCA on flattened slice-dimension features

The PCA analysis uses the absolute-difference tensor only.

Preparation:

- flatten `[S, D]` into one feature vector per window
- `matrix = spatial_mean_abs_stack.reshape(N, S * D)`

The number of retained components is:

- `min(--pca-components, N, S * D)`

Only the first principal component (`PC1`) is exported.

#### `pca_dim_loadings.csv`

The code reshapes `PC1` back to `[S, D]` and computes per-dimension summaries:

- `pc1_abs_loading_sum`: sum of absolute PC1 loadings across slices for that dimension
- `pc1_abs_loading_mean`: mean absolute PC1 loading across slices
- `pc1_best_slice`: slice with the largest absolute loading for that dimension
- `pc1_loading_at_best_slice`: signed PC1 loading at that slice

Rows are sorted by descending `pc1_abs_loading_sum`.

Interpretation:

- ranks dimensions by how strongly they participate in the dominant mode of cross-window variation

#### `pca_slice_dim_loadings.csv`

This output expands the reshaped `PC1` matrix into one row per `[slice_idx, dim]` pair:

- `slice_idx`
- `dim`
- `pc1_loading`

Interpretation:

- exposes the full first-component structure across time and embedding channels

#### PCA skip behavior

If there are not enough samples or dimensions to estimate at least one component, the stage writes:

- `pca_skipped.txt`

## Stats output files

When `--stats all` is used, the stats run may write the following files.

### `metadata.json`

Run manifest with:

- requested and resolved stats
- source run IDs
- processed sample count
- overlap slice count
- embedding dimension
- generated file list

### `summary.json`

High-level run summary with:

- model config and checkpoint references
- processed sample count
- overlap slice count
- grid size
- embedding dimension
- `slice_curve_mean_abs`
- `slice_curve_std_abs`
- `motion_correlation_per_slice`
- source window count and skipped windows

### `slice_curve.csv`

One row per overlap slice:

- `slice_idx`
- `mean_abs_diff`
- `std_abs_diff`
- `motion_correlation`

### `dimension_rankings_all.csv`

One row per overlap slice and embedding dimension, sorted by `slice_idx` and descending `snr`.

Columns:

- `slice_idx`
- `dim`
- `mean_abs_diff`
- `std_abs_diff`
- `snr`
- `mean_signed_diff`
- `t_stat`
- `p_value`
- `fdr_q_value`
- `significant_fdr_q_lt_0_001`

### `dimension_rankings_boundary_top.csv`

Subset of the ranking table restricted to boundary slices only:

- first overlap slice (`slice_idx = 0`)
- last overlap slice (`slice_idx = overlap_slices - 1`)

The implementation sorts those rows by boundary slice and descending `snr`, then keeps the first `2 * --top-k` rows. With the default `--top-k 25`, this keeps up to `50` rows total.

Columns are the same as `dimension_rankings_all.csv`.

### `pca_dim_loadings.csv`

One row per embedding dimension summarizing the strength of that dimension in `PC1`.

Columns:

- `dim`
- `pc1_abs_loading_sum`
- `pc1_abs_loading_mean`
- `pc1_best_slice`
- `pc1_loading_at_best_slice`

### `pca_slice_dim_loadings.csv`

One row per slice and dimension for the first principal component.

Columns:

- `slice_idx`
- `dim`
- `pc1_loading`

### `spatial_heatmaps.npz`

Compressed NumPy archive containing:

- `per_slice`: average spatial heatmap for each overlap slice
- `overall`: average spatial heatmap across all slices

### `overall_heatmap.png`

Rendered PNG visualization of the `overall` heatmap from `spatial_heatmaps.npz`.

## Interpreting results

- high `mean_abs_diff` means stronger latent change magnitude
- high `snr` means the change is strong and relatively stable across windows
- large absolute `mean_signed_diff` means the latent shift is directionally biased, not just large
- low `fdr_q_value` means a dimension is statistically convincing after multiple-testing correction within a slice
- high positive `motion_correlation` means simple pixel motion tracks latent change for that slice
- strong PCA loadings indicate dimensions and slices that dominate the main pattern of variation across windows

## Notes for macOS

- video decoding is done with OpenCV instead of `decord`
- V-JEPA model code is fetched through `torch.hub` by default
- for offline use, pass `--repo-dir /absolute/path/to/vjepa2`

## Current example outputs in this workspace

The workspace already contains examples under:

- `outputs/extractions/`
- `outputs/model_runs/`
- `outputs/heatmaps/`
- `outputs/stats_runs/`

One existing stats run includes:

- `dimension_rankings_all.csv`
- `dimension_rankings_boundary_top.csv`
- `pca_dim_loadings.csv`
- `pca_slice_dim_loadings.csv`
- `slice_curve.csv`
- `spatial_heatmaps.npz`
- `summary.json`

That layout matches the code paths described above.