from __future__ import annotations

DEFAULT_CHECKPOINT = "/Users/pishty/ws/vjepa-gradio-playground/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"
DEFAULT_VIDEO_DIR = "/Users/pishty/ws/vjepa2.1/videos"
DEFAULT_OUTPUT_ROOT = "/Users/pishty/ws/vjepa2.1/outputs"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RUN_FOLDERS = {
    "extract": "extractions",
    "model": "model_runs",
    "heatmap": "heatmaps",
    "stats": "stats_runs",
}
STAT_ALIASES = {
    "all": {
        "summary",
        "slice_curve",
        "dimension_rankings_all",
        "dimension_rankings_boundary_top",
        "pca_dim_loadings",
        "pca_slice_dim_loadings",
        "spatial_heatmaps",
    },
    "dimension_rankings": {"dimension_rankings_all", "dimension_rankings_boundary_top"},
    "pca": {"pca_dim_loadings", "pca_slice_dim_loadings"},
}
SUPPORTED_STATS = sorted(
    {
        "summary",
        "slice_curve",
        "dimension_rankings_all",
        "dimension_rankings_boundary_top",
        "pca_dim_loadings",
        "pca_slice_dim_loadings",
        "spatial_heatmaps",
    }
)
