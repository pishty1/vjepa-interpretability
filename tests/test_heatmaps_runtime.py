from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.vjepa21_pipeline.heatmaps_stage import command_heatmaps
from scripts.vjepa21_pipeline.runtime import import_runtime_dependencies, load_window_outputs, write_stacked_latent_comparison_jpg


class HeatmapRuntimeTests(unittest.TestCase):
    def test_write_stacked_latent_comparison_jpg(self):
        cv2, _, _, _, _, _ = import_runtime_dependencies()
        first = np.linspace(-1.0, 1.0, 576 * 768, dtype=np.float32).reshape(576, 768)
        last = np.flipud(first)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "latent_comparison.jpg"
            info = write_stacked_latent_comparison_jpg(cv2, np, first, last, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(info["matrix_shape"], [576, 768])
            self.assertGreater(info["color_limit"], 0.0)

    def test_load_window_outputs_reads_boundary_latent_diffs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_run_dir = Path(tmp_dir)
            target_dir = model_run_dir / "sample_video" / "window_0000"
            target_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target_dir / "window_output.npz",
                spatial_mean_abs=np.ones((19, 768), dtype=np.float32),
                spatial_mean_signed=np.zeros((19, 768), dtype=np.float32),
                slice_magnitudes=np.ones((19,), dtype=np.float32),
                heatmap=np.ones((19, 24, 24), dtype=np.float32),
                boundary_latent_diffs=np.ones((2, 576, 768), dtype=np.float32),
            )
            metadata = {
                "windows": [
                    {
                        "window_id": "sample_video_window_0000",
                        "output_npz": "sample_video/window_0000/window_output.npz",
                    }
                ]
            }

            outputs = load_window_outputs(np, model_run_dir, metadata)

            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0]["boundary_latent_diffs"].shape, (2, 576, 768))

    def test_command_heatmaps_writes_stacked_jpg(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            model_run_dir = output_root / "model_runs" / "model_validation_heatmaps"
            target_dir = model_run_dir / "sample_video" / "window_0000"
            target_dir.mkdir(parents=True, exist_ok=True)
            (model_run_dir / "metadata.json").write_text(
                """
{
  "run_id": "model_validation_heatmaps",
  "stage": "model",
  "created_at": "2026-04-14T00:00:00+00:00",
  "source_extract_run_id": "extract_validation_heatmaps",
  "config": {
    "latent_shift": 1
  },
  "windows": [
    {
      "window_id": "sample_video_window_0000",
      "video": "/tmp/sample_video.mp4",
      "video_slug": "sample_video",
      "relative_window_dir": "sample_video/window_0000",
      "output_npz": "sample_video/window_0000/window_output.npz",
      "overlap_slices": 19,
      "grid_size": [24, 24],
      "embedding_dim": 768
    }
  ]
}
                """.strip()
            )
            np.savez_compressed(
                target_dir / "window_output.npz",
                spatial_mean_abs=np.ones((19, 768), dtype=np.float32),
                spatial_mean_signed=np.zeros((19, 768), dtype=np.float32),
                slice_magnitudes=np.ones((19,), dtype=np.float32),
                heatmap=np.ones((19, 24, 24), dtype=np.float32),
                boundary_latent_diffs=np.stack(
                    [
                        np.linspace(-1.0, 1.0, 576 * 768, dtype=np.float32).reshape(576, 768),
                        np.linspace(1.0, -1.0, 576 * 768, dtype=np.float32).reshape(576, 768),
                    ],
                    axis=0,
                ),
            )

            exit_code = command_heatmaps(
                SimpleNamespace(
                    output_root=str(output_root),
                    extract_run_id=None,
                    model_run_id="model_validation_heatmaps",
                )
            )

            heatmap_root = output_root / "heatmaps"
            runs = [path for path in heatmap_root.iterdir() if path.is_dir()]
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(runs), 1)
            rendered = runs[0] / "sample_video" / "window_0000" / "latent_comparison.jpg"
            metadata = runs[0] / "sample_video" / "window_0000" / "window_heatmap_metadata.json"
            self.assertTrue(rendered.exists())
            self.assertTrue(metadata.exists())


if __name__ == "__main__":
    unittest.main()
