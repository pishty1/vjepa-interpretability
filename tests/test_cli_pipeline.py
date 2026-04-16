from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from scripts.vjepa21_pipeline.cli import build_parser
from scripts.vjepa21_pipeline.extract_stage import command_extract


class PipelineCliParserTests(unittest.TestCase):
    def test_run_pipeline_defaults(self):
        parser = build_parser()

        args = parser.parse_args(["run-pipeline", "--frames", "40", "--experiments", "10"])

        self.assertEqual(args.command, "run-pipeline")
        self.assertEqual(args.clip_num_frames, 40)
        self.assertEqual(args.window_shift_frames, 2)
        self.assertEqual(args.num_experiments, 10)
        self.assertEqual(args.device, "mps")

    def test_run_pipeline_parses_simplified_args(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "run-pipeline",
                "--frames",
                "48",
                "--shift",
                "4",
                "--experiments",
                "12",
            ]
        )

        self.assertEqual(args.clip_num_frames, 48)
        self.assertEqual(args.window_shift_frames, 4)
        self.assertEqual(args.num_experiments, 12)

    def test_only_run_pipeline_is_exposed(self):
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["extract"])

    def test_extract_rejects_odd_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "even number"):
                command_extract(
                    SimpleNamespace(
                        video_dir=tmp_dir,
                        output_root=tmp_dir,
                        seed=7,
                        num_experiments=10,
                        clip_num_frames=41,
                        sampling_stride=1,
                        window_shift_frames=2,
                        save_format="jpg",
                    )
                )

    def test_extract_rejects_odd_shift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "even number"):
                command_extract(
                    SimpleNamespace(
                        video_dir=tmp_dir,
                        output_root=tmp_dir,
                        seed=7,
                        num_experiments=10,
                        clip_num_frames=40,
                        sampling_stride=1,
                        window_shift_frames=3,
                        save_format="jpg",
                    )
                )


if __name__ == "__main__":
    unittest.main()
