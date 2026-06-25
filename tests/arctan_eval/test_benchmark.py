import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd


class TestArctanEvalBenchmark(unittest.IsolatedAsyncioTestCase):
    async def test_run_builds_derived_dataset_and_runs_both_conditions(self):
        from arcval.arctan_eval import benchmark

        calls = []

        async def fake_run_single_provider_eval(**kwargs):
            calls.append(kwargs)
            provider_dir = Path(kwargs["output_dir"]) / kwargs["provider"]
            provider_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"id": "row_a", "gt": "hello", "pred": "hello"}]).to_csv(
                provider_dir / "results.csv", index=False
            )
            (provider_dir / "metrics.json").write_text('{"wer": 0.1, "cer": 0.05}')
            return {
                "provider": kwargs["provider"],
                "status": "completed",
                "metrics": {"wer": 0.1, "cer": 0.05},
                "output_dir": str(provider_dir),
            }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with (
                patch.object(
                    benchmark,
                    "build_arctan_input_dir",
                    return_value=out / "_derived" / "arctan_input",
                ) as build_mock,
                patch.object(
                    benchmark,
                    "validate_stt_input_dir",
                    return_value=(True, ""),
                ),
                patch.object(
                    benchmark,
                    "run_single_provider_eval",
                    AsyncMock(side_effect=fake_run_single_provider_eval),
                ),
                patch.object(
                    benchmark,
                    "generate_leaderboard",
                    return_value=str(out / "leaderboard"),
                ),
            ):
                result = await benchmark.run(
                    providers=["deepgram"],
                    input_dir="/input",
                    output_dir=str(out),
                    debug=True,
                    debug_count=1,
                    overwrite=True,
                )

        build_mock.assert_called_once()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["input_dir"], "/input")
        self.assertEqual(calls[0]["output_dir"], str(out / "baseline"))
        self.assertEqual(calls[1]["input_dir"], str(out / "_derived" / "arctan_input"))
        self.assertEqual(calls[1]["output_dir"], str(out / "arctan"))
        self.assertEqual(result["status"], "completed")

    async def test_run_marks_error_when_leaderboard_fails(self):
        from arcval.arctan_eval import benchmark

        async def fake_run_single_provider_eval(**kwargs):
            return {
                "provider": kwargs["provider"],
                "status": "completed",
                "metrics": {"wer": 0.1, "cer": 0.05},
            }

        with (
            patch.object(
                benchmark,
                "build_arctan_input_dir",
                return_value=Path("/tmp/derived"),
            ),
            patch.object(
                benchmark,
                "validate_stt_input_dir",
                return_value=(True, ""),
            ),
            patch.object(
                benchmark,
                "run_single_provider_eval",
                AsyncMock(side_effect=fake_run_single_provider_eval),
            ),
            patch.object(
                benchmark,
                "generate_leaderboard",
                side_effect=ValueError("broken leaderboard"),
            ),
        ):
            result = await benchmark.run(
                providers=["deepgram"],
                input_dir="/input",
                output_dir="/tmp/out",
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["leaderboard_error"], "broken leaderboard")
