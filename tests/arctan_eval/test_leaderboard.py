import json
import tempfile
import unittest
from pathlib import Path

import openpyxl  # noqa: F401
import pandas as pd


def _write_condition(
    base: Path, condition: str, provider: str, metrics: dict, results: list[dict]
):
    provider_dir = base / condition / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame(results).to_csv(provider_dir / "results.csv", index=False)


class TestArctanEvalLeaderboard(unittest.TestCase):
    def test_generates_summary_with_deltas_and_detail_sheet(self):
        from arcval.arctan_eval.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_condition(
                base,
                "baseline",
                "deepgram",
                {
                    "wer": 0.2,
                    "cer": 0.1,
                    "semantic_match": {"type": "binary", "mean": 0.8},
                },
                [{"id": "a", "gt": "hello", "pred": "helo", "semantic_match": True}],
            )
            _write_condition(
                base,
                "arctan",
                "deepgram",
                {
                    "wer": 0.1,
                    "cer": 0.05,
                    "semantic_match": {"type": "binary", "mean": 0.9},
                },
                [{"id": "a", "gt": "hello", "pred": "hello", "semantic_match": True}],
            )

            save_dir = generate_leaderboard(str(base))
            xlsx = Path(save_dir) / "arctan_eval_leaderboard.xlsx"
            self.assertTrue(xlsx.exists())

            summary = pd.read_excel(xlsx, sheet_name="summary")
            self.assertIn("baseline_wer", summary.columns)
            self.assertIn("arctan_wer", summary.columns)
            self.assertIn("wer_delta", summary.columns)
            self.assertIn("baseline_semantic_match", summary.columns)
            self.assertIn("arctan_semantic_match", summary.columns)
            self.assertIn("semantic_match_delta", summary.columns)
            self.assertAlmostEqual(summary.loc[0, "wer_delta"], -0.1, places=5)

            detail = pd.read_excel(xlsx, sheet_name="deepgram")
            self.assertIn("baseline_pred", detail.columns)
            self.assertIn("arctan_pred", detail.columns)
            self.assertEqual(detail.loc[0, "baseline_pred"], "helo")
            self.assertEqual(detail.loc[0, "arctan_pred"], "hello")

    def test_missing_condition_results_raise(self):
        from arcval.arctan_eval.leaderboard import generate_leaderboard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_condition(
                base,
                "baseline",
                "deepgram",
                {"wer": 0.2},
                [{"id": "a", "gt": "hello", "pred": "helo"}],
            )
            with self.assertRaisesRegex(ValueError, "Provider mismatch"):
                generate_leaderboard(str(base))
