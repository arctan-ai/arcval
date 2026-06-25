"""Comparison leaderboard for baseline vs Arctan-isolated STT runs."""

import argparse
from pathlib import Path

import pandas as pd

from arcval.stt.leaderboard import _unique_sheet_name
from arcval.utils import read_leaderboard_metrics


def _read_required_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv missing for {results_path.parent}")
    return pd.read_csv(results_path)


def _provider_dirs(condition_dir: Path) -> dict[str, Path]:
    if not condition_dir.exists():
        return {}
    return {
        path.name: path for path in sorted(condition_dir.iterdir()) if path.is_dir()
    }


def _build_detail_sheet(
    provider: str, baseline_df: pd.DataFrame, arctan_df: pd.DataFrame
) -> pd.DataFrame:
    required_cols = {"id", "gt", "pred"}
    if not required_cols.issubset(baseline_df.columns):
        raise ValueError(
            f"Baseline results for {provider} are missing columns: {sorted(required_cols - set(baseline_df.columns))}"
        )
    if not required_cols.issubset(arctan_df.columns):
        raise ValueError(
            f"Arctan results for {provider} are missing columns: {sorted(required_cols - set(arctan_df.columns))}"
        )

    baseline_ids = set(baseline_df["id"].tolist())
    arctan_ids = set(arctan_df["id"].tolist())
    if baseline_ids != arctan_ids:
        raise ValueError(
            f"ID mismatch between baseline and arctan results for {provider}"
        )

    baseline_gt = baseline_df[["id", "gt"]].rename(columns={"gt": "baseline_gt"})
    arctan_gt = arctan_df[["id", "gt"]].rename(columns={"gt": "arctan_gt"})
    gt_compare = baseline_gt.merge(arctan_gt, on="id", how="inner")
    if not gt_compare["baseline_gt"].equals(gt_compare["arctan_gt"]):
        raise ValueError(
            f"GT mismatch between baseline and arctan results for {provider}"
        )

    baseline_renamed = baseline_df.rename(
        columns={col: f"baseline_{col}" for col in baseline_df.columns if col != "id"}
    )
    arctan_renamed = arctan_df.rename(
        columns={col: f"arctan_{col}" for col in arctan_df.columns if col != "id"}
    )

    merged = baseline_renamed.merge(arctan_renamed, on="id", how="inner")
    merged.insert(1, "gt", merged.pop("baseline_gt"))
    del merged["arctan_gt"]

    preferred = ["id", "gt", "baseline_pred", "arctan_pred"]
    other_cols = [col for col in merged.columns if col not in preferred]
    return merged[preferred + other_cols]


def generate_leaderboard(output_dir: str, save_dir: str | None = None) -> str:
    """Generate a comparison workbook for baseline vs Arctan runs."""
    base_path = Path(output_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_path}")

    save_path = (
        base_path / "leaderboard"
        if save_dir is None
        else Path(save_dir).expanduser().resolve()
    )
    save_path.mkdir(parents=True, exist_ok=True)

    baseline_dirs = _provider_dirs(base_path / "baseline")
    arctan_dirs = _provider_dirs(base_path / "arctan")
    if not baseline_dirs and not arctan_dirs:
        print(f"No provider folders found under {base_path}")
        return str(save_path)

    if set(baseline_dirs) != set(arctan_dirs):
        raise ValueError(
            "Provider mismatch between baseline and arctan results: "
            f"baseline={sorted(baseline_dirs)} arctan={sorted(arctan_dirs)}"
        )

    provider_data: dict[str, dict] = {}
    all_metric_keys: set[str] = set()
    for provider in sorted(baseline_dirs):
        baseline_dir = baseline_dirs[provider]
        arctan_dir = arctan_dirs[provider]
        baseline_metrics = read_leaderboard_metrics(baseline_dir / "metrics.json")
        arctan_metrics = read_leaderboard_metrics(arctan_dir / "metrics.json")
        baseline_results = _read_required_results(baseline_dir / "results.csv")
        arctan_results = _read_required_results(arctan_dir / "results.csv")
        provider_data[provider] = {
            "baseline_metrics": baseline_metrics,
            "arctan_metrics": arctan_metrics,
            "baseline_results": baseline_results,
            "arctan_results": arctan_results,
        }
        all_metric_keys.update(baseline_metrics.keys())
        all_metric_keys.update(arctan_metrics.keys())

    metric_keys = [key for key in ("wer", "cer") if key in all_metric_keys]
    metric_keys.extend(
        sorted(key for key in all_metric_keys if key not in {"wer", "cer"})
    )

    summary_rows = []
    detail_frames = {}
    for provider, data in provider_data.items():
        baseline_metrics = data["baseline_metrics"]
        arctan_metrics = data["arctan_metrics"]
        row = {
            "provider": provider,
            "baseline_count": len(data["baseline_results"]),
            "arctan_count": len(data["arctan_results"]),
        }
        for metric in metric_keys:
            baseline_value = baseline_metrics.get(metric)
            arctan_value = arctan_metrics.get(metric)
            row[f"baseline_{metric}"] = baseline_value
            row[f"arctan_{metric}"] = arctan_value
            row[f"{metric}_delta"] = (
                None
                if baseline_value is None or arctan_value is None
                else arctan_value - baseline_value
            )
        summary_rows.append(row)
        detail_frames[provider] = _build_detail_sheet(
            provider,
            data["baseline_results"],
            data["arctan_results"],
        )

    summary_df = pd.DataFrame(summary_rows)
    workbook_path = save_path / "arctan_eval_leaderboard.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = set()
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer, sheet_name="summary", index=False, float_format="%.5f"
        )
        for provider, df in detail_frames.items():
            sheet_name = _unique_sheet_name(provider, sheet_names)
            df.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.5f")

    print(f"Saved leaderboard workbook to {workbook_path}")
    return str(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Arctan-vs-baseline STT comparison leaderboard"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing baseline/ and arctan/ result subdirectories",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save leaderboard files (defaults to output_dir/leaderboard)",
    )
    args = parser.parse_args()
    generate_leaderboard(output_dir=args.output_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
