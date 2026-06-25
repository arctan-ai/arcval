"""Run STT providers on baseline audio and Arctan-isolated audio."""

import argparse
import asyncio
import json
import os
import sys
from os.path import exists, join

from arcval.arctan_eval.leaderboard import generate_leaderboard
from arcval.arctan_eval.preprocess import build_arctan_input_dir
from arcval.stt.benchmark import MAX_PARALLEL_PROVIDERS
from arcval.stt.eval import (
    STT_LANGUAGES,
    STT_PROVIDERS,
    run_single_provider_eval,
    validate_stt_input_dir,
)
from arcval.utils import StreamTee


def _format_metrics(metrics: dict) -> str:
    judge_scores = {
        key: value["mean"]
        for key, value in metrics.items()
        if isinstance(value, dict) and "type" in value
    }
    parts = [
        f"WER={metrics.get('wer', 0):.4f}",
        f"CER={metrics.get('cer', 0):.4f}",
        f"Sarvam Intent Score={metrics.get('sarvam_intent_score', 0):.4f}",
        f"Sarvam Entity Score={metrics.get('sarvam_entity_score', 0):.4f}",
    ]
    if judge_scores:
        parts.append(", ".join(f"{k}={v:.4f}" for k, v in judge_scores.items()))
    return ", ".join(parts)


async def run(
    providers: list[str],
    input_dir: str,
    output_dir: str = "./out",
    language: str = "english",
    input_file_name: str = "stt.csv",
    debug: bool = False,
    debug_count: int = 5,
    ignore_retry: bool = False,
    overwrite: bool = False,
    max_parallel: int = MAX_PARALLEL_PROVIDERS,
    judge_evaluators: list[dict] | None = None,
    skip_llm_judge: bool = False,
    skip_intent_entity: bool = False,
) -> dict:
    invalid_providers = [
        provider for provider in providers if provider not in STT_PROVIDERS
    ]
    if invalid_providers:
        raise ValueError(
            f"Invalid STT provider(s): {', '.join(invalid_providers)}. "
            f"Available providers: {', '.join(STT_PROVIDERS)}"
        )

    is_valid, error_msg = validate_stt_input_dir(input_dir, input_file_name)
    if not is_valid:
        raise ValueError(f"Input validation error: {error_msg}")

    derived_input_dir = join(output_dir, "_derived", "arctan_input")
    baseline_output_dir = join(output_dir, "baseline")
    arctan_output_dir = join(output_dir, "arctan")

    build_arctan_input_dir(
        input_dir=input_dir,
        output_dir=derived_input_dir,
        input_file_name=input_file_name,
        debug=debug,
        debug_count=debug_count,
        overwrite=overwrite,
    )

    semaphore = asyncio.Semaphore(max_parallel)
    results: dict[str, dict] = {}

    async def run_provider(provider: str) -> tuple[str, dict]:
        async with semaphore:
            baseline_result = await run_single_provider_eval(
                provider=provider,
                language=language,
                input_dir=input_dir,
                input_file_name=input_file_name,
                output_dir=baseline_output_dir,
                debug=debug,
                debug_count=debug_count,
                ignore_retry=ignore_retry,
                overwrite=overwrite,
                judge_evaluators=judge_evaluators,
                skip_llm_judge=skip_llm_judge,
                skip_intent_entity=skip_intent_entity,
            )
            arctan_result = await run_single_provider_eval(
                provider=provider,
                language=language,
                input_dir=derived_input_dir,
                input_file_name=input_file_name,
                output_dir=arctan_output_dir,
                debug=debug,
                debug_count=debug_count,
                ignore_retry=ignore_retry,
                overwrite=overwrite,
                judge_evaluators=judge_evaluators,
                skip_llm_judge=skip_llm_judge,
                skip_intent_entity=skip_intent_entity,
            )
            return provider, {
                "baseline": baseline_result,
                "arctan": arctan_result,
            }

    provider_results = await asyncio.gather(
        *(run_provider(provider) for provider in providers)
    )
    for provider, result in provider_results:
        results[provider] = result

    leaderboard_dir = join(output_dir, "leaderboard")
    leaderboard_error = None
    try:
        generate_leaderboard(output_dir=output_dir, save_dir=leaderboard_dir)
    except Exception as exc:
        leaderboard_error = str(exc)

    status = "completed"
    if leaderboard_error or any(
        result[condition].get("status") == "error"
        for result in results.values()
        for condition in ("baseline", "arctan")
    ):
        status = "error"

    return {
        "status": status,
        "output_dir": output_dir,
        "derived_input_dir": derived_input_dir,
        "leaderboard_dir": leaderboard_dir,
        "leaderboard_error": leaderboard_error,
        "providers": results,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark STT providers on baseline and Arctan-isolated audio"
    )
    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="STT provider(s) to evaluate (space-separated for multiple)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        choices=STT_LANGUAGES,
        help="Language of the audio files",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="Path to the input directory containing the audio files and stt.csv",
    )
    parser.add_argument(
        "-f",
        "--input-file-name",
        type=str,
        default="stt.csv",
        help="Name of the input file containing the dataset to evaluate",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Path to the output directory to save the results",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run the evaluation on the first N audio files",
    )
    parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of audio files to run the evaluation on in debug mode",
    )
    parser.add_argument(
        "--ignore_retry",
        action="store_true",
        help="Ignore retrying if all the audios are not processed and move on to evaluators",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of resuming from last checkpoint",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with an `evaluators` list",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM judge evaluation and only compute WER/CER metrics",
    )
    _ie_group = parser.add_mutually_exclusive_group()
    _ie_group.add_argument(
        "--skip-intent-entity",
        action="store_true",
        dest="skip_intent_entity",
        default=None,
        help="Skip the intent/entity preservation judge",
    )
    _ie_group.add_argument(
        "--no-skip-intent-entity",
        action="store_false",
        dest="skip_intent_entity",
        help="Run the intent/entity preservation judge",
    )

    args = parser.parse_args()
    if args.skip_intent_entity is None:
        setattr(args, "skip_intent_entity", False)

    providers = args.provider
    if not providers:
        print("\033[31mError: --provider is required\033[0m")
        sys.exit(1)
    if not args.input_dir:
        print("\033[31mError: --input-dir is required\033[0m")
        sys.exit(1)

    for provider in providers:
        if provider not in STT_PROVIDERS:
            print(f"\033[31mError: Invalid provider '{provider}'.\033[0m")
            print(f"Available providers: {', '.join(STT_PROVIDERS)}")
            sys.exit(1)

    is_valid, error_msg = validate_stt_input_dir(args.input_dir, args.input_file_name)
    if not is_valid:
        print(f"\033[31mInput validation error: {error_msg}\033[0m")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    judge_evaluators = None
    if args.config:
        with open(args.config) as fp:
            judge_evaluators = json.load(fp).get("evaluators")

    log_path = join(args.output_dir, "logs")
    if exists(log_path):
        os.remove(log_path)
    log_file = open(log_path, "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamTee(original_stdout, log_file)
    sys.stderr = StreamTee(original_stderr, log_file)

    try:
        print("\n\033[91mArctan Eval\033[0m\n")
        print(f"Provider(s): {', '.join(providers)}")
        print(f"Language: {args.language}")
        print(f"Input: {args.input_dir}")
        print(f"Output: {args.output_dir}")
        print("")

        result = await run(
            providers=providers,
            language=args.language,
            input_dir=args.input_dir,
            input_file_name=args.input_file_name,
            output_dir=args.output_dir,
            debug=args.debug,
            debug_count=args.debug_count,
            ignore_retry=args.ignore_retry,
            overwrite=args.overwrite,
            judge_evaluators=judge_evaluators,
            skip_llm_judge=args.skip_llm_judge,
            skip_intent_entity=args.skip_intent_entity,
        )

        print(f"\n\033[92m{'=' * 60}\033[0m")
        print(f"\033[92mSummary\033[0m")
        print(f"\033[92m{'=' * 60}\033[0m\n")

        has_errors = False
        for provider in providers:
            provider_result = result["providers"].get(provider, {})
            baseline_result = provider_result.get("baseline", {})
            arctan_result = provider_result.get("arctan", {})
            if baseline_result.get("status") == "error":
                print(
                    f"  {provider} baseline: \033[31mError - {baseline_result.get('error')}\033[0m"
                )
                has_errors = True
            else:
                print(
                    f"  {provider} baseline: {_format_metrics(baseline_result.get('metrics', {}))}"
                )
            if arctan_result.get("status") == "error":
                print(
                    f"  {provider} arctan: \033[31mError - {arctan_result.get('error')}\033[0m"
                )
                has_errors = True
            else:
                print(
                    f"  {provider} arctan: {_format_metrics(arctan_result.get('metrics', {}))}"
                )

        if result.get("leaderboard_error"):
            print(f"\n\033[31mLeaderboard error: {result['leaderboard_error']}\033[0m")
            has_errors = True
        else:
            print(
                f"\n\033[92mLeaderboard saved to: {result.get('leaderboard_dir')}\033[0m"
            )

        if has_errors:
            sys.exit(1)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    asyncio.run(main())
