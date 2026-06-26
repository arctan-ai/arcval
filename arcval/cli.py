"""
CLI entry point for arcval package.

Usage:
    # Interactive mode (recommended):
    arcval                                        # Main menu
    arcval stt                                    # Interactive STT evaluation
    arcval tts                                    # Interactive TTS evaluation
    arcval llm                                    # Interactive LLM tests
    arcval simulations                            # Interactive simulations
    arcval status                                  # Check provider connectivity

    # Direct mode:
    arcval llm -c config.json -m openai/gpt-4.1 -p openrouter -o ./out
    arcval simulations --type text -c config.json -m openai/gpt-4.1 -p openrouter -o ./out
    arcval simulations --type voice -c config.json -o ./out
"""

import argparse
import asyncio
import json
import os
import runpy
import sys
from importlib.metadata import version as get_version

from dotenv import find_dotenv, load_dotenv


def _args_to_argv(args, exclude_keys=None, flag_mapping=None):
    """Convert argparse namespace to sys.argv format.

    Args:
        args: argparse.Namespace object
        exclude_keys: set of keys to exclude from conversion
        flag_mapping: dict mapping attribute names to their original flag names
                     (e.g., {'debug_count': '--debug_count', 'input_dir': '--input-dir'})
    """
    exclude_keys = exclude_keys or set()
    flag_mapping = flag_mapping or {}
    argv = []

    for key, value in vars(args).items():
        if key in exclude_keys or value is None:
            continue

        # Use mapping if available, otherwise convert underscores to hyphens
        if key in flag_mapping:
            flag = flag_mapping[key]
        else:
            # Default: convert underscores to hyphens (for flags like --input-dir)
            flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:  # Only add flag if True
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])

    return argv


def _load_cli_dotenv() -> None:
    """Load .env from the directory where the arcval command is run."""
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path, override=True)


def _launch_ink_ui(mode: str):
    """Launch the bundled Ink UI for interactive TTS/STT evaluation."""
    import shutil
    from pathlib import Path

    node_bin = shutil.which("node")
    if not node_bin:
        print(
            f"Error: Node.js is required for the interactive {mode.upper()} UI.\n"
            "Install it from https://nodejs.org/ or via your package manager."
        )
        sys.exit(1)

    bundle_path = Path(__file__).parent / "ui" / "cli.bundle.mjs"
    if not bundle_path.exists():
        print(
            f"Error: UI bundle not found at {bundle_path}\n"
            "Run 'cd ui && npm run bundle' to build it."
        )
        sys.exit(1)

    import subprocess

    result = subprocess.run([node_bin, str(bundle_path), mode])
    sys.exit(result.returncode)


def _print_sample_output(result: dict) -> None:
    """Print sample_output from a verify result, if present."""
    sample = result.get("sample_output")
    if sample is None:
        return
    print("\n  Sample output received from agent:")
    if isinstance(sample, dict):
        if sample.get("response") is not None:
            print(f"  response: {sample['response']}")
        if sample.get("tool_calls"):
            print(f"  tool_calls: {json.dumps(sample['tool_calls'], indent=2)}")
    else:
        print(f"  {sample}")


def _run_agent_verify(
    agent_url: str,
    agent_headers_raw: str | None,
    models: list[str] | None = None,
) -> None:
    """Verify an external agent connection and print the result."""
    from arcval.connections import TextAgentConnection

    headers = None
    if agent_headers_raw:
        try:
            headers = json.loads(agent_headers_raw)
        except json.JSONDecodeError:
            print("✗ --agent-headers is not valid JSON")
            sys.exit(1)

    agent = TextAgentConnection(url=agent_url, headers=headers)

    # If models provided, send the first model name in the verify request
    model_hint: str | None = models[0] if models else None

    body_preview = (
        '{"messages": [...], "model": "' + model_hint + '"}'
        if model_hint
        else '{"messages": [{"role": "user", "content": "Hi"}]}'
    )

    print(f"\nVerifying agent connection: {agent_url}")
    print(f"Sending: {body_preview}")
    print("─" * 60)

    result = asyncio.run(agent.verify(model=model_hint))

    if result["ok"]:
        print("✓ Connection verified — response format is correct")
        _print_sample_output(result)
    else:
        print(f"✗ Verification failed: {result['error']}")
        _print_sample_output(result)
        sys.exit(1)


def _is_interactive_run(args) -> bool:
    """True if this is an interactive UI session (no Slack notifications)."""
    if args.component is None:
        return True
    if args.component in ("stt",) and not args.eval_only and args.provider is None:
        return True
    if args.component in ("tts",) and args.provider is None:
        return True
    if args.component in ("llm",) and args.config is None and not args.verify:
        return True
    if (
        args.component in ("simulations",)
        and (args.type is None or args.config is None)
        and not args.verify
    ):
        return True
    return False


def _skip_slack(args) -> bool:
    """True for quick commands that don't need Slack notifications."""
    if args.component in ("status", "agent"):
        return True
    if getattr(args, "verify", False):
        return True
    if getattr(args, "sim_subcmd", None) == "leaderboard":
        return True
    return False


def _build_cmd_desc(args) -> str:
    """Build a human-readable command description from parsed args."""
    parts = [f"arcval {args.component or 'menu'}"]
    if getattr(args, "provider", None):
        parts.append(f"-p {' '.join(args.provider)}")
    if getattr(args, "model", None):
        models = args.model
        if isinstance(models, list):
            parts.append(f"-m {' '.join(models)}")
        else:
            parts.append(f"-m {models}")
    if getattr(args, "config", None):
        parts.append(f"-c {args.config}")
    if getattr(args, "type", None):
        parts.append(f"--type {args.type}")
    if getattr(args, "eval_only", False):
        parts.append("--eval-only")
    if getattr(args, "dataset", None):
        parts.append(f"--dataset {args.dataset}")
    return " ".join(parts)


def _send_slack(text: str) -> None:
    """Send a Slack message via webhook. No-op if SLACK_WEBHOOK_URL is not set."""
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        from arcval.slack import send_message

        send_message(text, url)
    except Exception:
        pass


def _find_leaderboard_xlsx(output_dir: str) -> str | None:
    """Return the first leaderboard workbook path, if present."""
    from pathlib import Path

    candidate_dirs = [
        Path(output_dir) / "leaderboard",
        Path.cwd() / "leaderboard",
    ]
    for leaderboard_dir in candidate_dirs:
        if not leaderboard_dir.is_dir():
            continue
        files = sorted(
            path for path in leaderboard_dir.iterdir() if path.suffix.lower() == ".xlsx"
        )
        if files:
            return str(files[0])
    return None


def _upload_slack_leaderboard(output_dir: str, text: str) -> tuple[bool, str | None]:
    """Upload the leaderboard workbook to Slack when upload creds are present."""
    xlsx_path = _find_leaderboard_xlsx(output_dir)
    if not xlsx_path:
        return False, "no leaderboard xlsx found"
    if not os.environ.get("SLACK_BOT_TOKEN") or not os.environ.get("SLACK_CHANNEL_ID"):
        return False, "missing SLACK_BOT_TOKEN or SLACK_CHANNEL_ID"
    try:
        from arcval.slack import upload_file

        upload_file(xlsx_path, initial_comment=text)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _format_leaderboard_table(output_dir: str) -> str | None:
    """Read the leaderboard file and return a Slack mrkdwn table.

    Scans ``{output_dir}/leaderboard/`` for ``.xlsx`` (STT/TTS) or ``.csv``
    (LLM tests, simulations) and reads the summary sheet / first sheet.
    Returns a triple-backtick code block with the data, or ``None`` if no
    leaderboard file exists.
    """
    from pathlib import Path

    import pandas as pd

    leaderboard_dir = Path(output_dir) / "leaderboard"
    if not leaderboard_dir.is_dir():
        return None

    files = sorted(leaderboard_dir.iterdir())
    xlsx = [f for f in files if f.suffix == ".xlsx"]
    csv = [f for f in files if f.suffix == ".csv"]

    if not xlsx and not csv:
        return None

    try:
        if xlsx:
            df = pd.read_excel(xlsx[0], sheet_name=0, engine="openpyxl")
        else:
            df = pd.read_csv(csv[0])

        # Truncate to first 12 rows so the message stays within Slack's 4k limit
        if len(df) > 12:
            df = df.head(12)

        table_str = df.to_string(index=False)
        return f"```\n{table_str}\n```"
    except Exception:
        return None


def main():
    """Main CLI entry point that dispatches to component-specific scripts."""
    # Load environment variables from .env file
    _load_cli_dotenv()

    parser = argparse.ArgumentParser(
        prog="arcval",
        usage="arcval [-h] [-v] {stt,arctan-eval,tts,llm,simulations,general,status} ...",
        description="Voice agent evaluation and benchmarking toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    arcval                                        # Main menu (interactive)
    arcval stt                                    # Interactive STT evaluation
    arcval arctan-eval                            # Compare baseline vs Arctan-isolated STT
    arcval tts                                    # Interactive TTS evaluation
    arcval llm                                    # Interactive LLM tests
    arcval llm -c config.json                     # Run LLM tests directly
    arcval simulations                            # Interactive simulations
    arcval simulations --type text -c config.json # Run text simulation directly
    arcval general --dataset data.json -c config.json  # Score input/output pairs
    arcval status                                 # Check provider connectivity
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {get_version('arcval')}",
    )

    subparsers = parser.add_subparsers(
        dest="component",
        help="Component to run",
        metavar="{stt,arctan-eval,tts,llm,simulations,general,status}",
    )
    subparsers.required = False  # Allow `arcval` alone for main menu

    # ── STT ───────────────────────────────────────────────────────
    # `arcval stt` with no args → interactive UI
    # `arcval stt -p provider1 provider2 ... -i input-dir ...` → run benchmark (multi) or eval (single)
    stt_parser = subparsers.add_parser(
        "stt",
        help="Speech-to-text evaluation",
    )
    stt_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="STT provider(s) to evaluate (space-separated for multiple)",
    )
    stt_parser.add_argument("-l", "--language", type=str, default="english")
    stt_parser.add_argument("-i", "--input-dir", type=str)
    stt_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    stt_parser.add_argument("-f", "--input-file-name", type=str, default="stt.csv")
    stt_parser.add_argument("-d", "--debug", action="store_true")
    stt_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    stt_parser.add_argument("--ignore_retry", action="store_true")
    stt_parser.add_argument("--overwrite", action="store_true")
    stt_parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Generate leaderboard after evaluation (for single provider)",
    )
    stt_parser.add_argument("-s", "--save-dir", type=str)
    stt_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with judge settings (model, prompt)",
    )
    stt_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip STT inference and run evaluators directly on a dataset of (gt, pred) pairs",
    )
    stt_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON (list of {id, gt, pred}). Required with --eval-only.",
    )
    _stt_llm_group = stt_parser.add_mutually_exclusive_group()
    _stt_llm_group.add_argument(
        "--skip-llm-judge",
        action="store_true",
        dest="skip_llm_judge",
        default=None,
        help="Skip LLM judge evaluation and only compute WER/CER metrics",
    )
    _stt_llm_group.add_argument(
        "--no-skip-llm-judge",
        action="store_false",
        dest="skip_llm_judge",
        help="Run the LLM judge evaluation",
    )
    _stt_ie_group = stt_parser.add_mutually_exclusive_group()
    _stt_ie_group.add_argument(
        "--skip-intent-entity",
        action="store_true",
        dest="skip_intent_entity",
        default=None,
        help="Skip the intent/entity preservation judge (default: yes)",
    )
    _stt_ie_group.add_argument(
        "--no-skip-intent-entity",
        action="store_false",
        dest="skip_intent_entity",
        help="Run the intent/entity preservation judge even when non-TTY",
    )

    arctan_eval_parser = subparsers.add_parser(
        "arctan-eval",
        help="Compare STT providers with and without Arctan voice isolation",
    )
    arctan_eval_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="STT provider(s) to evaluate (space-separated for multiple)",
    )
    arctan_eval_parser.add_argument("-l", "--language", type=str, default="english")
    arctan_eval_parser.add_argument("-i", "--input-dir", type=str)
    arctan_eval_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    arctan_eval_parser.add_argument(
        "-f", "--input-file-name", type=str, default="stt.csv"
    )
    arctan_eval_parser.add_argument("-d", "--debug", action="store_true")
    arctan_eval_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    arctan_eval_parser.add_argument("--ignore_retry", action="store_true")
    arctan_eval_parser.add_argument("--overwrite", action="store_true")
    arctan_eval_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with judge settings (model, prompt)",
    )
    arctan_eval_parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM judge evaluation and only compute WER/CER metrics",
    )
    _arctan_ie_group = arctan_eval_parser.add_mutually_exclusive_group()
    _arctan_ie_group.add_argument(
        "--skip-intent-entity",
        action="store_true",
        dest="skip_intent_entity",
        default=None,
        help="Skip the intent/entity preservation judge (default: skip on non-TTY, prompt on TTY)",
    )
    _arctan_ie_group.add_argument(
        "--no-skip-intent-entity",
        action="store_false",
        dest="skip_intent_entity",
        help="Run the intent/entity preservation judge even when non-TTY",
    )

    # ── TTS ───────────────────────────────────────────────────────
    # `arcval tts` with no args → interactive UI
    # `arcval tts -p provider -i input ...` → single provider (eval.py)
    # `arcval tts -p provider1 provider2 -i input ...` → multi-provider (benchmark.py)
    tts_parser = subparsers.add_parser(
        "tts",
        help="Text-to-speech evaluation",
    )
    tts_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="TTS provider(s) to use for evaluation (space-separated for multiple)",
    )
    tts_parser.add_argument("-l", "--language", type=str, default="english")
    tts_parser.add_argument("-i", "--input", type=str)
    tts_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    tts_parser.add_argument("-d", "--debug", action="store_true")
    tts_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    tts_parser.add_argument("--overwrite", action="store_true")
    tts_parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Generate leaderboard after evaluation (for single provider)",
    )
    tts_parser.add_argument("-s", "--save-dir", type=str)
    tts_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with judge settings (model, prompt)",
    )

    # ── LLM tests ───────────────────────────────────────────────
    # `arcval llm` with no args → interactive UI
    # `arcval llm -c config.json -m model ...` → single model (run_tests.py)
    # `arcval llm -c config.json -m model1 model2 ...` → multi-model (benchmark.py)
    # `arcval llm --verify --agent-url URL` → verify external agent connection
    llm_parser = subparsers.add_parser(
        "llm",
        help="LLM evaluation — test agent responses and tool calls",
    )
    llm_parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to test config JSON file"
    )
    llm_parser.add_argument(
        "-o", "--output-dir", type=str, default="./out", help="Output directory"
    )
    llm_parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs="+",
        help="Model(s) to use for evaluation (space-separated for multiple)",
    )
    llm_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="openrouter",
        choices=["openai", "openrouter"],
        help="LLM provider",
    )
    llm_parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=None,
        help="Number of test cases to evaluate in parallel per model",
    )
    llm_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode: evaluate only the first N test cases (see --debug_count)",
    )
    llm_parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of test cases to evaluate in debug mode (default: 5)",
    )
    llm_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force a clean run instead of resuming completed test cases from a prior results.json",
    )
    llm_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an external agent connection by sending a preset message and checking the response format",
    )
    llm_parser.add_argument(
        "--agent-url",
        type=str,
        default=None,
        help="External agent endpoint URL (required with --verify)",
    )
    llm_parser.add_argument(
        "--agent-headers",
        type=str,
        default=None,
        help='HTTP headers for the agent as a JSON string, e.g. \'{"Authorization": "Bearer sk-..."}\'',
    )
    llm_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip agent connection verification (used internally when already verified)",
    )
    llm_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip LLM inference and run evaluators on a dataset of (test_case, output) pairs",
    )
    llm_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON for --eval-only (list of {test_case, output} items)",
    )
    # ── Simulations ─────────────────────────────────────────────
    # `arcval simulations` with no args → interactive UI
    # `arcval simulations --type text -c config.json ...` → run directly
    # `arcval simulations --verify --agent-url URL` → verify external agent
    sim_parser = subparsers.add_parser(
        "simulations",
        help="Run text or voice simulations",
    )
    sim_parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=None,
        choices=["text", "voice"],
        help="Simulation type: text or voice",
    )
    sim_parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to simulation config JSON"
    )
    sim_parser.add_argument(
        "-o", "--output-dir", type=str, default="./out", help="Output directory"
    )
    sim_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model name (text simulations)",
    )
    sim_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="openrouter",
        choices=["openai", "openrouter"],
        help="LLM provider (text simulations)",
    )
    sim_parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=1,
        help="Number of simulations to run in parallel",
    )
    sim_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an external agent connection by sending a preset message and checking the response format",
    )
    sim_parser.add_argument(
        "--agent-url",
        type=str,
        default=None,
        help="External agent endpoint URL (required with --verify)",
    )
    sim_parser.add_argument(
        "--agent-headers",
        type=str,
        default=None,
        help='HTTP headers for the agent as a JSON string, e.g. \'{"Authorization": "Bearer sk-..."}\'',
    )
    sim_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip agent connection verification (used internally when already verified)",
    )
    sim_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip simulation and run evaluators on a dataset of pre-existing transcripts (text simulation only)",
    )
    sim_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON for --eval-only (list of {conversation_history, name?})",
    )

    # Hidden internal subcommand for simulation leaderboard
    sim_subparsers = sim_parser.add_subparsers(dest="sim_subcmd", metavar="")
    sim_lb_parser = sim_subparsers.add_parser("leaderboard")
    sim_lb_parser.add_argument("-o", "--output-dir", type=str, required=True)
    sim_lb_parser.add_argument("-s", "--save-dir", type=str, required=True)

    # ── General task eval ───────────────────────────────────────
    # `arcval general --dataset data.json --config config.json` →
    # score a dataset of {id, input, output} rows with the general
    # (non-conversational) task judge.
    general_parser = subparsers.add_parser(
        "general",
        help="General task evaluation — judge arbitrary input/output pairs",
    )
    general_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON (list of {id, input, output})",
    )
    general_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file defining the `evaluators` list",
    )
    general_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Output directory",
    )

    # ── Status ────────────────────────────────────────────────────
    status_parser = subparsers.add_parser(
        "status",
        help="Check API key configuration and provider connectivity",
    )
    status_parser.add_argument(
        "--table",
        action="store_true",
        default=False,
        help="Display results as a formatted table instead of JSON",
    )

    # ── Agent test (hidden — interactive voice testing) ─────────
    agent_parser = subparsers.add_parser("agent")
    agent_subparsers = agent_parser.add_subparsers(dest="command", help="Agent command")
    agent_subparsers.required = True

    agent_test_parser = agent_subparsers.add_parser(
        "test", help="Run interactive agent test"
    )
    agent_test_parser.add_argument("-c", "--config", type=str, required=True)
    agent_test_parser.add_argument("-o", "--output-dir", type=str, default="./out")

    # ─────────────────────────────────────────────────────────────
    args = parser.parse_args()

    # ── Slack notification setup ──────────────────────────────
    slack_enabled = not _is_interactive_run(args) and not _skip_slack(args)
    desc = _build_cmd_desc(args) if slack_enabled else ""

    if slack_enabled:
        _send_slack(
            f"⚙️ Arcval Run Started\n"
            f"• Command: `{desc}`\n"
            f"• Output: `{os.path.abspath(args.output_dir)}`"
        )

    def _dispatch(args):
        # No component specified → launch main menu UI
        if args.component is None:
            _launch_ink_ui("menu")

        # ── Dispatch ────────────────────────────────────────────────
        if args.component == "stt":
            # Resolve intent/entity flag — prompt on TTY when not set
            _skip_ie = args.skip_intent_entity
            if _skip_ie is None:
                if sys.stdin.isatty():
                    _resp = input("Skip intent/entity judge? [Y/n]: ").strip().lower()
                    _skip_ie = _resp not in ("n", "no")
                else:
                    _skip_ie = True
                print()  # spacing after prompt
            _skip_llm = args.skip_llm_judge
            if _skip_llm is None:
                _skip_llm = True

            # eval-only: skip STT inference and run evaluators on a (gt, pred) dataset.
            # benchmark: --provider given → run inference + evaluators.
            # neither: launch interactive UI.
            if args.eval_only:
                from arcval.stt.benchmark import main as stt_benchmark_main

                if not args.dataset:
                    print(
                        "\033[31mError: --dataset is required with --eval-only\033[0m"
                    )
                    sys.exit(1)

                argv = ["arcval", "--eval-only", "--dataset", args.dataset]
                argv.extend(["-o", args.output_dir])
                if args.config:
                    argv.extend(["--config", args.config])
                if _skip_llm:
                    argv.append("--skip-llm-judge")
                else:
                    argv.append("--no-skip-llm-judge")
                if _skip_ie:
                    argv.append("--skip-intent-entity")
                else:
                    argv.append("--no-skip-intent-entity")

                sys.argv = argv
                asyncio.run(stt_benchmark_main())
            elif args.provider is not None:
                from arcval.stt.benchmark import main as stt_benchmark_main

                providers = args.provider
                argv = ["arcval", "-p"] + providers
                argv.extend(["-l", args.language])
                argv.extend(["-i", args.input_dir])
                argv.extend(["-o", args.output_dir])
                argv.extend(["-f", args.input_file_name])
                if args.debug:
                    argv.append("-d")
                argv.extend(["-dc", str(args.debug_count)])
                if args.ignore_retry:
                    argv.append("--ignore_retry")
                if args.overwrite:
                    argv.append("--overwrite")
                if args.save_dir:
                    argv.extend(["-s", args.save_dir])
                if args.config:
                    argv.extend(["--config", args.config])
                if _skip_llm:
                    argv.append("--skip-llm-judge")
                else:
                    argv.append("--no-skip-llm-judge")
                if _skip_ie:
                    argv.append("--skip-intent-entity")
                else:
                    argv.append("--no-skip-intent-entity")

                sys.argv = argv
                asyncio.run(stt_benchmark_main())
            else:
                _launch_ink_ui("stt")

        elif args.component == "arctan-eval":
            _skip_ie = args.skip_intent_entity
            if _skip_ie is None:
                if sys.stdin.isatty():
                    _resp = input("Skip intent/entity judge? [Y/n]: ").strip().lower()
                    _skip_ie = _resp not in ("n", "no")
                else:
                    _skip_ie = True
                print()

            from arcval.arctan_eval.benchmark import main as arctan_eval_main

            argv = ["arcval"]
            if args.provider:
                argv.extend(["-p"] + args.provider)
            argv.extend(["-l", args.language])
            if args.input_dir:
                argv.extend(["-i", args.input_dir])
            argv.extend(["-o", args.output_dir])
            argv.extend(["-f", args.input_file_name])
            if args.debug:
                argv.append("-d")
            argv.extend(["-dc", str(args.debug_count)])
            if args.ignore_retry:
                argv.append("--ignore_retry")
            if args.overwrite:
                argv.append("--overwrite")
            if args.config:
                argv.extend(["--config", args.config])
            if args.skip_llm_judge:
                argv.append("--skip-llm-judge")
            if _skip_ie:
                argv.append("--skip-intent-entity")
            else:
                argv.append("--no-skip-intent-entity")

            sys.argv = argv
            asyncio.run(arctan_eval_main())

        elif args.component == "tts":
            # If provider is given, run evaluation directly; otherwise launch interactive UI
            if args.provider is not None:
                from arcval.tts.benchmark import main as tts_benchmark_main

                providers = args.provider
                argv = ["arcval", "-p"] + providers
                argv.extend(["-l", args.language])
                argv.extend(["-i", args.input])
                argv.extend(["-o", args.output_dir])
                if args.debug:
                    argv.append("-d")
                argv.extend(["-dc", str(args.debug_count)])
                if args.overwrite:
                    argv.append("--overwrite")
                if args.config:
                    argv.extend(["--config", args.config])

                sys.argv = argv
                asyncio.run(tts_benchmark_main())
            else:
                _launch_ink_ui("tts")

        elif args.component == "llm":
            if getattr(args, "eval_only", False):
                if args.config is None:
                    print("Error: --config is required with --eval-only")
                    sys.exit(1)
                if not getattr(args, "dataset", None):
                    print("Error: --dataset is required with --eval-only")
                    sys.exit(1)

                from arcval.llm.run_tests import main as llm_run_tests_main

                argv = [
                    "arcval",
                    "-c",
                    args.config,
                    "-o",
                    args.output_dir,
                    "--eval-only",
                    "--dataset",
                    args.dataset,
                ]
                if getattr(args, "parallel", None) is not None:
                    argv.extend(["-n", str(args.parallel)])
                if getattr(args, "debug", False):
                    argv.append("-d")
                    argv.extend(["-dc", str(args.debug_count)])
                sys.argv = argv
                asyncio.run(llm_run_tests_main())
            elif getattr(args, "verify", False):
                if not args.agent_url:
                    print("Error: --agent-url is required with --verify")
                    sys.exit(1)
                _run_agent_verify(
                    args.agent_url,
                    args.agent_headers,
                    models=args.model,
                )
            elif args.config is None:
                # No config → interactive mode
                _launch_ink_ui("llm")
            else:
                # Direct mode: run tests with provided config
                import json as _json

                from arcval.utils import apply_debug_limit

                with open(args.config) as _f:
                    _config = _json.load(_f)

                if getattr(args, "debug", False) and _config.get("test_cases"):
                    _config["test_cases"] = apply_debug_limit(
                        _config["test_cases"], True, args.debug_count
                    )

                if _config.get("agent_url"):
                    # Agent connection path
                    from arcval.connections import TextAgentConnection
                    from arcval.llm import tests as _tests

                    _agent = TextAgentConnection(
                        url=_config["agent_url"],
                        headers=_config.get("agent_headers"),
                    )
                    _models = args.model if args.model else []

                    # Verify once per model (skip if already verified upstream e.g. interactive UI)
                    if not getattr(args, "skip_verify", False):
                        _models_to_verify = _models if _models else [None]
                        for _m in _models_to_verify:
                            _label = f"model: {_m}" if _m else "connection"
                            print(f"\nVerifying agent {_label}: {_config['agent_url']}")
                            _verify_result = asyncio.run(_agent.verify(model=_m))
                            if not _verify_result["ok"]:
                                print(
                                    f"✗ Verification failed: {_verify_result['error']}"
                                )
                                _print_sample_output(_verify_result)
                                sys.exit(1)
                            print(f"✓ Verified")
                            _print_sample_output(_verify_result)
                            print()

                    from arcval.llm._output import print_benchmark_summary
                    from arcval.llm.tests_leaderboard import generate_leaderboard

                    # Run — all models together (tests.run fans them out in parallel)
                    if _models:
                        print(f"\n\033[92m{'=' * 60}\033[0m")
                        print(f"\033[92m  Models: {', '.join(_models)}\033[0m")
                        print(f"\033[92m{'=' * 60}\033[0m\n")
                        _results = asyncio.run(
                            _tests.run(
                                agent=_agent,
                                test_cases=_config["test_cases"],
                                output_dir=args.output_dir,
                                models=_models,
                                evaluators=_config.get("evaluators"),
                                test_parallel=args.parallel,
                                overwrite=args.overwrite,
                            )
                        )
                        _model_results = {
                            _m: _results.get(_m, _results) for _m in _models
                        }

                        _lb_dir = os.path.join(args.output_dir, "leaderboard")
                        generate_leaderboard(
                            output_dir=args.output_dir, save_dir=_lb_dir
                        )
                        _has_errors = print_benchmark_summary(
                            models=_models,
                            model_results=_model_results,
                            leaderboard_dir=_lb_dir,
                        )
                        if _has_errors:
                            sys.exit(1)
                    else:
                        asyncio.run(
                            _tests.run(
                                agent=_agent,
                                test_cases=_config["test_cases"],
                                output_dir=args.output_dir,
                                evaluators=_config.get("evaluators"),
                                test_parallel=args.parallel,
                                overwrite=args.overwrite,
                            )
                        )
                else:
                    from arcval.llm.benchmark import main as llm_benchmark_main

                    models = args.model if args.model else ["gpt-4.1"]

                    argv = ["arcval", "-c", args.config]
                    argv.extend(["-o", args.output_dir])
                    argv.extend(["-m"] + models)
                    argv.extend(["-p", args.provider])
                    if getattr(args, "parallel", None) is not None:
                        argv.extend(["-n", str(args.parallel)])
                    if getattr(args, "overwrite", False):
                        argv.append("--overwrite")
                    if getattr(args, "debug", False):
                        argv.append("-d")
                        argv.extend(["-dc", str(args.debug_count)])

                    sys.argv = argv
                    asyncio.run(llm_benchmark_main())

        elif args.component == "simulations":
            if getattr(args, "verify", False):
                if not args.agent_url:
                    print("Error: --agent-url is required with --verify")
                    sys.exit(1)
                _model_str = getattr(args, "model", None)
                _run_agent_verify(
                    args.agent_url,
                    args.agent_headers,
                    models=[_model_str] if _model_str else None,
                )
            # Hidden leaderboard subcommand (used by Ink UI)
            elif getattr(args, "sim_subcmd", None) == "leaderboard":
                from arcval.llm.simulation_leaderboard import (
                    main as leaderboard_main,
                )

                sys.argv = ["arcval"] + _args_to_argv(
                    args,
                    exclude_keys={
                        "component",
                        "sim_subcmd",
                        "type",
                        "config",
                        "model",
                        "provider",
                        "parallel",
                        "port",
                    },
                )
                leaderboard_main()
            elif args.type is None or args.config is None:
                # Missing type or config → interactive mode
                _launch_ink_ui("simulations")
            elif args.type == "text":
                from arcval.llm.run_simulation import main as llm_simulation_main

                # Eval-only: skip agent verification entirely; the dataset already
                # contains the transcripts and we only run evaluators on them.
                if getattr(args, "eval_only", False):
                    if not getattr(args, "dataset", None):
                        print("Error: --dataset is required with --eval-only")
                        sys.exit(1)
                    sys.argv = ["arcval"] + _args_to_argv(
                        args,
                        exclude_keys={
                            "component",
                            "sim_subcmd",
                            "type",
                            "skip_verify",
                        },
                    )
                    asyncio.run(llm_simulation_main())
                    return

                # Pre-verify agent connection if config has agent_url
                if args.config:
                    import json as _json

                    with open(args.config) as _f:
                        _sim_config = _json.load(_f)
                    if _sim_config.get("agent_url") and not getattr(
                        args, "skip_verify", False
                    ):
                        from arcval.connections import TextAgentConnection

                        _sim_agent = TextAgentConnection(
                            url=_sim_config["agent_url"],
                            headers=_sim_config.get("agent_headers"),
                        )
                        print(
                            f"\nVerifying agent connection: {_sim_config['agent_url']}"
                        )
                        _verify = asyncio.run(_sim_agent.verify())
                        if not _verify["ok"]:
                            print(f"✗ Verification failed: {_verify['error']}")
                            _print_sample_output(_verify)
                            sys.exit(1)
                        print("✓ Verified")
                        _print_sample_output(_verify)
                        print()

                sys.argv = ["arcval"] + _args_to_argv(
                    args,
                    exclude_keys={"component", "sim_subcmd", "type", "skip_verify"},
                )
                asyncio.run(llm_simulation_main())
            elif args.type == "voice":
                from arcval.agent.run_simulation import main as agent_main

                sys.argv = ["arcval"] + _args_to_argv(
                    args,
                    exclude_keys={
                        "component",
                        "sim_subcmd",
                        "type",
                        "model",
                        "provider",
                    },
                )
                asyncio.run(agent_main())

        elif args.component == "general":
            from arcval.general.eval import main as general_eval_main

            if not args.dataset:
                print("\033[31mError: --dataset is required\033[0m")
                sys.exit(1)
            if not args.config:
                print("\033[31mError: --config is required\033[0m")
                sys.exit(1)

            argv = ["arcval", "--dataset", args.dataset, "-c", args.config]
            argv.extend(["-o", args.output_dir])
            sys.argv = argv
            asyncio.run(general_eval_main())

        elif args.component == "status":
            from arcval.status import run_status_live

            table_mode = getattr(args, "table", False)
            asyncio.run(run_status_live(table=table_mode))

        elif args.component == "agent":
            if args.command == "test":
                test_args = _args_to_argv(args, exclude_keys={"component", "command"})
                test_args = [
                    arg.replace("--output-dir", "--output_dir") for arg in test_args
                ]

                test_module_path = os.path.join(
                    os.path.dirname(__file__), "agent", "test.py"
                )
                sys.argv = ["arcval-test"] + test_args
                runpy.run_path(test_module_path, run_name="__main__")

        else:
            parser.print_help()
            sys.exit(1)

    try:
        _dispatch(args)
    except SystemExit as e:
        if slack_enabled:
            emoji = "✅" if e.code == 0 else "❌"
            status = "Complete" if e.code == 0 else "Failed"
            msg = (
                f"{emoji} Arcval Run {status}\n"
                f"• Command: `{desc}`\n"
                f"• Output: `{os.path.abspath(args.output_dir)}`\n"
                f"• Status: {'Success' if e.code == 0 else f'Error (exit {e.code})'}"
            )
            uploaded, upload_error = (False, None)
            if e.code == 0:
                uploaded, upload_error = _upload_slack_leaderboard(args.output_dir, msg)
            if not uploaded:
                if e.code == 0 and upload_error:
                    msg += f"\n• Leaderboard upload: {upload_error}"
                _send_slack(msg)
        raise

    except Exception as e:
        if slack_enabled:
            import traceback

            tb = traceback.format_exc()[:1500]
            _send_slack(
                f"❌ Arcval Run Failed\n"
                f"• Command: `{desc}`\n"
                f"• Output: `{os.path.abspath(args.output_dir)}`\n"
                f"• Error: `{e}`\n"
                f"```\n{tb}\n```"
            )
        sys.exit(1)

    else:
        if slack_enabled:
            msg = (
                f"✅ Arcval Run Complete\n"
                f"• Command: `{desc}`\n"
                f"• Output: `{os.path.abspath(args.output_dir)}`\n"
                f"• Status: Success"
            )
            uploaded, upload_error = _upload_slack_leaderboard(args.output_dir, msg)
            if not uploaded:
                if upload_error:
                    msg += f"\n• Leaderboard upload: {upload_error}"
                _send_slack(msg)


if __name__ == "__main__":
    main()
