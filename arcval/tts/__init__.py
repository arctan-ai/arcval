# arcval.tts module
"""
Text-to-Speech evaluation and benchmarking module.

Library Usage:
    from arcval.tts import run, run_single, generate_leaderboard

    # Run TTS benchmark across multiple providers (parallel + auto-leaderboard)
    import asyncio
    result = asyncio.run(run(
        providers=["google", "openai"],
        language="english",
        input="./data/sample.csv",
        output_dir="./out"
    ))

    # Run single provider evaluation (no leaderboard)
    result = asyncio.run(run_single(
        provider="google",
        language="english",
        input_file="./data/sample.csv",
        output_dir="./out"
    ))

    # Generate leaderboard separately
    generate_leaderboard(output_dir="./out", save_dir="./out/leaderboard")
"""

# Multi-provider benchmark (parallel execution + auto-leaderboard)
from arcval.tts.benchmark import run

# Single provider evaluation
from arcval.tts.eval import run_single_provider_eval as run_single

# Leaderboard generation
from arcval.tts.leaderboard import generate_leaderboard

# Validation utilities
from arcval.tts.eval import validate_tts_input_file

__all__ = ["run", "run_single", "generate_leaderboard", "validate_tts_input_file"]
