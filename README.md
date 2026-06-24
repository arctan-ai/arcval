# Arcval

[![WhatsApp](https://img.shields.io/badge/WhatsApp-Join%20Community-25D366?logo=whatsapp&logoColor=white)](https://chat.whatsapp.com/JygDNcZ943a3VmZDXYMg5Z)
[![codecov](https://codecov.io/gh/ARTPARK-SAHAI-ORG/arcval/branch/main/graph/badge.svg)](https://codecov.io/gh/ARTPARK-SAHAI-ORG/arcval)
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

CLI for [Arcval](https://arcval.artpark.ai): an AI evaluation platform for non-profits

With [Arcval](https://arcval.artpark.ai), you can move from slow, manual testing to a fast, automated, and repeatable testing process for your entire agent stack:

- `Text to Text (LLMs)`: Evaluate the response quality and tool calling of your LLMs for multi-turn conversations and find the find LLM for your agent
- `Human alignment`: Create LLM judges to make your evaluations scalable and reliable with human in the loop.
- `Speech to Text (STT)`: Benchmark multiple providers (Google, Sarvam, ElevenLabs and more) on your dataset across 10+ indic languages using metrics optimised for agentic use cases
- `Text to Speech (TTS)`: Benchmark generated speech by multiple providers automatically using an Audio LLM Judge across 10+ indic languages
- `Simulations`: Simulate realistic conversations using realistic user personas and scenarios to test failure modes for your agent (including interruptions for voice agents)

Arcval is built on top of [pipecat](https://github.com/pipecat-ai/pipecat), a framework for building agents.

## Installation

```bash
pip install arcval-agent
```

## Usage

```bash
arcval              # Interactive main menu
arcval stt          # Benchmark STT providers
arcval tts          # Benchmark TTS providers
arcval llm          # Interactive LLM evaluation
arcval simulations  # Interactive text or voice simulations
```

- [CLI Documentation](https://arcval.artpark.ai/docs/cli/overview)

## Contributing

Install development dependencies once (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --extra dev
```

### Running tests

Run the full test suite:

```bash
uv run pytest tests/
```

### Pre-commit

Enable the project's git hooks so the pre-commit test
runner fires on commits to `main`:

```bash
git config core.hooksPath .githooks
```

Every contributor needs to run it once.

## License

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
