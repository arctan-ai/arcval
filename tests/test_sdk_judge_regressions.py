"""
Regression tests for SDK-path judge behavior.

Covers two bugs flagged on PR #47:

1. Simulation SDK (`calibrate.llm.simulations.run`) must write a `type`
   field into each criterion in `metrics.json`. Without it, the simulation
   leaderboard's default-binary fallback multiplies rating means by 100
   (corrupting e.g. `4.2` → `420`).

2. Voice-sim SDK (`calibrate.agent.simulation.run_single`) must fall back
   to `judge.model` for the STT judge when `judge.stt_model` is omitted,
   matching the config-driven flow in
   `calibrate/agent/run_simulation.py::_run_single_simulation_inner`.
   Otherwise `simulation.run(...)` and `simulation.run_single(...)` can
   produce different STT scores for the same judge config.

Run with:
    python -m pytest tests/test_sdk_judge_regressions.py -v
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Bug 1: SDK simulation metrics.json must carry `type`
# ---------------------------------------------------------------------------


class TestSDKSimulationMetricsCarryType(unittest.IsolatedAsyncioTestCase):

    async def test_rating_criterion_aggregate_gets_type_field(self):
        """After `calibrate.llm.simulations.run(...)` completes, the saved
        metrics.json should include `type: "rating"` for each rating
        criterion so the leaderboard doesn't multiply its mean by 100."""
        from calibrate.llm import simulations

        # Fake `run_single_simulation_task` that returns the shape the SDK expects:
        # (simulation_metrics_row, evaluation_results)
        # Each evaluation_result carries {name, type, value, reasoning}.
        async def fake_task(
            semaphore, config, persona_index, user_persona, scenario_index,
            scenario, output_dir, args, agent=None,
        ):
            async with semaphore:
                sim_name = f"sim_{persona_index}_{scenario_index}"
                eval_results = [
                    {"name": "tool_usage", "type": "binary", "value": 1.0, "reasoning": "ok"},
                    {"name": "fluency", "type": "rating", "value": 4.0, "reasoning": "good"},
                ]
                sim_metrics = {"name": sim_name, "tool_usage": 1.0, "fluency": 4.0}
                return sim_metrics, eval_results

        with tempfile.TemporaryDirectory() as tmp, \
             patch(
                 "calibrate.llm.run_simulation.run_single_simulation_task",
                 side_effect=fake_task,
             ):
            await simulations.run(
                system_prompt="...",
                tools=[],
                personas=[{"characteristics": "friendly", "language": "english"}],
                scenarios=[{"description": "basic inquiry"}],
                evaluation_criteria=[
                    {"name": "tool_usage", "description": "calls tool"},
                    {
                        "name": "fluency",
                        "type": "rating",
                        "scale_min": 1,
                        "scale_max": 5,
                        "description": "rate fluency",
                    },
                ],
                output_dir=tmp,
                model="gpt-4.1",
                provider="openai",
            )

            # Find the model subfolder that was created (single model path uses _flat_output=True)
            with open(os.path.join(tmp, "metrics.json")) as f:
                metrics = json.load(f)

        self.assertIn("tool_usage", metrics)
        self.assertIn("fluency", metrics)
        self.assertEqual(metrics["tool_usage"]["type"], "binary")
        self.assertEqual(metrics["fluency"]["type"], "rating")
        # Raw means are preserved (no 100x scaling at this stage)
        self.assertAlmostEqual(metrics["fluency"]["mean"], 4.0)
        self.assertAlmostEqual(metrics["tool_usage"]["mean"], 1.0)


# ---------------------------------------------------------------------------
# Bug 2: judge.model falls back to STT judge in agent.simulation.run_single
# ---------------------------------------------------------------------------


class TestAgentSimulationRunSingleSTTFallback(unittest.IsolatedAsyncioTestCase):

    async def _call_run_single(self, judge):
        """Invoke agent.simulation.run_single with the given judge dict,
        capturing the kwargs passed to the underlying _run_simulation."""
        from calibrate.agent import simulation as agent_sim

        captured_kwargs = {}

        async def fake_run_simulation(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "transcript": [],
                "evaluation_results": [],
                "metrics": {},
                "stt_outputs": [],
                "tool_calls": [],
            }

        # Patch the symbol in calibrate.agent.__init__ namespace
        # (imported locally inside run_single)
        with patch(
            "calibrate.agent.run_simulation.run_simulation",
            side_effect=fake_run_simulation,
        ):
            await agent_sim.run_single(
                system_prompt="You are a user",
                language="english",
                gender="female",
                evaluation_criteria=[{"name": "x", "description": "y"}],
                output_dir="/tmp/any",
                judge=judge,
            )
        return captured_kwargs

    async def test_judge_model_propagates_to_stt_when_stt_model_missing(self):
        kwargs = await self._call_run_single({"model": "custom-judge-model"})
        self.assertEqual(kwargs.get("judge_model"), "custom-judge-model")
        # Critical: STT judge inherits from judge.model when stt_model is absent
        self.assertEqual(kwargs.get("stt_judge_model"), "custom-judge-model")

    async def test_explicit_stt_model_overrides_judge_model(self):
        kwargs = await self._call_run_single({
            "model": "primary-model",
            "stt_model": "stt-specific-model",
        })
        self.assertEqual(kwargs.get("judge_model"), "primary-model")
        self.assertEqual(kwargs.get("stt_judge_model"), "stt-specific-model")

    async def test_no_judge_dict_leaves_both_unset(self):
        kwargs = await self._call_run_single(None)
        self.assertNotIn("judge_model", kwargs)
        self.assertNotIn("stt_judge_model", kwargs)


if __name__ == "__main__":
    unittest.main()
