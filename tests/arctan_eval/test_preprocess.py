import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def _write_wav(
    path: Path,
    samples: np.ndarray,
    *,
    channels: int,
    sample_width: int = 2,
    sample_rate: int = 16000,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class FakeProcessor:
    instances = []

    def __init__(self, license_key, config):
        self.license_key = license_key
        self.config = config
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def process(self, chunk):
        self.calls.append(np.array(chunk, copy=True))
        return chunk

    def close(self):
        self.closed = True


class TestArctanPreprocess(unittest.TestCase):
    def setUp(self):
        FakeProcessor.instances = []

    def test_missing_license_key_raises(self):
        from arcval.arctan_eval.preprocess import isolate_wav

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            _write_wav(src, np.array([0, 1, -1], dtype=np.int16), channels=1)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "ARCTAN_SDK_KEY"):
                    isolate_wav(src, Path(tmp) / "out.wav")

    def test_non_pcm16_input_raises(self):
        from arcval.arctan_eval.preprocess import isolate_wav

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            _write_wav(
                src, np.array([0, 1, 2], dtype=np.uint8), channels=1, sample_width=1
            )
            with patch.dict(os.environ, {"ARCTAN_SDK_KEY": "test-key"}):
                with self.assertRaisesRegex(ValueError, "16-bit PCM"):
                    isolate_wav(src, Path(tmp) / "out.wav")

    def test_isolate_wav_writes_mono_pcm16_and_closes_processor(self):
        from arcval.arctan_eval import preprocess

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            dst = Path(tmp) / "out.wav"
            _write_wav(
                src,
                np.array([100, -100, 50, -50], dtype=np.int16),
                channels=1,
            )
            with (
                patch.dict(os.environ, {"ARCTAN_SDK_KEY": "test-key"}),
                patch.object(
                    preprocess,
                    "Processor",
                    FakeProcessor,
                ),
                patch.object(preprocess, "ProcessorConfig", lambda **kwargs: kwargs),
            ):
                preprocess.isolate_wav(src, dst)

            self.assertTrue(dst.exists())
            with wave.open(str(dst), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(len(FakeProcessor.instances), 1)
            self.assertTrue(FakeProcessor.instances[0].closed)
            self.assertGreaterEqual(len(FakeProcessor.instances[0].calls), 1)

    def test_stereo_input_is_mixed_to_mono(self):
        from arcval.arctan_eval import preprocess

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.wav"
            dst = Path(tmp) / "out.wav"
            stereo = np.array([1000, 3000, -2000, 2000], dtype=np.int16)
            _write_wav(src, stereo, channels=2)
            with (
                patch.dict(os.environ, {"ARCTAN_SDK_KEY": "test-key"}),
                patch.object(
                    preprocess,
                    "Processor",
                    FakeProcessor,
                ),
                patch.object(preprocess, "ProcessorConfig", lambda **kwargs: kwargs),
            ):
                preprocess.isolate_wav(src, dst)

            with wave.open(str(dst), "rb") as wf:
                out = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            self.assertEqual(out.tolist(), [2000, 0])

    def test_build_arctan_input_dir_copies_csv_and_honors_debug_and_overwrite(self):
        from arcval.arctan_eval import preprocess

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "input"
            audio_dir = input_dir / "audios"
            audio_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"id": "row_a", "text": "hello"},
                    {"id": "row_b", "text": "world"},
                ]
            ).to_csv(input_dir / "stt.csv", index=False)
            _write_wav(
                audio_dir / "row_a.wav", np.array([1, 2], dtype=np.int16), channels=1
            )
            _write_wav(
                audio_dir / "row_b.wav", np.array([3, 4], dtype=np.int16), channels=1
            )

            calls = []

            def fake_isolate(src, dst, *, license_key=None):
                calls.append((Path(src).name, Path(dst).name, license_key))
                Path(dst).write_bytes(Path(src).read_bytes())
                return Path(dst)

            with (
                patch.dict(os.environ, {"ARCTAN_SDK_KEY": "test-key"}),
                patch.object(
                    preprocess,
                    "isolate_wav",
                    side_effect=fake_isolate,
                ),
            ):
                derived = preprocess.build_arctan_input_dir(
                    str(input_dir),
                    str(base / "derived"),
                    debug=True,
                    debug_count=1,
                    overwrite=False,
                )
                df = pd.read_csv(derived / "stt.csv")
                self.assertEqual(df["id"].tolist(), ["row_a"])
                self.assertEqual([call[0] for call in calls], ["row_a.wav"])

                (derived / "audios" / "row_a.wav").write_bytes(b"stale")
                preprocess.build_arctan_input_dir(
                    str(input_dir),
                    str(base / "derived"),
                    overwrite=True,
                )
                self.assertFalse(
                    (derived / "audios" / "row_a.wav").read_bytes() == b"stale"
                )
