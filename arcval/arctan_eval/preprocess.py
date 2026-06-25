import os
import shutil
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from arctan import Processor, ProcessorConfig


def _require_license_key() -> str:
    license_key = os.environ.get("ARCTAN_SDK_KEY")
    if not license_key:
        raise RuntimeError("ARCTAN_SDK_KEY environment variable not set")
    return license_key


def isolate_wav(
    input_path: str | Path,
    output_path: str | Path,
    *,
    license_key: str | None = None,
) -> Path:
    """Run the Arctan voice isolator on one PCM16 WAV file."""
    src = Path(input_path)
    dst = Path(output_path)
    key = license_key or _require_license_key()

    with wave.open(str(src), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()
        raw = wf.readframes(total_frames)

    if sample_width != 2:
        raise ValueError(
            f"Expected a 16-bit PCM WAV file, received {sample_width * 8}-bit audio"
        )

    chunk_size = sample_rate // 100
    if chunk_size <= 0:
        raise ValueError(f"Invalid sample rate for chunking: {sample_rate}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    original_num_samples = len(audio)
    processor = Processor(
        license_key=key,
        config=ProcessorConfig(
            sample_rate=sample_rate,
            num_channels=1,
            num_frames=chunk_size,
        ),
    )

    output_chunks: list[np.ndarray] = []
    try:
        for start in range(0, original_num_samples, chunk_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            processed = np.asarray(processor.process(chunk.reshape(1, -1)))
            output_chunks.append(processed[0] if processed.ndim == 2 else processed)
    finally:
        processor.close()

    denoised = np.concatenate(output_chunks)[:original_num_samples]
    denoised_int16 = np.clip(
        np.rint(denoised * 32767.0),
        -32768,
        32767,
    ).astype(np.int16)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(denoised_int16.tobytes())

    return dst


def build_arctan_input_dir(
    input_dir: str,
    output_dir: str,
    *,
    input_file_name: str = "stt.csv",
    debug: bool = False,
    debug_count: int = 5,
    overwrite: bool = False,
) -> Path:
    """Create a derived STT input dir whose audio has been isolated by Arctan."""
    source_dir = Path(input_dir)
    derived_dir = Path(output_dir)
    source_csv = source_dir / input_file_name
    source_audio_dir = source_dir / "audios"

    if overwrite and derived_dir.exists():
        shutil.rmtree(derived_dir)

    derived_audio_dir = derived_dir / "audios"
    derived_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(source_csv)
    if debug:
        rows = rows.head(debug_count)

    rows.to_csv(derived_dir / input_file_name, index=False)

    license_key = _require_license_key()
    for row_id in rows["id"].tolist():
        src_audio = source_audio_dir / f"{row_id}.wav"
        dst_audio = derived_audio_dir / f"{row_id}.wav"
        if overwrite or not dst_audio.exists():
            isolate_wav(src_audio, dst_audio, license_key=license_key)

    return derived_dir
