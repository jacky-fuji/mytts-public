from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "report_audio_loudness_manifest.json"
AUDIO_ROOTS = [ROOT / "samples", ROOT / "outputs"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

TARGET_RMS_DBFS = -20.0
PEAK_CEILING_DBFS = -1.0
MAX_BOOST_DB = 12.0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return 20.0 * math.log10(value)


def analyze_audio(path: Path) -> dict:
    with sf.SoundFile(str(path)) as audio_file:
        rate = audio_file.samplerate
        channels = audio_file.channels
        frames = len(audio_file)

        total_square = 0.0
        total_samples = 0
        peak = 0.0
        for block in audio_file.blocks(blocksize=65536, dtype="float64", always_2d=True):
            if block.size == 0:
                continue
            abs_block = np.abs(block)
            total_square += float(np.sum(block * block))
            total_samples += int(block.size)
            peak = max(peak, float(np.max(abs_block)))

    if total_samples == 0:
        return {
            "durationSec": 0.0,
            "sampleRate": 0,
            "channels": 0,
            "rmsDbfs": None,
            "peakDbfs": None,
            "gainDb": 0.0,
            "gain": 1.0,
            "mode": "silent",
        }

    rms_value = math.sqrt(total_square / total_samples)
    rms_db = dbfs(rms_value)
    peak_db = dbfs(peak)

    if rms_db is None or peak_db is None:
        gain_db = 0.0
        mode = "silent"
    else:
        gain_db = TARGET_RMS_DBFS - rms_db
        gain_db = min(gain_db, PEAK_CEILING_DBFS - peak_db)
        gain_db = min(gain_db, MAX_BOOST_DB)
        mode = "rms_gain_peak_limited"

    return {
        "durationSec": round(frames / rate, 3) if rate else 0.0,
        "sampleRate": rate,
        "channels": channels,
        "rmsDbfs": round(rms_db, 2) if rms_db is not None else None,
        "peakDbfs": round(peak_db, 2) if peak_db is not None else None,
        "gainDb": round(gain_db, 2),
        "gain": round(10 ** (gain_db / 20.0), 6),
        "mode": mode,
    }


def main() -> None:
    files = sorted(
        path
        for root in AUDIO_ROOTS
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )
    manifest: dict[str, dict] = {}
    skipped: list[str] = []
    for path in files:
        try:
            manifest[rel(path)] = analyze_audio(path)
        except Exception as exc:
            skipped.append(rel(path))
            manifest[rel(path)] = {
                "gainDb": 0.0,
                "gain": 1.0,
                "mode": f"error:{type(exc).__name__}",
            }

    payload = {
        "targetRmsDbfs": TARGET_RMS_DBFS,
        "peakCeilingDbfs": PEAK_CEILING_DBFS,
        "maxBoostDb": MAX_BOOST_DB,
        "method": "linear playback gain based on WAV RMS, limited by peak headroom; original files are unchanged",
        "files": manifest,
        "skippedOrUnmeasured": skipped,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={OUT}")
    print(f"files={len(files)}")
    print(f"measured={len(files) - len(skipped)}")
    print(f"unmeasured={len(skipped)}")


if __name__ == "__main__":
    main()
