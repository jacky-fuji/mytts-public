from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "generation_benchmark" / "round1_h4n_ref"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "comparisons"

MODEL_ORDER = [
    "Irodori-TTS 500M",
    "Irodori-TTS 600M VoiceDesign",
    "VoxCPM2 ultimate",
    "CosyVoice2",
    "Qwen3-TTS 1.7B",
    "Fish Speech S2 Pro",
]

MODEL_LABELS = {
    "Irodori-TTS 500M": "Irodori-TTS 500M",
    "Irodori-TTS 600M VoiceDesign": "Irodori-TTS 600M VoiceDesign",
    "openbmb/VoxCPM2": "VoxCPM2 ultimate",
    "cosyvoice2": "CosyVoice2",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "Qwen3-TTS 1.7B",
    "fishaudio/s2-pro": "Fish Speech S2 Pro",
}

SAMPLE_ORDER = {
    "genbench_short_01": 0,
    "genbench_narration_01": 1,
    "genbench_technical_01": 2,
}


def project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def duration_sec(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def collect_rows(input_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest in sorted(input_root.rglob("generation_manifest.tsv")):
        for row in read_manifest(manifest):
            if not row.get("sample_id", "").startswith("genbench_"):
                continue
            output_wav = project_path(row["output_wav"])
            seconds_elapsed = float(row["seconds_elapsed"])
            audio_duration = float(row["duration_sec"] or 0.0) if row.get("duration_sec") else duration_sec(output_wav)
            model = MODEL_LABELS.get(row["model"], row["model"])
            rtf = seconds_elapsed / audio_duration if audio_duration > 0 else 0.0
            wall_seconds = float(row.get("wall_seconds") or seconds_elapsed)
            rows.append(
                {
                    "model": model,
                    "engine": row["engine"],
                    "ref_profile": row["ref_profile"],
                    "sample_id": row["sample_id"],
                    "category": row["category"],
                    "seconds_elapsed": f"{seconds_elapsed:.3f}",
                    "wall_seconds": f"{wall_seconds:.3f}",
                    "duration_sec": f"{audio_duration:.3f}",
                    "rtf": f"{rtf:.3f}",
                    "output_wav": str(output_wav.relative_to(PROJECT_ROOT)),
                    "text_file": row["text_file"],
                    "notes": row["notes"],
                }
            )
    return sorted(rows, key=lambda row: (MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 999, SAMPLE_ORDER.get(row["sample_id"], 999)))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["model"]].append(row)

    summary: list[dict[str, str]] = []
    for model, model_rows in buckets.items():
        count = len(model_rows)
        total_elapsed = sum(float(row["seconds_elapsed"]) for row in model_rows)
        total_wall = sum(float(row["wall_seconds"]) for row in model_rows)
        total_duration = sum(float(row["duration_sec"]) for row in model_rows)
        avg_rtf = total_elapsed / total_duration if total_duration > 0 else 0.0
        avg_wall_rtf = total_wall / total_duration if total_duration > 0 else 0.0
        summary.append(
            {
                "model": model,
                "sample_count": str(count),
                "total_seconds_elapsed": f"{total_elapsed:.3f}",
                "total_wall_seconds": f"{total_wall:.3f}",
                "total_audio_duration_sec": f"{total_duration:.3f}",
                "avg_rtf": f"{avg_rtf:.3f}",
                "avg_wall_rtf": f"{avg_wall_rtf:.3f}",
            }
        )
    return sorted(summary, key=lambda row: float(row["avg_rtf"]))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize generation-time benchmark manifests.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    input_root = args.input_root if args.input_root.is_absolute() else PROJECT_ROOT / args.input_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    rows = collect_rows(input_root)
    if not rows:
        raise RuntimeError(f"No generation benchmark rows found under {input_root}")
    summary = summarize(rows)

    write_tsv(
        output_dir / "generation_time_benchmark_results.tsv",
        rows,
        [
            "model",
            "engine",
            "ref_profile",
            "sample_id",
            "category",
            "seconds_elapsed",
            "wall_seconds",
            "duration_sec",
            "rtf",
            "output_wav",
            "text_file",
            "notes",
        ],
    )
    write_tsv(
        output_dir / "generation_time_benchmark_summary.tsv",
        summary,
        [
            "model",
            "sample_count",
            "total_seconds_elapsed",
            "total_wall_seconds",
            "total_audio_duration_sec",
            "avg_rtf",
            "avg_wall_rtf",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
