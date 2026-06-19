from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("〇", "0")
    text = re.sub(r"[\s　、。，．,.!?！？「」『』（）()\[\]【】・:：;；|｜\-ー]", "", text)
    return text


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def score(expected: str, actual: str) -> tuple[float, float]:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    if not expected_norm:
        return 0.0, 1.0
    distance = edit_distance(expected_norm, actual_norm)
    return SequenceMatcher(None, expected_norm, actual_norm).ratio(), distance / len(expected_norm)


def read_samples(sample_dir: Path) -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8").strip() for path in sample_dir.glob("*.txt")}


def find_sample_name(wav_path: Path, samples: dict[str, str]) -> str:
    stem = wav_path.stem
    for sample_name in sorted(samples, key=len, reverse=True):
        if sample_name in stem:
            return sample_name
    return ""


def transcribe_one(model, wav_path: Path, language: str, use_itn: bool) -> str:
    result = model.generate(
        input=str(wav_path),
        cache={},
        language=language,
        use_itn=use_itn,
        batch_size=1,
    )
    if not result:
        return ""
    return rich_transcription_postprocess(result[0].get("text", "")).strip()


def duration_sec(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return round(info.frames / info.samplerate, 3)


def model_label(wav_path: Path, input_root: Path) -> str:
    try:
        rel = wav_path.relative_to(input_root)
    except ValueError:
        rel = wav_path
    known_models = {
        "cosyvoice2",
        "cosyvoice3",
        "fish-speech",
        "qwen3-tts",
        "voxcpm2",
        "irodori-tts-500m-v3",
        "irodori-tts-600m-v3-voicedesign",
    }
    for part in rel.parts:
        if part.lower() in known_models:
            return part
    return rel.parts[0] if len(rel.parts) > 1 else wav_path.parent.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe generated TTS outputs and compare them with sample text.")
    parser.add_argument("--input-root", default=str(PROJECT_ROOT / "outputs" / "round2_improvement"))
    parser.add_argument("--sample-dir", default=str(PROJECT_ROOT / "samples" / "text"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "transcripts"))
    parser.add_argument("--output-prefix", default="round2_generated_asr")
    parser.add_argument("--model", default="FunAudioLLM/SenseVoiceSmall")
    parser.add_argument("--hub", choices=["hf", "ms"], default="hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--use-itn", action="store_true")
    parser.add_argument("--include-unmatched", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(Path(args.sample_dir))
    wavs = sorted(input_root.rglob("*.wav"))

    model = AutoModel(model=args.model, device=args.device, hub=args.hub)
    rows: list[dict[str, str]] = []
    for wav_path in wavs:
        sample = find_sample_name(wav_path, samples)
        if not sample and not args.include_unmatched:
            print(f"Skipping unmatched sample: {wav_path}", flush=True)
            continue
        expected = samples.get(sample, "")
        print(f"Transcribing {wav_path}", flush=True)
        transcript = transcribe_one(model, wav_path, args.language, args.use_itn)
        ratio, cer = score(expected, transcript)
        rows.append(
            {
                "model": model_label(wav_path, input_root),
                "sample": sample,
                "duration_sec": f"{duration_sec(wav_path):.3f}",
                "ratio": f"{ratio:.4f}",
                "cer": f"{cer:.4f}",
                "needs_review": "yes" if ratio < 0.72 or cer > 0.45 or not sample else "no",
                "wav": str(wav_path.relative_to(PROJECT_ROOT)),
                "expected": expected,
                "transcript": transcript,
            }
        )

    tsv_path = output_dir / f"{args.output_prefix}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "sample", "duration_sec", "ratio", "cer", "needs_review", "wav", "expected", "transcript"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = summary.setdefault(row["model"], {"count": 0, "review": 0, "ratio_sum": 0.0, "duration_sum": 0.0})
        bucket["count"] += 1
        bucket["review"] += 1 if row["needs_review"] == "yes" else 0
        bucket["ratio_sum"] += float(row["ratio"])
        bucket["duration_sum"] += float(row["duration_sec"])

    md_path = output_dir / f"{args.output_prefix}_summary.md"
    worst = sorted(rows, key=lambda row: float(row["ratio"]))[:25]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated ASR Summary\n\n")
        handle.write("| Model | Count | Review | Avg Ratio | Avg Duration |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for name, bucket in sorted(summary.items()):
            count = bucket["count"]
            handle.write(
                f"| {name} | {int(count)} | {int(bucket['review'])} | "
                f"{bucket['ratio_sum'] / count:.4f} | {bucket['duration_sum'] / count:.2f}s |\n"
            )
        handle.write("\n## Lowest ASR Matches\n\n")
        handle.write("| Model | Sample | Ratio | CER | Duration | WAV | Transcript |\n")
        handle.write("|---|---|---:|---:|---:|---|---|\n")
        for row in worst:
            handle.write(
                f"| {row['model']} | {row['sample']} | {row['ratio']} | {row['cer']} | "
                f"{row['duration_sec']} | `{row['wav']}` | {row['transcript']} |\n"
            )

    print(f"wrote\t{tsv_path}")
    print(f"wrote\t{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
