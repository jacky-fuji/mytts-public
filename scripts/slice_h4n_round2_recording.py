from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str


@dataclass(frozen=True)
class Segment:
    index: int
    start_sec: float
    end_sec: float
    transcript: str = ""

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def normalize_text(text: str) -> str:
    text = text.lower()
    text = normalize_japanese_numbers(text)
    return re.sub(r"[\s　、。，．,.!?！？「」『』（）()\[\]【】・:：;；|｜\-ー]", "", text)


KANJI_DIGITS = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_under_10000(value: str) -> int:
    total = 0
    current: int | None = None
    for char in value:
        if char in KANJI_DIGITS:
            current = KANJI_DIGITS[char]
        elif char == "十":
            total += (1 if current is None else current) * 10
            current = None
        elif char == "百":
            total += (1 if current is None else current) * 100
            current = None
        elif char == "千":
            total += (1 if current is None else current) * 1000
            current = None
    if current is not None:
        total += current
    return total


def parse_japanese_number(value: str) -> str:
    if not value:
        return value
    if "点" in value:
        left, right = value.split("点", 1)
        right_digits = "".join(str(KANJI_DIGITS.get(char, char)) for char in right)
        return f"{parse_japanese_number(left)}点{right_digits}"
    if all(char in KANJI_DIGITS for char in value):
        return "".join(str(KANJI_DIGITS[char]) for char in value)
    if "万" in value:
        left, right = value.split("万", 1)
        total = parse_under_10000(left) * 10000
        if right:
            total += parse_under_10000(right)
        return str(total)
    return str(parse_under_10000(value))


def normalize_japanese_numbers(text: str) -> str:
    number_chars = "〇零一二三四五六七八九十百千万点"
    return re.sub(f"[{number_chars}]+", lambda match: parse_japanese_number(match.group(0)), text)


def score_text(expected: str, actual: str) -> float:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    if not expected_norm or not actual_norm:
        return 0.0
    return SequenceMatcher(None, expected_norm, actual_norm).ratio()


def read_prompts(path: Path, start_id: int, end_id: int) -> list[Prompt]:
    prompts: list[Prompt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        prompt_id, text = line.split("|", 1)
        number = int(prompt_id)
        if start_id <= number <= end_id:
            prompts.append(Prompt(prompt_id=prompt_id, text=text.strip()))
    expected = end_id - start_id + 1
    if len(prompts) != expected:
        raise ValueError(f"Expected {expected} prompts, found {len(prompts)} in {path}")
    return prompts


def db(value: float) -> float:
    if value <= 0:
        return -999.0
    return 20.0 * math.log10(value)


def detect_segments(
    mono: np.ndarray,
    sr: int,
    *,
    threshold_db: float,
    merge_gap_sec: float,
    min_duration_sec: float,
    pad_start_sec: float,
    pad_end_sec: float,
) -> tuple[list[Segment], float]:
    win = max(1, int(sr * 0.03))
    hop = max(1, int(sr * 0.01))
    rms: list[float] = []
    times: list[float] = []
    for start in range(0, len(mono) - win + 1, hop):
        frame = mono[start : start + win]
        rms.append(float(np.sqrt(np.mean(frame * frame))))
        times.append((start + win / 2) / sr)

    rms_arr = np.asarray(rms)
    noise_floor = float(np.percentile(rms_arr, 10))
    threshold = max(10 ** (threshold_db / 20.0), noise_floor * 8)
    active = rms_arr > threshold
    duration = len(mono) / sr

    raw: list[tuple[float, float]] = []
    in_segment = False
    start_sec = 0.0
    last_active_sec = 0.0
    for time_sec, is_active in zip(times, active):
        if is_active and not in_segment:
            in_segment = True
            start_sec = time_sec
            last_active_sec = time_sec
        elif is_active:
            last_active_sec = time_sec
        elif in_segment and time_sec - last_active_sec > merge_gap_sec:
            raw.append((start_sec, last_active_sec))
            in_segment = False
    if in_segment:
        raw.append((start_sec, last_active_sec))

    segments: list[Segment] = []
    for index, (start, end) in enumerate(raw, start=1):
        padded_start = max(0.0, start - pad_start_sec)
        padded_end = min(duration, end + pad_end_sec)
        if padded_end - padded_start >= min_duration_sec:
            segments.append(Segment(index=len(segments) + 1, start_sec=padded_start, end_sec=padded_end))
    return segments, threshold


def write_segment(path: Path, mono: np.ndarray, sr: int, segment: Segment) -> None:
    start = max(0, int(round(segment.start_sec * sr)))
    end = min(len(mono), int(round(segment.end_sec * sr)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono[start:end], sr, subtype="PCM_24")


def transcribe_one(model: AutoModel, wav_path: Path, language: str, use_itn: bool) -> str:
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


def align_segments(prompts: list[Prompt], segments: list[Segment]) -> list[tuple[int, int, float]]:
    m = len(prompts)
    n = len(segments)
    scores = np.zeros((m, n), dtype=np.float64)
    for i, prompt in enumerate(prompts):
        for j, segment in enumerate(segments):
            scores[i, j] = score_text(prompt.text, segment.transcript)

    dp = np.full((m + 1, n + 1), -1e9, dtype=np.float64)
    action: list[list[tuple[str, int, int] | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0, 0] = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            current = dp[i, j]
            if current < -1e8:
                continue
            if j < n and current >= dp[i, j + 1]:
                dp[i, j + 1] = current
                action[i][j + 1] = ("skip", i, j)
            if i < m and j < n:
                later_bias = 0.0005 * j
                value = current + scores[i, j] + later_bias
                if value > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = value
                    action[i + 1][j + 1] = ("take", i, j)

    i, j = m, n
    pairs: list[tuple[int, int, float]] = []
    while i > 0 or j > 0:
        step = action[i][j]
        if step is None:
            raise RuntimeError("Failed to backtrack alignment")
        kind, prev_i, prev_j = step
        if kind == "take":
            pairs.append((prev_i, prev_j, float(scores[prev_i, prev_j])))
        i, j = prev_i, prev_j
    pairs.reverse()
    return pairs


def prefer_later_retries(
    prompts: list[Prompt],
    segments: list[Segment],
    aligned: list[tuple[int, int, float]],
    *,
    max_score_drop: float = 0.08,
    min_retry_score: float = 0.10,
) -> list[tuple[int, int, float]]:
    adjusted: list[tuple[int, int, float]] = []
    for index, (prompt_index, segment_index, current_score) in enumerate(aligned):
        next_segment_index = aligned[index + 1][1] if index + 1 < len(aligned) else len(segments)
        best_segment_index = segment_index
        best_score = current_score
        for candidate_index in range(segment_index + 1, next_segment_index):
            candidate = segments[candidate_index]
            candidate_score = score_text(prompts[prompt_index].text, candidate.transcript)
            if candidate_score >= min_retry_score and candidate_score >= current_score - max_score_drop:
                best_segment_index = candidate_index
                best_score = candidate_score
        adjusted.append((prompt_index, best_segment_index, best_score))
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description="Slice a long Zoom H4n round2 recording into prompted sentence WAV files.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "samples" / "voice" / "MONO-000.wav"))
    parser.add_argument("--script", default=str(PROJECT_ROOT / "samples" / "voice_scripts" / "round2_recording_script_ja.txt"))
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=25)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "samples" / "voice" / "h4n_round2_wav"))
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--threshold-db", type=float, default=-44.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.8)
    parser.add_argument("--min-duration-sec", type=float, default=0.6)
    parser.add_argument("--pad-start-sec", type=float, default=0.15)
    parser.add_argument("--pad-end-sec", type=float, default=0.25)
    parser.add_argument("--asr-model", default="FunAudioLLM/SenseVoiceSmall")
    parser.add_argument("--hub", choices=["hf", "ms"], default="hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--use-itn", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    prompts = read_prompts(Path(args.script), args.start_id, args.end_id)
    audio, sr = sf.read(str(input_path), always_2d=True, dtype="float64")
    mono = audio.mean(axis=1)
    duration = len(mono) / sr
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono * mono)))

    segments, threshold = detect_segments(
        mono,
        sr,
        threshold_db=args.threshold_db,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        pad_start_sec=args.pad_start_sec,
        pad_end_sec=args.pad_end_sec,
    )

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else PROJECT_ROOT / "outputs" / "segmentation" / f"h4n_round2_{args.start_id:03d}_{args.end_id:03d}"
    )
    candidate_dir = work_dir / "candidates"
    final_dir = Path(args.output_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel(model=args.asr_model, device=args.device, hub=args.hub)
    transcribed: list[Segment] = []
    for segment in segments:
        wav_path = candidate_dir / f"cand_{segment.index:03d}.wav"
        write_segment(wav_path, mono, sr, segment)
        print(f"Transcribing candidate {segment.index:03d}: {segment.start_sec:.2f}-{segment.end_sec:.2f}s", flush=True)
        transcript = transcribe_one(model, wav_path, args.language, args.use_itn)
        transcribed.append(
            Segment(
                index=segment.index,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                transcript=transcript,
            )
        )

    aligned = prefer_later_retries(prompts, transcribed, align_segments(prompts, transcribed))
    chosen_by_segment = {segment_index for _, segment_index, _ in aligned}

    candidates_path = work_dir / "candidates.tsv"
    with candidates_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "start_sec", "end_sec", "duration_sec", "chosen", "transcript"],
            delimiter="\t",
        )
        writer.writeheader()
        for j, segment in enumerate(transcribed):
            writer.writerow(
                {
                    "candidate": f"cand_{segment.index:03d}",
                    "start_sec": f"{segment.start_sec:.3f}",
                    "end_sec": f"{segment.end_sec:.3f}",
                    "duration_sec": f"{segment.duration_sec:.3f}",
                    "chosen": "yes" if j in chosen_by_segment else "no",
                    "transcript": segment.transcript,
                }
            )

    manifest_rows: list[dict[str, str]] = []
    for prompt_index, segment_index, score in aligned:
        prompt = prompts[prompt_index]
        segment = transcribed[segment_index]
        output_path = final_dir / f"h4n_round2_{prompt.prompt_id}.wav"
        write_segment(output_path, mono, sr, segment)
        manifest_rows.append(
            {
                "id": prompt.prompt_id,
                "wav": str(output_path.relative_to(PROJECT_ROOT)),
                "source": str(input_path.relative_to(PROJECT_ROOT)),
                "candidate": f"cand_{segment.index:03d}",
                "start_sec": f"{segment.start_sec:.3f}",
                "end_sec": f"{segment.end_sec:.3f}",
                "duration_sec": f"{segment.duration_sec:.3f}",
                "score": f"{score:.4f}",
                "needs_review": "yes" if score < 0.72 else "no",
                "expected": prompt.text,
                "transcript": segment.transcript,
            }
        )

    manifest_path = work_dir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "wav",
                "source",
                "candidate",
                "start_sec",
                "end_sec",
                "duration_sec",
                "score",
                "needs_review",
                "expected",
                "transcript",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = work_dir / "summary.md"
    review_rows = [row for row in manifest_rows if row["needs_review"] == "yes"]
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# H4n Round2 {args.start_id:03d}-{args.end_id:03d} slicing summary\n\n")
        handle.write(f"- Source: `{input_path.relative_to(PROJECT_ROOT)}`\n")
        handle.write(f"- Duration: {duration:.3f}s\n")
        handle.write(f"- Sample rate: {sr} Hz\n")
        handle.write(f"- Channels read: {audio.shape[1]}\n")
        handle.write(f"- Mono peak: {db(peak):.2f} dBFS\n")
        handle.write(f"- Mono RMS: {db(rms):.2f} dBFS\n")
        handle.write(f"- VAD threshold: {db(threshold):.2f} dBFS\n")
        handle.write(f"- Candidate segments: {len(transcribed)}\n")
        handle.write(f"- Final segments: {len(manifest_rows)}\n")
        handle.write(f"- Review segments: {len(review_rows)}\n\n")
        handle.write("## Review\n\n")
        if not review_rows:
            handle.write("No low-score segments.\n")
        else:
            handle.write("| ID | Score | Candidate | Duration | Expected | Transcript |\n")
            handle.write("|---|---:|---|---:|---|---|\n")
            for row in review_rows:
                handle.write(
                    f"| {row['id']} | {row['score']} | {row['candidate']} | {row['duration_sec']} | "
                    f"{row['expected']} | {row['transcript']} |\n"
                )

    print(f"wrote\t{candidates_path}")
    print(f"wrote\t{manifest_path}")
    print(f"wrote\t{summary_path}")
    print(f"wrote\t{final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
