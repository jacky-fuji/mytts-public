from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf
from scipy.signal import lfilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = [
    "irodori_500m",
    "irodori_voicedesign",
    "qwen3_tts",
    "fish_speech",
]

MODEL_LABELS = {
    "original": "Original H4n",
    "irodori_500m": "Irodori 500M",
    "irodori_voicedesign": "Irodori VoiceDesign",
    "qwen3_tts": "Qwen3-TTS 1.7B",
    "fish_speech": "Fish Speech S2 Pro",
}


@dataclass
class Target:
    sample: str
    original_wav: Path
    text_file: Path
    category: str
    notes: str


@dataclass
class AudioFeatures:
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    mfcc: np.ndarray
    mfcc_vector: np.ndarray
    mel_db: np.ndarray
    f0: np.ndarray
    f0_times: np.ndarray
    f0_voiced_ratio: float
    f0_median_hz: float
    f0_std_hz: float
    formants_hz: tuple[float, float, float]
    spectral_vector: np.ndarray


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_manifest(path: Path) -> list[Target]:
    rows: list[Target] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                Target(
                    sample=row["sample"],
                    original_wav=resolve_path(row["original_wav"]),
                    text_file=resolve_path(row["text_file"]),
                    category=row["category"],
                    notes=row["notes"],
                )
            )
    return rows


def find_generated(sample: str, root: Path) -> dict[str, Path]:
    irodori = root / "irodori"
    qwen = root / "qwen3-tts"
    fish = root / "fish_speech"
    paths = {
        "irodori_500m": irodori / f"irodori_500m_v3_h4n_ref20s_steps24_{sample}.wav",
        "irodori_voicedesign": irodori / f"irodori_600m_v3_voicedesign_h4n_ref20s_steps24_{sample}.wav",
        "fish_speech": fish / f"fish_s2_h4n_ref10s_{sample}.wav",
    }
    matches = sorted(qwen.glob(f"*_{sample}.wav"))
    if matches:
        paths["qwen3_tts"] = matches[0]
    return paths


def load_audio(path: Path, sr: int) -> tuple[np.ndarray, int]:
    data, native_sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = np.mean(data, axis=1)
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
    y, _ = librosa.effects.trim(y, top_db=35)
    if y.size == 0:
        raise ValueError(f"Audio is empty after trimming: {path}")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.95
    return y.astype(np.float32), sr


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def median_or_nan(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def lpc_formants(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> tuple[float, float, float]:
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    if frames.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = np.percentile(rms, 60)
    selected = frames[rms >= threshold]
    if selected.shape[0] > 220:
        indices = np.linspace(0, selected.shape[0] - 1, 220).astype(int)
        selected = selected[indices]

    order = min(16, max(10, int(sr / 1000) + 2))
    collected: list[list[float]] = []
    window = np.hamming(frame_length)
    for frame in selected:
        frame = lfilter([1.0, -0.97], [1.0], frame * window)
        try:
            coeffs = librosa.lpc(frame, order=order)
        except FloatingPointError:
            continue
        roots = np.roots(coeffs)
        roots = roots[np.imag(roots) >= 0.01]
        if roots.size == 0:
            continue
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * sr / (2 * np.pi)
        bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.maximum(np.abs(roots), 1e-12))
        keep = (freqs > 90) & (freqs < 4000) & (bandwidths < 700)
        formants = sorted(freqs[keep])
        if len(formants) >= 3:
            collected.append(formants[:3])

    if not collected:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(collected, dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def extract_features(path: Path, sr: int = 22050) -> AudioFeatures:
    y, sr = load_audio(path, sr)
    duration_sec = float(len(y) / sr)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    hop_length = 256
    n_fft = 1024

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_vector = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta, axis=1),
        ]
    )

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=n_fft, hop_length=hop_length, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    voiced = f0[np.isfinite(f0)]
    f0_voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    f0_median_hz = float(np.median(voiced)) if voiced.size else float("nan")
    f0_std_hz = float(np.std(voiced)) if voiced.size else float("nan")

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
    spectral_vector = np.array(
        [
            np.mean(centroid),
            np.std(centroid),
            np.mean(bandwidth),
            np.std(bandwidth),
            np.mean(rolloff),
            np.std(rolloff),
            np.mean(zcr),
            np.std(zcr),
            f0_median_hz,
            f0_std_hz,
            f0_voiced_ratio,
        ],
        dtype=np.float64,
    )

    return AudioFeatures(
        duration_sec=duration_sec,
        peak_dbfs=dbfs(peak),
        rms_dbfs=dbfs(rms),
        mfcc=mfcc,
        mfcc_vector=mfcc_vector,
        mel_db=mel_db,
        f0=f0,
        f0_times=times,
        f0_voiced_ratio=f0_voiced_ratio,
        f0_median_hz=f0_median_hz,
        f0_std_hz=f0_std_hz,
        formants_hz=lpc_formants(y, sr, frame_length=1024, hop_length=hop_length),
        spectral_vector=spectral_vector,
    )


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    a = np.nan_to_num(a.astype(np.float64))
    b = np.nan_to_num(b.astype(np.float64))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def standardize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = np.concatenate([a, b], axis=1)
    mean = np.mean(both, axis=1, keepdims=True)
    std = np.std(both, axis=1, keepdims=True) + 1e-8
    return (a - mean) / std, (b - mean) / std


def dtw_average_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[1] > 900:
        step = int(np.ceil(a.shape[1] / 900))
        a = a[:, ::step]
    if b.shape[1] > 900:
        step = int(np.ceil(b.shape[1] / 900))
        b = b[:, ::step]
    a, b = standardize_pair(a, b)
    cost, path = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    return float(cost[-1, -1] / max(len(path), 1))


def exp_similarity(distance: float, scale: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return float(math.exp(-max(distance, 0.0) / scale))


def compare_features(original: AudioFeatures, generated: AudioFeatures) -> dict[str, float]:
    mfcc_cosine = cosine01(original.mfcc_vector, generated.mfcc_vector)
    spectral_cosine = cosine01(original.spectral_vector, generated.spectral_vector)
    mfcc_dtw = dtw_average_distance(original.mfcc, generated.mfcc)
    mel_dtw = dtw_average_distance(original.mel_db, generated.mel_db)

    f0_delta = abs(generated.f0_median_hz - original.f0_median_hz)
    f0_similarity = exp_similarity(f0_delta, 45.0)

    orig_formants = np.asarray(original.formants_hz, dtype=np.float64)
    gen_formants = np.asarray(generated.formants_hz, dtype=np.float64)
    formant_rel = np.nanmean(np.abs(gen_formants - orig_formants) / np.maximum(orig_formants, 1.0))
    formant_similarity = exp_similarity(formant_rel, 0.35)

    duration_ratio = generated.duration_sec / max(original.duration_sec, 1e-6)
    duration_similarity = exp_similarity(abs(math.log(max(duration_ratio, 1e-6))), 0.45)
    mfcc_dtw_similarity = exp_similarity(mfcc_dtw, 2.5)
    mel_dtw_similarity = exp_similarity(mel_dtw, 2.5)

    composite = (
        0.28 * mfcc_cosine
        + 0.22 * mfcc_dtw_similarity
        + 0.18 * mel_dtw_similarity
        + 0.12 * spectral_cosine
        + 0.10 * f0_similarity
        + 0.07 * formant_similarity
        + 0.03 * duration_similarity
    )

    return {
        "duration_ratio": duration_ratio,
        "duration_similarity": duration_similarity,
        "mfcc_cosine": mfcc_cosine,
        "mfcc_dtw_distance": mfcc_dtw,
        "mfcc_dtw_similarity": mfcc_dtw_similarity,
        "mel_dtw_distance": mel_dtw,
        "mel_dtw_similarity": mel_dtw_similarity,
        "spectral_cosine": spectral_cosine,
        "f0_median_delta_hz": generated.f0_median_hz - original.f0_median_hz,
        "f0_similarity": f0_similarity,
        "formant_relative_distance": float(formant_rel),
        "formant_similarity": formant_similarity,
        "composite_similarity": composite,
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrogram_grid(sample: str, features: dict[str, AudioFeatures], output: Path) -> None:
    keys = ["original"] + MODEL_ORDER
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 9), constrained_layout=True)
    vmin = -80
    vmax = 0
    for ax, key in zip(axes, keys):
        feat = features[key]
        librosa.display.specshow(
            feat.mel_db,
            sr=22050,
            hop_length=256,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(MODEL_LABELS[key], fontsize=10)
        ax.label_outer()
    fig.suptitle(f"Mel spectrogram comparison: {sample}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_f0(sample: str, features: dict[str, AudioFeatures], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    colors = {
        "original": "black",
        "irodori_500m": "#1f77b4",
        "irodori_voicedesign": "#ff7f0e",
        "qwen3_tts": "#2ca02c",
        "fish_speech": "#d62728",
    }
    for key in ["original"] + MODEL_ORDER:
        feat = features[key]
        f0 = feat.f0.astype(np.float64)
        f0[~np.isfinite(f0)] = np.nan
        ax.plot(feat.f0_times, f0, label=MODEL_LABELS[key], linewidth=1.5, color=colors[key], alpha=0.9)
    ax.set_title(f"F0 contour comparison: {sample}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(60, 260)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_model_summary(rows: list[dict[str, object]], output: Path) -> None:
    metrics = [
        ("composite_similarity", "Composite"),
        ("mfcc_cosine", "MFCC cosine"),
        ("mfcc_dtw_similarity", "MFCC DTW"),
        ("mel_dtw_similarity", "Mel DTW"),
        ("f0_similarity", "F0"),
        ("formant_similarity", "Formant"),
    ]
    means: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        means[model] = {
            metric: float(np.mean([float(row[metric]) for row in model_rows]))
            for metric, _label in metrics
        }

    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for index, model in enumerate(MODEL_ORDER):
        values = [means[model][metric] for metric, _label in metrics]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _metric, label in metrics], rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Similarity score (higher is better)")
    ax.set_title("Average acoustic similarity by model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def format_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_summary_md(path: Path, rows: list[dict[str, object]], targets: list[Target], figures_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figures_rel = Path(os.path.relpath(figures_dir, start=path.parent))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Acoustic similarity evaluation\n\n")
        handle.write("Date: 2026-06-16\n\n")
        handle.write("## Purpose\n\n")
        handle.write(
            "Compare generated speech against held-out H4n recordings of the exact same text. "
            "The reference voice used for generation is `h4n_ref20s_neutral_091_094.wav` for Qwen/Irodori and "
            "`h4n_ref10s_neutral_091_093.wav` for Fish Speech, so the target utterances below are not part of the reference prompt.\n\n"
        )
        handle.write("## Method\n\n")
        handle.write(
            "This is an acoustic-phonetic similarity analysis, not a biometric speaker-verification score. "
            "For each original/generated pair, the script extracts MFCCs, log-mel spectrograms, F0 with `librosa.pyin`, "
            "spectral statistics, and approximate LPC formants. Sequence-level similarity uses DTW over MFCC and mel features. "
            "The composite score is a weighted heuristic: MFCC cosine, MFCC-DTW, mel-DTW, spectral cosine, F0, formants, and duration.\n\n"
        )
        handle.write("## Targets\n\n")
        handle.write("| Sample | Category | Original | Text |\n")
        handle.write("|---|---|---|---|\n")
        for target in targets:
            text = target.text_file.read_text(encoding="utf-8").strip()
            handle.write(
                f"| {target.sample} | {target.category} | `{target.original_wav.relative_to(PROJECT_ROOT)}` | {text} |\n"
            )

        handle.write("\n## Average by model\n\n")
        handle.write("| Model | Composite | MFCC cosine | MFCC DTW sim | Mel DTW sim | F0 sim | Formant sim | Duration ratio |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for model in MODEL_ORDER:
            model_rows = [row for row in rows if row["model"] == model]
            handle.write(
                f"| {MODEL_LABELS[model]} | "
                f"{np.mean([float(row['composite_similarity']) for row in model_rows]):.4f} | "
                f"{np.mean([float(row['mfcc_cosine']) for row in model_rows]):.4f} | "
                f"{np.mean([float(row['mfcc_dtw_similarity']) for row in model_rows]):.4f} | "
                f"{np.mean([float(row['mel_dtw_similarity']) for row in model_rows]):.4f} | "
                f"{np.mean([float(row['f0_similarity']) for row in model_rows]):.4f} | "
                f"{np.mean([float(row['formant_similarity']) for row in model_rows]):.4f} | "
                f"{np.mean([float(row['duration_ratio']) for row in model_rows]):.3f} |\n"
            )

        handle.write("\n## Per-sample composite score\n\n")
        handle.write("| Sample | Irodori 500M | Irodori VoiceDesign | Qwen3-TTS | Fish Speech |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for target in targets:
            sample_rows = {row["model"]: row for row in rows if row["sample"] == target.sample}
            handle.write(
                f"| {target.sample} | "
                f"{float(sample_rows['irodori_500m']['composite_similarity']):.4f} | "
                f"{float(sample_rows['irodori_voicedesign']['composite_similarity']):.4f} | "
                f"{float(sample_rows['qwen3_tts']['composite_similarity']):.4f} | "
                f"{float(sample_rows['fish_speech']['composite_similarity']):.4f} |\n"
            )

        handle.write("\n## Figures\n\n")
        summary = figures_rel / "model_similarity_summary.png"
        handle.write(f"![Average model similarity]({summary.as_posix()})\n\n")
        for target in targets:
            spec = figures_rel / f"spectrogram_{target.sample}.png"
            f0 = figures_rel / f"f0_{target.sample}.png"
            handle.write(f"### {target.sample}\n\n")
            handle.write(f"![Mel spectrogram {target.sample}]({spec.as_posix()})\n\n")
            handle.write(f"![F0 contour {target.sample}]({f0.as_posix()})\n\n")

        handle.write("## Interpretation notes\n\n")
        handle.write(
            "- A higher composite score means the generated audio is closer to the held-out real recording under this acoustic feature set.\n"
            "- MFCC and mel-DTW are sensitive to articulation, spectral envelope, and pronunciation timing.\n"
            "- F0 similarity mainly captures pitch range and intonation, not timbre.\n"
            "- LPC formants are approximate because no Praat/parselmouth dependency is used; they are useful for relative comparison only.\n"
            "- The result should be read together with manual listening because voice likeness, naturalness, and content correctness are distinct axes.\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare generated TTS audio against same-text original recordings.")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "samples" / "manifests" / "similarity_eval_targets.tsv"))
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "outputs" / "similarity_eval"))
    parser.add_argument("--metrics-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "acoustic_similarity_metrics.tsv"))
    parser.add_argument("--summary-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "acoustic_similarity_eval.md"))
    parser.add_argument("--figures-dir", default=str(PROJECT_ROOT / "docs" / "assets" / "acoustic_similarity"))
    args = parser.parse_args()

    manifest = Path(args.manifest)
    generated_root = Path(args.generated_root)
    figures_dir = Path(args.figures_dir)
    targets = read_manifest(manifest)

    rows: list[dict[str, object]] = []
    for target in targets:
        generated_paths = find_generated(target.sample, generated_root)
        missing = [model for model in MODEL_ORDER if model not in generated_paths or not generated_paths[model].exists()]
        if missing:
            raise FileNotFoundError(f"Missing generated audio for {target.sample}: {missing}")

        feature_map: dict[str, AudioFeatures] = {"original": extract_features(target.original_wav)}
        for model in MODEL_ORDER:
            feature_map[model] = extract_features(generated_paths[model])

        original = feature_map["original"]
        for model in MODEL_ORDER:
            generated = feature_map[model]
            metrics = compare_features(original, generated)
            row: dict[str, object] = {
                "sample": target.sample,
                "category": target.category,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "original_wav": str(target.original_wav.relative_to(PROJECT_ROOT)),
                "generated_wav": str(generated_paths[model].relative_to(PROJECT_ROOT)),
                "original_duration_sec": format_float(original.duration_sec, 3),
                "generated_duration_sec": format_float(generated.duration_sec, 3),
                "original_f0_median_hz": format_float(original.f0_median_hz, 2),
                "generated_f0_median_hz": format_float(generated.f0_median_hz, 2),
                "original_f1_hz": format_float(original.formants_hz[0], 1),
                "generated_f1_hz": format_float(generated.formants_hz[0], 1),
                "original_f2_hz": format_float(original.formants_hz[1], 1),
                "generated_f2_hz": format_float(generated.formants_hz[1], 1),
                "original_f3_hz": format_float(original.formants_hz[2], 1),
                "generated_f3_hz": format_float(generated.formants_hz[2], 1),
            }
            row.update({name: format_float(value, 6) for name, value in metrics.items()})
            rows.append(row)

        plot_spectrogram_grid(target.sample, feature_map, figures_dir / f"spectrogram_{target.sample}.png")
        plot_f0(target.sample, feature_map, figures_dir / f"f0_{target.sample}.png")

    write_tsv(Path(args.metrics_path), rows)
    plot_model_summary(rows, figures_dir / "model_similarity_summary.png")
    write_summary_md(Path(args.summary_path), rows, targets, figures_dir)

    print(f"wrote\t{args.metrics_path}")
    print(f"wrote\t{args.summary_path}")
    print(f"wrote\t{args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
