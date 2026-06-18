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
SR = 22050
HOP_LENGTH = 256
N_FFT = 1024


@dataclass(frozen=True)
class Target:
    sample: str
    original_wav: Path
    text_file: Path
    category: str
    notes: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    color: str


@dataclass
class Features:
    path: Path
    y: np.ndarray
    sr: int
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_vector: np.ndarray
    mel_db: np.ndarray
    f0: np.ndarray
    f0_times: np.ndarray
    f0_voiced_ratio: float
    f0_median_hz: float
    f0_p10_hz: float
    f0_p90_hz: float
    f0_range_hz: float
    f0_std_hz: float
    f0_slope_hz_per_sec: float
    formants_hz: tuple[float, float, float]
    spectral_vector: np.ndarray
    speech_segments: list[tuple[int, int]]


MODEL_SPECS = [
    ModelSpec("irodori_500m", "Irodori-TTS 500M", "#1f77b4"),
    ModelSpec("irodori_voicedesign", "Irodori-TTS 600M VoiceDesign", "#ff7f0e"),
    ModelSpec("qwen3_tts", "Qwen3-TTS 1.7B", "#2ca02c"),
    ModelSpec("fish_speech", "Fish Speech S2 Pro", "#d62728"),
    ModelSpec("cosyvoice2", "CosyVoice2", "#9467bd"),
    ModelSpec("voxcpm2_ultimate", "VoxCPM2 ultimate", "#8c564b"),
]

MODEL_BY_KEY = {model.key: model for model in MODEL_SPECS}


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def read_manifest(path: Path) -> list[Target]:
    targets: list[Target] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            targets.append(
                Target(
                    sample=row["sample"],
                    original_wav=resolve_path(row["original_wav"]),
                    text_file=resolve_path(row["text_file"]),
                    category=row["category"],
                    notes=row["notes"],
                )
            )
    return targets


def generated_paths(sample: str, root: Path) -> dict[str, Path]:
    qwen_matches = sorted((root / "qwen3-tts").glob(f"*_{sample}.wav"))
    paths = {
        "irodori_500m": root / "irodori" / f"irodori_500m_v3_h4n_ref20s_steps24_{sample}.wav",
        "irodori_voicedesign": root / "irodori" / f"irodori_600m_v3_voicedesign_h4n_ref20s_steps24_{sample}.wav",
        "fish_speech": root / "fish_speech" / f"fish_s2_h4n_ref10s_{sample}.wav",
        "cosyvoice2": root / "cosyvoice2" / f"cosyvoice2_h4n_ref20s_similarity_same_text_s1p00_{sample}.wav",
        "voxcpm2_ultimate": root
        / "voxcpm"
        / f"voxcpm2_h4n_ref20s_neutral_091_094_ultimate_cfg2.0_ts10_{sample}.wav",
    }
    if qwen_matches:
        paths["qwen3_tts"] = qwen_matches[0]
    return paths


def load_audio(path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    data, native_sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = np.mean(data, axis=1)
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
    y, _ = librosa.effects.trim(y, top_db=35)
    if y.size == 0:
        raise ValueError(f"audio is empty after trimming: {path}")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.95
    return y.astype(np.float32), sr


def lpc_formants(y: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = HOP_LENGTH) -> tuple[float, float, float]:
    if y.size < frame_length:
        return (float("nan"), float("nan"), float("nan"))
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = np.percentile(rms, 60)
    selected = frames[rms >= threshold]
    if selected.shape[0] > 240:
        selected = selected[np.linspace(0, selected.shape[0] - 1, 240).astype(int)]

    order = min(16, max(10, int(sr / 1000) + 2))
    window = np.hamming(frame_length)
    formants: list[list[float]] = []
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
        values = sorted(freqs[keep])
        if len(values) >= 3:
            formants.append(values[:3])
    if not formants:
        return (float("nan"), float("nan"), float("nan"))
    med = np.nanmedian(np.asarray(formants, dtype=np.float64), axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def f0_slope(times: np.ndarray, f0: np.ndarray) -> float:
    mask = np.isfinite(f0)
    if np.sum(mask) < 4:
        return float("nan")
    x = times[mask]
    y = f0[mask]
    if x[-1] - x[0] < 0.2:
        return float("nan")
    slope, _intercept = np.polyfit(x - x[0], y, 1)
    return float(slope)


def merge_segments(segments: list[tuple[int, int]], sr: int, min_gap_sec: float = 0.16) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged: list[tuple[int, int]] = []
    gap = int(min_gap_sec * sr)
    start, end = segments[0]
    for next_start, next_end in segments[1:]:
        if next_start - end <= gap:
            end = next_end
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def detect_speech_segments(y: np.ndarray, sr: int) -> list[tuple[int, int]]:
    intervals = librosa.effects.split(y, top_db=28, frame_length=1024, hop_length=HOP_LENGTH)
    segments = [(int(start), int(end)) for start, end in intervals if (end - start) / sr >= 0.18]
    return merge_segments(segments, sr)


def extract_features(path: Path) -> Features:
    y, sr = load_audio(path)
    duration_sec = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_vector = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta, axis=1),
        ]
    )
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    voiced = f0[np.isfinite(f0)]
    if voiced.size:
        f0_median = float(np.median(voiced))
        f0_p10 = float(np.percentile(voiced, 10))
        f0_p90 = float(np.percentile(voiced, 90))
        f0_range = f0_p90 - f0_p10
        f0_std = float(np.std(voiced))
    else:
        f0_median = f0_p10 = f0_p90 = f0_range = f0_std = float("nan")

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
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
            f0_median,
            f0_range,
            f0_std,
            float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0,
        ],
        dtype=np.float64,
    )

    return Features(
        path=path,
        y=y,
        sr=sr,
        duration_sec=float(duration_sec),
        peak_dbfs=dbfs(peak),
        rms_dbfs=dbfs(rms),
        mfcc=mfcc,
        mfcc_delta=mfcc_delta,
        mfcc_vector=mfcc_vector,
        mel_db=mel_db,
        f0=f0,
        f0_times=times,
        f0_voiced_ratio=float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0,
        f0_median_hz=f0_median,
        f0_p10_hz=f0_p10,
        f0_p90_hz=f0_p90,
        f0_range_hz=f0_range,
        f0_std_hz=f0_std,
        f0_slope_hz_per_sec=f0_slope(times, f0),
        formants_hz=lpc_formants(y, sr),
        spectral_vector=spectral_vector,
        speech_segments=detect_speech_segments(y, sr),
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


def dtw_path_and_local_distances(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    a_std, b_std = standardize_pair(a, b)
    if a_std.shape[1] > 1000:
        a_std = a_std[:, :: int(np.ceil(a_std.shape[1] / 1000))]
    if b_std.shape[1] > 1000:
        b_std = b_std[:, :: int(np.ceil(b_std.shape[1] / 1000))]
    cost, path = librosa.sequence.dtw(X=a_std, Y=b_std, metric="euclidean")
    path = np.asarray(path[::-1], dtype=int)
    local = np.linalg.norm(a_std[:, path[:, 0]].T - b_std[:, path[:, 1]].T, axis=1)
    avg = float(cost[-1, -1] / max(len(path), 1))
    return avg, path[:, 0], path[:, 1], local


def exp_similarity(distance: float, scale: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return float(math.exp(-max(distance, 0.0) / scale))


def paired_f0_metrics(original: Features, generated: Features, orig_idx: np.ndarray, gen_idx: np.ndarray) -> dict[str, float]:
    max_orig = len(original.f0) - 1
    max_gen = len(generated.f0) - 1
    oi = np.clip(orig_idx, 0, max_orig)
    gi = np.clip(gen_idx, 0, max_gen)
    orig_f0 = original.f0[oi]
    gen_f0 = generated.f0[gi]
    mask = np.isfinite(orig_f0) & np.isfinite(gen_f0)
    if np.sum(mask) < 4:
        return {
            "f0_mae_hz": float("nan"),
            "f0_corr": float("nan"),
            "f0_contour_similarity": 0.0,
            "f0_median_delta_hz": generated.f0_median_hz - original.f0_median_hz,
            "f0_range_delta_hz": generated.f0_range_hz - original.f0_range_hz,
            "f0_slope_delta": generated.f0_slope_hz_per_sec - original.f0_slope_hz_per_sec,
        }
    o = orig_f0[mask]
    g = gen_f0[mask]
    mae = float(np.mean(np.abs(g - o)))
    corr = float(np.corrcoef(o, g)[0, 1]) if np.std(o) > 1e-6 and np.std(g) > 1e-6 else 0.0
    corr01 = (corr + 1.0) / 2.0
    mae_sim = exp_similarity(mae, 45.0)
    range_sim = exp_similarity(abs((generated.f0_range_hz or 0.0) - (original.f0_range_hz or 0.0)), 55.0)
    contour = 0.55 * mae_sim + 0.30 * corr01 + 0.15 * range_sim
    return {
        "f0_mae_hz": mae,
        "f0_corr": corr,
        "f0_contour_similarity": contour,
        "f0_median_delta_hz": generated.f0_median_hz - original.f0_median_hz,
        "f0_range_delta_hz": generated.f0_range_hz - original.f0_range_hz,
        "f0_slope_delta": generated.f0_slope_hz_per_sec - original.f0_slope_hz_per_sec,
    }


def compare_features(original: Features, generated: Features) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    mfcc_dtw, orig_idx, gen_idx, local_mfcc = dtw_path_and_local_distances(original.mfcc, generated.mfcc)
    mel_dtw, _mel_orig_idx, _mel_gen_idx, local_mel = dtw_path_and_local_distances(original.mel_db, generated.mel_db)
    f0_metrics = paired_f0_metrics(original, generated, orig_idx, gen_idx)

    orig_formants = np.asarray(original.formants_hz, dtype=np.float64)
    gen_formants = np.asarray(generated.formants_hz, dtype=np.float64)
    formant_rel = float(np.nanmean(np.abs(gen_formants - orig_formants) / np.maximum(orig_formants, 1.0)))

    duration_ratio = generated.duration_sec / max(original.duration_sec, 1e-6)
    duration_similarity = exp_similarity(abs(math.log(max(duration_ratio, 1e-6))), 0.45)
    mfcc_cosine = cosine01(original.mfcc_vector, generated.mfcc_vector)
    spectral_cosine = cosine01(original.spectral_vector, generated.spectral_vector)
    mfcc_dtw_similarity = exp_similarity(mfcc_dtw, 2.5)
    mel_dtw_similarity = exp_similarity(mel_dtw, 2.5)
    formant_similarity = exp_similarity(formant_rel, 0.35)

    composite = (
        0.24 * mfcc_cosine
        + 0.18 * mfcc_dtw_similarity
        + 0.16 * mel_dtw_similarity
        + 0.12 * spectral_cosine
        + 0.20 * f0_metrics["f0_contour_similarity"]
        + 0.07 * formant_similarity
        + 0.03 * duration_similarity
    )

    metrics = {
        "composite_similarity": composite,
        "timbre_similarity": 0.50 * mfcc_cosine + 0.30 * spectral_cosine + 0.20 * formant_similarity,
        "sequence_similarity": 0.55 * mfcc_dtw_similarity + 0.45 * mel_dtw_similarity,
        "prosody_similarity": f0_metrics["f0_contour_similarity"],
        "duration_ratio": duration_ratio,
        "duration_similarity": duration_similarity,
        "mfcc_cosine": mfcc_cosine,
        "mfcc_dtw_distance": mfcc_dtw,
        "mfcc_dtw_similarity": mfcc_dtw_similarity,
        "mel_dtw_distance": mel_dtw,
        "mel_dtw_similarity": mel_dtw_similarity,
        "spectral_cosine": spectral_cosine,
        "formant_relative_distance": formant_rel,
        "formant_similarity": formant_similarity,
        **f0_metrics,
    }
    alignment = {
        "orig_idx": orig_idx,
        "gen_idx": gen_idx,
        "local_mfcc": local_mfcc,
        "local_mel": local_mel,
    }
    return metrics, alignment


def segment_feature(y: np.ndarray, sr: int) -> dict[str, object]:
    if y.size < 1024:
        return {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    f0, _voiced, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    voiced = f0[np.isfinite(f0)]
    f0_median = float(np.median(voiced)) if voiced.size else float("nan")
    f0_range = float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if voiced.size else float("nan")
    return {
        "duration_sec": len(y) / sr,
        "mfcc_vector": np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]),
        "f0_median_hz": f0_median,
        "f0_range_hz": f0_range,
        "formants_hz": lpc_formants(y, sr),
    }


def compare_segment_features(original: dict[str, object], generated: dict[str, object]) -> dict[str, float]:
    if not original or not generated:
        return {
            "segment_similarity": 0.0,
            "segment_mfcc_cosine": 0.0,
            "segment_duration_ratio": float("nan"),
            "segment_f0_delta_hz": float("nan"),
            "segment_f0_range_delta_hz": float("nan"),
            "segment_formant_similarity": 0.0,
        }
    mfcc_cos = cosine01(original["mfcc_vector"], generated["mfcc_vector"])  # type: ignore[arg-type]
    duration_ratio = float(generated["duration_sec"]) / max(float(original["duration_sec"]), 1e-6)
    duration_sim = exp_similarity(abs(math.log(max(duration_ratio, 1e-6))), 0.45)
    f0_delta = float(generated["f0_median_hz"]) - float(original["f0_median_hz"])
    f0_range_delta = float(generated["f0_range_hz"]) - float(original["f0_range_hz"])
    f0_sim = exp_similarity(abs(f0_delta), 45.0)
    orig_formants = np.asarray(original["formants_hz"], dtype=np.float64)
    gen_formants = np.asarray(generated["formants_hz"], dtype=np.float64)
    formant_rel = float(np.nanmean(np.abs(gen_formants - orig_formants) / np.maximum(orig_formants, 1.0)))
    formant_sim = exp_similarity(formant_rel, 0.35)
    segment_similarity = 0.42 * mfcc_cos + 0.26 * f0_sim + 0.22 * formant_sim + 0.10 * duration_sim
    return {
        "segment_similarity": segment_similarity,
        "segment_mfcc_cosine": mfcc_cos,
        "segment_duration_ratio": duration_ratio,
        "segment_f0_delta_hz": f0_delta,
        "segment_f0_range_delta_hz": f0_range_delta,
        "segment_formant_similarity": formant_sim,
    }


def segment_rows(sample: str, model_key: str, original: Features, generated: Features) -> list[dict[str, object]]:
    count = min(len(original.speech_segments), len(generated.speech_segments))
    rows: list[dict[str, object]] = []
    for index in range(count):
        os, oe = original.speech_segments[index]
        gs, ge = generated.speech_segments[index]
        original_feature = segment_feature(original.y[os:oe], original.sr)
        generated_feature = segment_feature(generated.y[gs:ge], generated.sr)
        metrics = compare_segment_features(original_feature, generated_feature)
        rows.append(
            {
                "sample": sample,
                "model": model_key,
                "model_label": MODEL_BY_KEY[model_key].label,
                "segment": index + 1,
                "original_start_sec": os / original.sr,
                "original_end_sec": oe / original.sr,
                "generated_start_sec": gs / generated.sr,
                "generated_end_sec": ge / generated.sr,
                **metrics,
            }
        )
    return rows


def summarize_worst_regions(
    sample: str,
    model_key: str,
    original: Features,
    generated: Features,
    alignment: dict[str, np.ndarray],
    window_sec: float = 0.45,
) -> list[dict[str, object]]:
    orig_idx = alignment["orig_idx"]
    gen_idx = alignment["gen_idx"]
    local_mfcc = alignment["local_mfcc"]
    times = librosa.frames_to_time(orig_idx, sr=original.sr, hop_length=HOP_LENGTH)
    if times.size == 0:
        return []
    window = max(1, int(window_sec / (HOP_LENGTH / original.sr)))
    scores = np.convolve(local_mfcc, np.ones(window) / window, mode="same")
    candidates = np.argsort(scores)[::-1]
    regions: list[dict[str, object]] = []
    used: list[tuple[float, float]] = []
    for idx in candidates:
        start = max(0.0, times[idx] - window_sec / 2)
        end = min(original.duration_sec, times[idx] + window_sec / 2)
        if any(not (end < us or start > ue) for us, ue in used):
            continue
        gi = np.clip(gen_idx[idx], 0, len(generated.f0) - 1)
        oi = np.clip(orig_idx[idx], 0, len(original.f0) - 1)
        orig_f0 = original.f0[oi] if oi < len(original.f0) else np.nan
        gen_f0 = generated.f0[gi] if gi < len(generated.f0) else np.nan
        f0_delta = float(gen_f0 - orig_f0) if np.isfinite(orig_f0) and np.isfinite(gen_f0) else float("nan")
        regions.append(
            {
                "sample": sample,
                "model": model_key,
                "model_label": MODEL_BY_KEY[model_key].label,
                "start_sec": start,
                "end_sec": end,
                "mean_local_mfcc_distance": float(scores[idx]),
                "approx_f0_delta_hz": f0_delta,
            }
        )
        used.append((start, end))
        if len(regions) >= 2:
            break
    return regions


def read_asr_summary(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    buckets: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            wav = row["wav"]
            key = ""
            if "irodori_500m" in wav:
                key = "irodori_500m"
            elif "irodori_600m" in wav:
                key = "irodori_voicedesign"
            elif "qwen3" in wav:
                key = "qwen3_tts"
            elif "fish_s2" in wav:
                key = "fish_speech"
            elif "cosyvoice2" in wav:
                key = "cosyvoice2"
            elif "voxcpm2" in wav and "ultimate" in wav:
                key = "voxcpm2_ultimate"
            else:
                continue
            buckets.setdefault(key, []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for key, rows in buckets.items():
        summary[key] = {
            "count": float(len(rows)),
            "review": float(sum(1 for row in rows if row.get("needs_review") == "yes")),
            "avg_ratio": float(np.mean([float(row["ratio"]) for row in rows])),
            "avg_duration": float(np.mean([float(row["duration_sec"]) for row in rows])),
        }
    return summary


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = "nan" if not np.isfinite(value) else f"{value:.6f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def model_average(rows: list[dict[str, object]], metric: str, model_key: str) -> float:
    values = [float(row[metric]) for row in rows if row["model"] == model_key]
    return float(np.mean(values)) if values else float("nan")


def plot_model_overview(rows: list[dict[str, object]], output: Path) -> None:
    metrics = [
        ("composite_similarity", "Overall"),
        ("timbre_similarity", "Timbre"),
        ("sequence_similarity", "Frame/DTW"),
        ("prosody_similarity", "Prosody"),
        ("formant_similarity", "Formants"),
    ]
    x = np.arange(len(metrics))
    width = 0.13
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for index, model in enumerate(MODEL_SPECS):
        values = [model_average(rows, metric, model.key) for metric, _label in metrics]
        ax.bar(x + (index - (len(MODEL_SPECS) - 1) / 2) * width, values, width=width, color=model.color, label=model.label)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _metric, label in metrics])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Similarity score (higher is better)")
    ax.set_title("Same-text acoustic similarity against held-out H4n recordings")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_spectrogram_grid(sample: str, feature_map: dict[str, Features], output: Path) -> None:
    keys = ["original"] + [model.key for model in MODEL_SPECS]
    labels = {"original": "Original H4n"} | {model.key: model.label for model in MODEL_SPECS}
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 12.5), constrained_layout=True)
    for ax, key in zip(axes, keys):
        feat = feature_map[key]
        librosa.display.specshow(
            feat.mel_db,
            sr=SR,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            vmin=-80,
            vmax=0,
            ax=ax,
        )
        ax.set_title(labels[key], fontsize=10)
        ax.label_outer()
    fig.suptitle(f"Mel spectrogram grid: {sample}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_f0_overlay(sample: str, feature_map: dict[str, Features], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(feature_map["original"].f0_times, feature_map["original"].f0, color="black", linewidth=2.0, label="Original H4n")
    for model in MODEL_SPECS:
        feat = feature_map[model.key]
        ax.plot(feat.f0_times, feat.f0, color=model.color, linewidth=1.35, alpha=0.9, label=model.label)
    ax.set_title(f"F0 contour overlay: {sample}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(60, 270)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_segment_heatmap(sample: str, segment_metrics: list[dict[str, object]], output: Path) -> None:
    sample_rows = [row for row in segment_metrics if row["sample"] == sample]
    if not sample_rows:
        return
    max_seg = max(int(row["segment"]) for row in sample_rows)
    matrix = np.full((len(MODEL_SPECS), max_seg), np.nan)
    for model_index, model in enumerate(MODEL_SPECS):
        for row in sample_rows:
            if row["model"] == model.key:
                matrix[model_index, int(row["segment"]) - 1] = float(row["segment_similarity"])
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.35, vmax=1.0, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(MODEL_SPECS)))
    ax.set_yticklabels([model.label for model in MODEL_SPECS], fontsize=8)
    ax.set_xticks(np.arange(max_seg))
    ax.set_xticklabels([f"seg{i + 1}" for i in range(max_seg)])
    ax.set_title(f"Speech-segment similarity: {sample}")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, label="Segment similarity")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_f0_delta_heatmap(sample: str, rows: list[dict[str, object]], output: Path) -> None:
    sample_rows = [row for row in rows if row["sample"] == sample]
    if not sample_rows:
        return
    values = np.asarray([[float(row["f0_median_delta_hz"]), float(row["f0_range_delta_hz"]), float(row["f0_slope_delta"])] for row in sample_rows])
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    image = ax.imshow(values, cmap="coolwarm", aspect="auto", vmin=-90, vmax=90)
    ax.set_yticks(np.arange(len(sample_rows)))
    ax.set_yticklabels([MODEL_BY_KEY[str(row["model"])].label for row in sample_rows], fontsize=8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Median F0 delta", "F0 range delta", "F0 slope delta"])
    ax.set_title(f"Pitch deviation from original: {sample}")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(image, ax=ax, label="Hz / Hz per sec")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def relative_link(from_path: Path, to_path: Path) -> str:
    return Path(os.path.relpath(to_path, start=from_path.parent)).as_posix()


def write_report(
    path: Path,
    targets: list[Target],
    metric_rows: list[dict[str, object]],
    segment_metrics: list[dict[str, object]],
    worst_regions: list[dict[str, object]],
    asr_summary: dict[str, dict[str, float]],
    figures_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = lambda name: relative_link(path, figures_dir / name)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Same-text deep acoustic similarity report\n\n")
        handle.write("Date: 2026-06-16\n\n")
        handle.write("## Scope\n\n")
        handle.write(
            "Held-out Zoom H4n Pro recordings were used as the target voice. Each TTS model generated the exact same text as the original recording, "
            "using H4n reference clips from different utterances (`091-094`, or `091-093` for Fish/GPT/F5-style 10s constraints). "
            "This report excludes F5-TTS and GPT-SoVITS from the main comparison based on the latest qualitative review.\n\n"
        )
        handle.write("Models compared: Irodori 500M, Irodori 600M VoiceDesign, Qwen3-TTS 1.7B, Fish Speech S2 Pro, CosyVoice2, and VoxCPM2 ultimate.\n\n")

        handle.write("## Method\n\n")
        handle.write(
            "The analysis is acoustic-phonetic rather than biometric speaker verification. For each original/generated pair, the script extracts MFCCs, "
            "log-mel spectrograms, F0 contours with `librosa.pyin`, spectral centroid/bandwidth/rolloff, and approximate LPC formants F1/F2/F3. "
            "MFCC and mel sequences are aligned with dynamic time warping. Speech regions are also split by energy-based pauses and compared segment by segment.\n\n"
        )
        handle.write("Composite score = timbre + frame/DTW similarity + prosody + formants + duration. Higher is better, but manual listening remains authoritative.\n\n")

        handle.write("## Inputs\n\n")
        handle.write("| Sample | Category | Text |\n")
        handle.write("|---|---|---|\n")
        for target in targets:
            text = target.text_file.read_text(encoding="utf-8").strip()
            handle.write(f"| `{target.sample}` | {target.category} | {text} |\n")

        handle.write("\n## Overall acoustic similarity\n\n")
        handle.write("| Model | Composite | Timbre | Frame/DTW | Prosody | Formants | Duration ratio |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for model in MODEL_SPECS:
            handle.write(
                f"| {model.label} | "
                f"{model_average(metric_rows, 'composite_similarity', model.key):.4f} | "
                f"{model_average(metric_rows, 'timbre_similarity', model.key):.4f} | "
                f"{model_average(metric_rows, 'sequence_similarity', model.key):.4f} | "
                f"{model_average(metric_rows, 'prosody_similarity', model.key):.4f} | "
                f"{model_average(metric_rows, 'formant_similarity', model.key):.4f} | "
                f"{model_average(metric_rows, 'duration_ratio', model.key):.3f} |\n"
            )
        handle.write(f"\n![Overall model similarity]({fig('model_overview.png')})\n\n")

        if asr_summary:
            handle.write("## ASR content preservation\n\n")
            handle.write("| Model | Count | Review | Avg Ratio | Avg Duration |\n")
            handle.write("|---|---:|---:|---:|---:|\n")
            for model in MODEL_SPECS:
                item = asr_summary.get(model.key)
                if not item:
                    continue
                handle.write(
                    f"| {model.label} | {int(item['count'])} | {int(item['review'])} | "
                    f"{item['avg_ratio']:.4f} | {item['avg_duration']:.2f}s |\n"
                )
            handle.write("\nASR checks content preservation only. It is not a voice-likeness score.\n\n")

        handle.write("## Per-sample composite score\n\n")
        handle.write("| Sample | " + " | ".join(model.label for model in MODEL_SPECS) + " |\n")
        handle.write("|---" + "|---:" * len(MODEL_SPECS) + "|\n")
        for target in targets:
            values = []
            for model in MODEL_SPECS:
                value = next(
                    float(row["composite_similarity"])
                    for row in metric_rows
                    if row["sample"] == target.sample and row["model"] == model.key
                )
                values.append(f"{value:.4f}")
            handle.write(f"| `{target.sample}` | " + " | ".join(values) + " |\n")

        handle.write("\n## Pitch and formant diagnostics\n\n")
        handle.write("| Model | Median F0 delta | F0 contour sim | F0 corr | Formant relative distance |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for model in MODEL_SPECS:
            handle.write(
                f"| {model.label} | "
                f"{model_average(metric_rows, 'f0_median_delta_hz', model.key):.2f} Hz | "
                f"{model_average(metric_rows, 'prosody_similarity', model.key):.4f} | "
                f"{model_average(metric_rows, 'f0_corr', model.key):.4f} | "
                f"{model_average(metric_rows, 'formant_relative_distance', model.key):.4f} |\n"
            )

        handle.write("\n## Sample-level figures\n\n")
        for target in targets:
            handle.write(f"### {target.sample}\n\n")
            handle.write(f"![Spectrogram grid {target.sample}]({fig(f'spectrogram_grid_{target.sample}.png')})\n\n")
            handle.write(f"![F0 overlay {target.sample}]({fig(f'f0_overlay_{target.sample}.png')})\n\n")
            handle.write(f"![Segment heatmap {target.sample}]({fig(f'segment_heatmap_{target.sample}.png')})\n\n")
            handle.write(f"![F0 delta heatmap {target.sample}]({fig(f'f0_delta_{target.sample}.png')})\n\n")

        handle.write("## Segment-level observations\n\n")
        handle.write("| Sample | Model | Weakest segment | Segment score | F0 delta | Duration ratio |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for target in targets:
            for model in MODEL_SPECS:
                rows = [row for row in segment_metrics if row["sample"] == target.sample and row["model"] == model.key]
                if not rows:
                    continue
                weakest = min(rows, key=lambda row: float(row["segment_similarity"]))
                handle.write(
                    f"| `{target.sample}` | {model.label} | {int(weakest['segment'])} | "
                    f"{float(weakest['segment_similarity']):.4f} | "
                    f"{float(weakest['segment_f0_delta_hz']):.1f} Hz | "
                    f"{float(weakest['segment_duration_ratio']):.3f} |\n"
                )

        handle.write("\n## DTW worst regions\n\n")
        handle.write("| Sample | Model | Original time | Local MFCC distance | Approx F0 delta |\n")
        handle.write("|---|---|---|---:|---:|\n")
        for row in worst_regions:
            handle.write(
                f"| `{row['sample']}` | {row['model_label']} | "
                f"{float(row['start_sec']):.2f}-{float(row['end_sec']):.2f}s | "
                f"{float(row['mean_local_mfcc_distance']):.4f} | "
                f"{float(row['approx_f0_delta_hz']):.1f} Hz |\n"
            )

        handle.write("\n## Interpretation\n\n")
        handle.write(
            "- Fish Speech remains the strongest acoustic match on average, especially in prosody and frame-level similarity, but it is slow.\n"
            "- CosyVoice2 is no longer a clear reject after the H4n recording. Its content preservation and timbre metrics are competitive, but prompt-length warnings and subjective voice likeness still need listening review.\n"
            "- VoxCPM2 ultimate preserves text well, but segment and F0 diagnostics are important because the perceived issue is prosody rather than content.\n"
            "- Irodori VoiceDesign generally improves over Irodori 500M on prosody, matching the qualitative sense that caption control helps but does not fully close the gap to Fish Speech.\n"
            "- Qwen3-TTS can match individual samples well, but pitch movement remains less predictable, especially in local F0 jumps.\n"
        )

        handle.write("\n## Artifacts\n\n")
        handle.write("- `docs/comparisons/acoustic_similarity_deep_metrics.tsv`\n")
        handle.write("- `docs/comparisons/acoustic_similarity_segment_metrics.tsv`\n")
        handle.write("- `docs/comparisons/acoustic_similarity_worst_regions.tsv`\n")
        handle.write("- `docs/assets/acoustic_similarity_deep/`\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed same-text acoustic similarity analysis.")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "samples" / "manifests" / "similarity_eval_targets.tsv"))
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "outputs" / "similarity_eval"))
    parser.add_argument("--asr-tsv", default=str(PROJECT_ROOT / "outputs" / "transcripts" / "similarity_eval_all_models_asr.tsv"))
    parser.add_argument("--metrics-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "acoustic_similarity_deep_metrics.tsv"))
    parser.add_argument("--segment-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "acoustic_similarity_segment_metrics.tsv"))
    parser.add_argument("--worst-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "acoustic_similarity_worst_regions.tsv"))
    parser.add_argument("--report-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "acoustic_similarity_deep_report.md"))
    parser.add_argument("--figures-dir", default=str(PROJECT_ROOT / "docs" / "assets" / "acoustic_similarity_deep"))
    args = parser.parse_args()

    targets = read_manifest(Path(args.manifest))
    generated_root = Path(args.generated_root)
    figures_dir = Path(args.figures_dir)
    metric_rows: list[dict[str, object]] = []
    segment_metrics: list[dict[str, object]] = []
    worst_regions: list[dict[str, object]] = []

    for target in targets:
        paths = generated_paths(target.sample, generated_root)
        missing = [model.key for model in MODEL_SPECS if model.key not in paths or not paths[model.key].exists()]
        if missing:
            raise FileNotFoundError(f"missing generated audio for {target.sample}: {missing}")

        feature_map: dict[str, Features] = {"original": extract_features(target.original_wav)}
        for model in MODEL_SPECS:
            feature_map[model.key] = extract_features(paths[model.key])

        original = feature_map["original"]
        for model in MODEL_SPECS:
            generated = feature_map[model.key]
            metrics, alignment = compare_features(original, generated)
            row = {
                "sample": target.sample,
                "category": target.category,
                "model": model.key,
                "model_label": model.label,
                "original_wav": str(target.original_wav.relative_to(PROJECT_ROOT)),
                "generated_wav": str(paths[model.key].relative_to(PROJECT_ROOT)),
                "original_duration_sec": original.duration_sec,
                "generated_duration_sec": generated.duration_sec,
                "original_f0_median_hz": original.f0_median_hz,
                "generated_f0_median_hz": generated.f0_median_hz,
                "original_f0_range_hz": original.f0_range_hz,
                "generated_f0_range_hz": generated.f0_range_hz,
                "original_f1_hz": original.formants_hz[0],
                "generated_f1_hz": generated.formants_hz[0],
                "original_f2_hz": original.formants_hz[1],
                "generated_f2_hz": generated.formants_hz[1],
                "original_f3_hz": original.formants_hz[2],
                "generated_f3_hz": generated.formants_hz[2],
                **metrics,
            }
            metric_rows.append(row)
            segment_metrics.extend(segment_rows(target.sample, model.key, original, generated))
            worst_regions.extend(summarize_worst_regions(target.sample, model.key, original, generated, alignment))

        plot_spectrogram_grid(target.sample, feature_map, figures_dir / f"spectrogram_grid_{target.sample}.png")
        plot_f0_overlay(target.sample, feature_map, figures_dir / f"f0_overlay_{target.sample}.png")
        plot_segment_heatmap(target.sample, segment_metrics, figures_dir / f"segment_heatmap_{target.sample}.png")
        plot_f0_delta_heatmap(target.sample, metric_rows, figures_dir / f"f0_delta_{target.sample}.png")

    write_tsv(Path(args.metrics_path), metric_rows)
    write_tsv(Path(args.segment_path), segment_metrics)
    write_tsv(Path(args.worst_path), worst_regions)
    plot_model_overview(metric_rows, figures_dir / "model_overview.png")
    write_report(
        Path(args.report_path),
        targets,
        metric_rows,
        segment_metrics,
        worst_regions,
        read_asr_summary(Path(args.asr_tsv)),
        figures_dir,
    )

    print(f"wrote\t{args.metrics_path}")
    print(f"wrote\t{args.segment_path}")
    print(f"wrote\t{args.worst_path}")
    print(f"wrote\t{args.report_path}")
    print(f"wrote\t{args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
