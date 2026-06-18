from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import jaconv
import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from analyze_acoustic_similarity_deep import (  # noqa: E402
    HOP_LENGTH,
    MODEL_BY_KEY,
    MODEL_SPECS,
    N_FFT,
    PROJECT_ROOT,
    SR,
    Target,
    cosine01,
    dtw_path_and_local_distances,
    exp_similarity,
    generated_paths,
    load_audio,
    lpc_formants,
    read_manifest,
    write_tsv,
)


FIGURES_DIR = PROJECT_ROOT / "docs" / "assets" / "acoustic_phonetic_micro"

READINGS = {
    "similarity_h4n_005": "はい、ではつぎに、すうじとこゆうめいしをふくむぶんしょうをもういちどかくにんしましょう。",
    "similarity_h4n_026": "あーるてぃーえっくす ごーまるろくまる てぃーあい、ぶいらむ じゅうろくぎがばいと、くーだ、ぱいそん、ぱいとーち をかくにんします。",
    "similarity_h4n_051": "これはせつめいどうがのぼうとうでつかう、おちついたよみあげのぶんしょうです。",
    "similarity_h4n_087": "いきもののこえと、なまびーるのちゅうもんでは、なまのよみかたがかわります。",
}


VOWELS = set("あいうえお")
SMALL_KANA = set("ぁぃぅぇぉゃゅょゎ")
PUNCTUATION = {"、": "minor_pause", "。": "final_pause"}

BASE_VOWEL = {
    "あ": "a",
    "か": "a",
    "さ": "a",
    "た": "a",
    "な": "a",
    "は": "a",
    "ま": "a",
    "や": "a",
    "ら": "a",
    "わ": "a",
    "が": "a",
    "ざ": "a",
    "だ": "a",
    "ば": "a",
    "ぱ": "a",
    "い": "i",
    "き": "i",
    "し": "i",
    "ち": "i",
    "に": "i",
    "ひ": "i",
    "み": "i",
    "り": "i",
    "ぎ": "i",
    "じ": "i",
    "ぢ": "i",
    "び": "i",
    "ぴ": "i",
    "う": "u",
    "く": "u",
    "す": "u",
    "つ": "u",
    "ぬ": "u",
    "ふ": "u",
    "む": "u",
    "ゆ": "u",
    "る": "u",
    "ぐ": "u",
    "ず": "u",
    "づ": "u",
    "ぶ": "u",
    "ぷ": "u",
    "え": "e",
    "け": "e",
    "せ": "e",
    "て": "e",
    "ね": "e",
    "へ": "e",
    "め": "e",
    "れ": "e",
    "げ": "e",
    "ぜ": "e",
    "で": "e",
    "べ": "e",
    "ぺ": "e",
    "お": "o",
    "こ": "o",
    "そ": "o",
    "と": "o",
    "の": "o",
    "ほ": "o",
    "も": "o",
    "よ": "o",
    "ろ": "o",
    "を": "o",
    "ご": "o",
    "ぞ": "o",
    "ど": "o",
    "ぼ": "o",
    "ぽ": "o",
}

SMALL_VOWEL = {
    "ぁ": "a",
    "ゃ": "a",
    "ぃ": "i",
    "ぅ": "u",
    "ゅ": "u",
    "ぇ": "e",
    "ぉ": "o",
    "ょ": "o",
}


@dataclass(frozen=True)
class UnitSpec:
    index: int
    label: str
    reading: str
    group: str
    onset_group: str
    vowel: str
    weight: float


@dataclass
class AudioLite:
    path: Path
    y: np.ndarray
    sr: int
    duration_sec: float
    mfcc: np.ndarray
    mel_db: np.ndarray
    f0: np.ndarray
    f0_times: np.ndarray


def setup_japanese_font() -> None:
    candidates = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def mora_vowel(mora: str, previous_vowel: str = "") -> str:
    if mora == "ー":
        return previous_vowel
    if mora in {"ん", "っ"}:
        return ""
    for char in reversed(mora):
        if char in SMALL_VOWEL:
            return SMALL_VOWEL[char]
        if char in BASE_VOWEL:
            return BASE_VOWEL[char]
    return ""


def classify_mora(mora: str, previous_vowel: str = "") -> tuple[str, str, str, float]:
    if mora in PUNCTUATION:
        pause = PUNCTUATION[mora]
        return ("pause", pause, "", 0.55 if pause == "minor_pause" else 0.85)
    if mora == "ー":
        return ("long_vowel", "none", previous_vowel, 0.65)
    if mora == "っ":
        return ("sokuon_closure", "geminate_closure", "", 0.35)
    if mora == "ん":
        return ("mora_nasal", "nasal_mora", "", 0.72)

    base = mora[0]
    vowel = mora_vowel(mora, previous_vowel)
    if base in VOWELS or base == "を":
        return ("vowel_only", "none", vowel, 0.95)
    if base in set("かきくけこたてとぱぴぷぺぽ"):
        return ("plosive", "voiceless_plosive", vowel, 1.0)
    if base in set("がぎぐげごだでどばびぶべぼ"):
        return ("voiced_plosive", "voiced_plosive", vowel, 1.05)
    if base in set("さすせそはひふへほ"):
        return ("fricative", "fricative", vowel, 1.0)
    if base in set("しざじずぜぞちつ"):
        return ("sibilant_affricate", "sibilant_affricate", vowel, 1.05)
    if base in set("なにぬねのまみむめも"):
        return ("nasal", "nasal", vowel, 0.95)
    if base in set("らりるれろ"):
        return ("tap", "tap", vowel, 0.9)
    if base in set("やゆよわ"):
        return ("glide", "glide", vowel, 0.9)
    return ("other_cv", "other_consonant", vowel, 1.0)


def tokenize_reading(reading: str) -> list[str]:
    reading = jaconv.kata2hira(reading)
    moras: list[str] = []
    for char in reading:
        if char.isspace():
            continue
        if char in PUNCTUATION:
            moras.append(char)
            continue
        if char == "ー":
            moras.append(char)
            continue
        if char in SMALL_KANA and moras and moras[-1] not in PUNCTUATION and moras[-1] != "ー":
            moras[-1] += char
            continue
        moras.append(char)
    return moras


def units_for_sample(sample: str) -> list[UnitSpec]:
    if sample not in READINGS:
        raise KeyError(f"missing reading for {sample}")
    units: list[UnitSpec] = []
    previous_vowel = ""
    for index, mora in enumerate(tokenize_reading(READINGS[sample]), start=1):
        group, onset_group, vowel, weight = classify_mora(mora, previous_vowel)
        if vowel:
            previous_vowel = vowel
        units.append(UnitSpec(index=index, label=mora, reading=mora, group=group, onset_group=onset_group, vowel=vowel, weight=weight))
    return units


def extract_audio(path: Path) -> AudioLite:
    y, sr = load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    f0, _voiced, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    return AudioLite(path=path, y=y, sr=sr, duration_sec=len(y) / sr, mfcc=mfcc, mel_db=mel_db, f0=f0, f0_times=f0_times)


def allocate_original_boundaries(audio: AudioLite, units: list[UnitSpec]) -> np.ndarray:
    weights = np.asarray([unit.weight for unit in units], dtype=np.float64)
    weights = np.maximum(weights, 0.05)
    boundaries = np.concatenate([[0.0], np.cumsum(weights) / np.sum(weights) * audio.duration_sec])

    rms = librosa.feature.rms(y=audio.y, frame_length=1024, hop_length=HOP_LENGTH)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=audio.sr, hop_length=HOP_LENGTH)
    if len(rms_times) < 4:
        return boundaries

    refined = boundaries.copy()
    for i in range(1, len(refined) - 1):
        target = refined[i]
        left_room = target - refined[i - 1]
        right_room = refined[i + 1] - target
        if left_room < 0.045 or right_room < 0.045:
            continue
        window = min(0.055, left_room * 0.35, right_room * 0.35)
        mask = (rms_times >= target - window) & (rms_times <= target + window)
        if np.sum(mask) < 3:
            continue
        local_times = rms_times[mask]
        local_rms = rms[mask]
        candidate = float(local_times[int(np.argmin(local_rms))])
        if refined[i - 1] + 0.025 < candidate < refined[i + 1] - 0.025:
            refined[i] = candidate
    refined[0] = 0.0
    refined[-1] = audio.duration_sec
    return np.maximum.accumulate(refined)


def stabilize_generated_boundaries(mapped: np.ndarray, units: list[UnitSpec], duration_sec: float) -> np.ndarray:
    """Keep DTW boundary mapping useful for very short mora windows.

    Raw DTW can map multiple adjacent original mora boundaries to the same
    generated frame when a model compresses a phrase. For phone-class analysis
    that creates zero-length windows, so we blend the DTW gaps with a small
    reading-duration prior and enforce a soft minimum gap.
    """
    mapped = np.asarray(mapped, dtype=np.float64)
    mapped[0] = 0.0
    mapped[-1] = duration_sec
    raw_gaps = np.maximum(np.diff(np.maximum.accumulate(mapped)), 0.0)
    weights = np.asarray([unit.weight for unit in units], dtype=np.float64)
    prior_gaps = weights / np.sum(weights) * duration_sec
    gaps = raw_gaps + 0.30 * prior_gaps
    min_gap = min(0.018, duration_sec / max(len(units) * 4.0, 1.0))
    gaps = np.maximum(gaps, min_gap)
    gaps = gaps / np.sum(gaps) * duration_sec
    return np.concatenate([[0.0], np.cumsum(gaps)])


def dtw_boundary_mapping(
    original: AudioLite,
    generated: AudioLite,
    original_boundaries: np.ndarray,
    units: list[UnitSpec],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _distance, orig_idx, gen_idx, local = dtw_path_and_local_distances(original.mfcc, generated.mfcc)
    order = np.argsort(orig_idx)
    xs = orig_idx[order].astype(np.float64)
    ys = gen_idx[order].astype(np.float64)
    unique_xs = np.unique(xs)
    unique_ys = np.asarray([np.median(ys[xs == x]) for x in unique_xs], dtype=np.float64)
    boundary_frames = original_boundaries * original.sr / HOP_LENGTH
    gen_frames = np.interp(boundary_frames, unique_xs, unique_ys, left=0.0, right=max(len(generated.f0) - 1, 0))
    generated_boundaries = librosa.frames_to_time(gen_frames, sr=generated.sr, hop_length=HOP_LENGTH)
    generated_boundaries[0] = 0.0
    generated_boundaries[-1] = generated.duration_sec
    generated_boundaries = np.maximum.accumulate(np.clip(generated_boundaries, 0.0, generated.duration_sec))
    generated_boundaries = stabilize_generated_boundaries(generated_boundaries, units, generated.duration_sec)
    return generated_boundaries, orig_idx, gen_idx, local


def segment(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, min(len(y), int(round(start_sec * sr))))
    end = max(start, min(len(y), int(round(end_sec * sr))))
    return y[start:end]


def f0_stats(audio: AudioLite, start_sec: float, end_sec: float) -> tuple[float, float]:
    mask = (audio.f0_times >= start_sec) & (audio.f0_times <= end_sec) & np.isfinite(audio.f0)
    values = audio.f0[mask]
    if values.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.median(values)), float(np.percentile(values, 90) - np.percentile(values, 10)) if values.size >= 3 else 0.0)


def mfcc_vector(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size < 64:
        return np.zeros(26, dtype=np.float64)
    frame = min(N_FFT, max(256, int(2 ** math.floor(math.log2(max(y.size, 256))))))
    hop = min(HOP_LENGTH, max(64, frame // 4))
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=40, n_fft=frame, hop_length=hop)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def spectral_stats(y: np.ndarray, sr: int) -> dict[str, float]:
    if y.size < 64:
        return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0, "highband_ratio": 0.0}
    frame = min(N_FFT, max(256, int(2 ** math.floor(math.log2(max(y.size, 256))))))
    hop = min(HOP_LENGTH, max(64, frame // 4))
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame, hop_length=hop)[0]))
    power = np.abs(librosa.stft(y, n_fft=frame, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame)
    total = float(np.mean(np.sum(power, axis=0))) + 1e-12
    high = float(np.mean(np.sum(power[freqs >= 3500], axis=0))) if power.size else 0.0
    return {"rms": rms, "zcr": zcr, "centroid": centroid, "highband_ratio": high / total}


def subwindows(audio: AudioLite, start_sec: float, end_sec: float, unit: UnitSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = segment(audio.y, audio.sr, start_sec, end_sec)
    dur = max(end_sec - start_sec, 1e-6)
    if unit.onset_group != "none" and unit.group != "pause":
        onset_end = start_sec + min(0.070, dur * 0.45)
        nucleus_start = start_sec + min(0.055, dur * 0.35)
    else:
        onset_end = start_sec + min(0.025, dur * 0.25)
        nucleus_start = start_sec + dur * 0.18
    nucleus_end = end_sec - dur * 0.08
    onset = segment(audio.y, audio.sr, start_sec, onset_end)
    nucleus = segment(audio.y, audio.sr, nucleus_start, max(nucleus_start + 0.01, nucleus_end))
    return full, onset, nucleus


def formant_distance(original: tuple[float, float, float], generated: tuple[float, float, float]) -> tuple[float, float, float, float]:
    o = np.asarray(original, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    mask = np.isfinite(o) & np.isfinite(g)
    if np.sum(mask) == 0:
        return (float("nan"), float("nan"), float("nan"), 0.0)
    rel = float(np.nanmean(np.abs(g[mask] - o[mask]) / np.maximum(o[mask], 1.0)))
    f1_delta = float(g[0] - o[0]) if np.isfinite(o[0]) and np.isfinite(g[0]) else float("nan")
    f2_delta = float(g[1] - o[1]) if np.isfinite(o[1]) and np.isfinite(g[1]) else float("nan")
    return (rel, f1_delta, f2_delta, exp_similarity(rel, 0.35))


def unit_metrics(
    sample: str,
    model_key: str,
    unit: UnitSpec,
    original: AudioLite,
    generated: AudioLite,
    original_bounds: np.ndarray,
    generated_bounds: np.ndarray,
) -> dict[str, object]:
    os = float(original_bounds[unit.index - 1])
    oe = float(original_bounds[unit.index])
    gs = float(generated_bounds[unit.index - 1])
    ge = float(generated_bounds[unit.index])
    odur = max(oe - os, 1e-6)
    gdur = max(ge - gs, 1e-6)
    duration_ratio = gdur / odur
    duration_sim = exp_similarity(abs(math.log(max(duration_ratio, 1e-6))), 0.45)

    ofull, oonset, onucleus = subwindows(original, os, oe, unit)
    gfull, gonset, gnucleus = subwindows(generated, gs, ge, unit)

    full_mfcc = cosine01(mfcc_vector(ofull, original.sr), mfcc_vector(gfull, generated.sr))
    onset_mfcc = cosine01(mfcc_vector(oonset, original.sr), mfcc_vector(gonset, generated.sr))
    nucleus_mfcc = cosine01(mfcc_vector(onucleus, original.sr), mfcc_vector(gnucleus, generated.sr))

    ospec = spectral_stats(oonset, original.sr)
    gspec = spectral_stats(gonset, generated.sr)
    highband_delta = gspec["highband_ratio"] - ospec["highband_ratio"]
    zcr_delta = gspec["zcr"] - ospec["zcr"]
    centroid_delta = gspec["centroid"] - ospec["centroid"]
    highband_sim = exp_similarity(abs(highband_delta), 0.22)
    zcr_sim = exp_similarity(abs(zcr_delta), 0.08)
    centroid_sim = exp_similarity(abs(centroid_delta), 1700.0)
    onset_similarity = 0.48 * onset_mfcc + 0.22 * highband_sim + 0.17 * zcr_sim + 0.13 * centroid_sim

    of0, of0_range = f0_stats(original, os, oe)
    gf0, gf0_range = f0_stats(generated, gs, ge)
    f0_delta = gf0 - of0 if np.isfinite(of0) and np.isfinite(gf0) else float("nan")
    f0_range_delta = gf0_range - of0_range if np.isfinite(of0_range) and np.isfinite(gf0_range) else float("nan")
    f0_sim = exp_similarity(abs(f0_delta), 45.0) if np.isfinite(f0_delta) else 0.0

    o_formants = lpc_formants(onucleus, original.sr)
    g_formants = lpc_formants(gnucleus, generated.sr)
    formant_rel, f1_delta, f2_delta, formant_sim = formant_distance(o_formants, g_formants)
    nucleus_similarity = 0.45 * nucleus_mfcc + 0.27 * formant_sim + 0.20 * f0_sim + 0.08 * duration_sim

    if unit.group == "pause":
        orms = spectral_stats(ofull, original.sr)["rms"]
        grms = spectral_stats(gfull, generated.sr)["rms"]
        rms_delta = 20 * math.log10(max(grms, 1e-9)) - 20 * math.log10(max(orms, 1e-9))
        silence_sim = exp_similarity(abs(rms_delta), 16.0)
        unit_similarity = 0.70 * duration_sim + 0.30 * silence_sim
    elif unit.group in {"vowel_only", "long_vowel"}:
        rms_delta = float("nan")
        unit_similarity = 0.52 * nucleus_similarity + 0.30 * full_mfcc + 0.18 * duration_sim
    elif unit.group in {"mora_nasal", "sokuon_closure"}:
        rms_delta = float("nan")
        unit_similarity = 0.38 * full_mfcc + 0.32 * onset_similarity + 0.20 * duration_sim + 0.10 * f0_sim
    else:
        rms_delta = float("nan")
        unit_similarity = 0.34 * onset_similarity + 0.31 * nucleus_similarity + 0.23 * full_mfcc + 0.12 * duration_sim

    return {
        "sample": sample,
        "model": model_key,
        "model_label": MODEL_BY_KEY[model_key].label,
        "unit_index": unit.index,
        "unit_label": unit.label,
        "unit_group": unit.group,
        "onset_group": unit.onset_group,
        "vowel": unit.vowel,
        "original_start_sec": os,
        "original_end_sec": oe,
        "generated_start_sec": gs,
        "generated_end_sec": ge,
        "original_duration_sec": odur,
        "generated_duration_sec": gdur,
        "duration_ratio": duration_ratio,
        "duration_similarity": duration_sim,
        "unit_similarity": unit_similarity,
        "full_mfcc_similarity": full_mfcc,
        "consonant_onset_similarity": onset_similarity,
        "vowel_nucleus_similarity": nucleus_similarity,
        "nucleus_mfcc_similarity": nucleus_mfcc,
        "formant_relative_distance": formant_rel,
        "f1_delta_hz": f1_delta,
        "f2_delta_hz": f2_delta,
        "f0_median_delta_hz": f0_delta,
        "f0_range_delta_hz": f0_range_delta,
        "onset_centroid_delta_hz": centroid_delta,
        "onset_highband_delta": highband_delta,
        "onset_zcr_delta": zcr_delta,
        "pause_rms_delta_db": rms_delta,
    }


def mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    groups = [
        ("all_mora", lambda row: row["unit_group"] != "pause"),
        ("vowel_nucleus", lambda row: row["unit_group"] in {"vowel_only", "long_vowel"} or bool(row["vowel"])),
        ("consonant_onset", lambda row: row["onset_group"] not in {"none", "minor_pause", "final_pause"}),
        ("plosive", lambda row: row["unit_group"] in {"plosive", "voiced_plosive"}),
        ("fricative_affricate", lambda row: row["unit_group"] in {"fricative", "sibilant_affricate"}),
        ("nasal", lambda row: row["unit_group"] in {"nasal", "mora_nasal"}),
        ("tap_glide", lambda row: row["unit_group"] in {"tap", "glide"}),
        ("pause_timing", lambda row: row["unit_group"] == "pause"),
    ]
    for model in MODEL_SPECS:
        model_rows = [row for row in rows if row["model"] == model.key]
        for group_name, predicate in groups:
            group_rows = [row for row in model_rows if predicate(row)]
            if not group_rows:
                continue
            summary.append(
                {
                    "model": model.key,
                    "model_label": model.label,
                    "analysis_group": group_name,
                    "count": len(group_rows),
                    "unit_similarity": mean_finite([float(row["unit_similarity"]) for row in group_rows]),
                    "duration_ratio": mean_finite([float(row["duration_ratio"]) for row in group_rows]),
                    "consonant_onset_similarity": mean_finite([float(row["consonant_onset_similarity"]) for row in group_rows]),
                    "vowel_nucleus_similarity": mean_finite([float(row["vowel_nucleus_similarity"]) for row in group_rows]),
                    "abs_f0_delta_hz": mean_finite([abs(float(row["f0_median_delta_hz"])) for row in group_rows]),
                    "formant_relative_distance": mean_finite([float(row["formant_relative_distance"]) for row in group_rows]),
                    "abs_onset_centroid_delta_hz": mean_finite([abs(float(row["onset_centroid_delta_hz"])) for row in group_rows]),
                }
            )
    return summary


def worst_units(rows: list[dict[str, object]], per_model: int = 10) -> list[dict[str, object]]:
    worst: list[dict[str, object]] = []
    for model in MODEL_SPECS:
        model_rows = [row for row in rows if row["model"] == model.key and row["unit_group"] != "pause"]
        worst.extend(sorted(model_rows, key=lambda row: float(row["unit_similarity"]))[:per_model])
    return worst


def plot_class_heatmap(summary_rows: list[dict[str, object]], output: Path) -> None:
    groups = ["all_mora", "vowel_nucleus", "consonant_onset", "plosive", "fricative_affricate", "nasal", "tap_glide", "pause_timing"]
    matrix = np.full((len(MODEL_SPECS), len(groups)), np.nan)
    for y, model in enumerate(MODEL_SPECS):
        for x, group in enumerate(groups):
            match = [row for row in summary_rows if row["model"] == model.key and row["analysis_group"] == group]
            if match:
                matrix[y, x] = float(match[0]["unit_similarity"])
    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.45, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(MODEL_SPECS)))
    ax.set_yticklabels([model.label for model in MODEL_SPECS], fontsize=9)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(
        ["All", "Vowel nucleus", "Consonant onset", "Plosive", "Fric./Affr.", "Nasal", "Tap/Glide", "Pause"],
        rotation=25,
        ha="right",
    )
    ax.set_title("Mora / phone-class acoustic similarity")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Similarity (higher is better)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_mora_heatmap(sample: str, units: list[UnitSpec], rows: list[dict[str, object]], output: Path) -> None:
    sample_rows = [row for row in rows if row["sample"] == sample]
    matrix = np.full((len(MODEL_SPECS), len(units)), np.nan)
    for y, model in enumerate(MODEL_SPECS):
        for row in sample_rows:
            if row["model"] == model.key:
                matrix[y, int(row["unit_index"]) - 1] = float(row["unit_similarity"])
    fig_w = max(13, len(units) * 0.34)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.45, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(MODEL_SPECS)))
    ax.set_yticklabels([model.label for model in MODEL_SPECS], fontsize=8)
    ax.set_xticks(np.arange(len(units)))
    ax.set_xticklabels([unit.label for unit in units], rotation=0, fontsize=8)
    ax.set_title(f"Mora-level similarity heatmap: {sample}")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]) and len(units) <= 42:
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(image, ax=ax, label="Unit similarity")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_annotated_spectrogram(
    sample: str,
    text: str,
    units: list[UnitSpec],
    audio_map: dict[str, AudioLite],
    boundary_map: dict[str, np.ndarray],
    output: Path,
) -> None:
    keys = ["original"] + [model.key for model in MODEL_SPECS]
    labels = {"original": "Original H4n"} | {model.key: model.label for model in MODEL_SPECS}
    fig, axes = plt.subplots(len(keys), 1, figsize=(15, 12.5), constrained_layout=True)
    fig.suptitle(f"{sample}: {text}", fontsize=14)
    for ax, key in zip(axes, keys):
        audio = audio_map[key]
        librosa.display.specshow(
            audio.mel_db,
            sr=audio.sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            vmin=-80,
            vmax=0,
            ax=ax,
        )
        ax.set_ylabel(labels[key], fontsize=9)
        bounds = boundary_map[key]
        for boundary in bounds[1:-1]:
            ax.axvline(boundary, color="white", alpha=0.18, linewidth=0.45)
        label_stride = max(1, int(math.ceil(len(units) / 34)))
        y_text = 0.96
        for i, unit in enumerate(units):
            if i % label_stride != 0 and unit.group != "pause":
                continue
            x = (bounds[i] + bounds[i + 1]) / 2
            color = "cyan" if unit.group != "pause" else "white"
            ax.text(x, y_text, unit.label, color=color, fontsize=7, ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.label_outer()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def plot_warped_f0(
    sample: str,
    text: str,
    units: list[UnitSpec],
    original: AudioLite,
    audio_map: dict[str, AudioLite],
    boundary_map: dict[str, np.ndarray],
    mapping_map: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.0), constrained_layout=True)
    fig.suptitle(f"{sample}: {text}", fontsize=12)
    ax.plot(original.f0_times, original.f0, color="black", linewidth=2.2, label="Original H4n")
    for model in MODEL_SPECS:
        generated = audio_map[model.key]
        orig_idx, gen_idx = mapping_map[model.key]
        orig_times = librosa.frames_to_time(orig_idx, sr=original.sr, hop_length=HOP_LENGTH)
        gen_times = librosa.frames_to_time(gen_idx, sr=generated.sr, hop_length=HOP_LENGTH)
        gen_f0_at_path = np.interp(gen_times, generated.f0_times, np.nan_to_num(generated.f0, nan=np.nan))
        valid = np.isfinite(gen_f0_at_path)
        ax.plot(orig_times[valid], gen_f0_at_path[valid], color=model.color, linewidth=1.15, alpha=0.84, label=model.label)
    bounds = boundary_map["original"]
    for boundary in bounds[1:-1]:
        ax.axvline(boundary, color="gray", alpha=0.18, linewidth=0.55)
    label_stride = max(1, int(math.ceil(len(units) / 36)))
    for i, unit in enumerate(units):
        if i % label_stride != 0 and unit.group != "pause":
            continue
        x = (bounds[i] + bounds[i + 1]) / 2
        ax.text(
            x,
            0.985,
            unit.label,
            fontsize=8,
            ha="center",
            va="top",
            color="dimgray",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_title("DTW-warped F0 contour on original H4n timeline", pad=10)
    ax.set_xlabel("Original H4n time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(60, 280)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_vowel_formant_space(rows: list[dict[str, object]], output: Path) -> None:
    vowels = ["a", "i", "u", "e", "o"]
    fig, axes = plt.subplots(1, len(vowels), figsize=(17, 4.8), constrained_layout=True, sharex=True, sharey=True)
    for ax, vowel in zip(axes, vowels):
        vowel_rows = [row for row in rows if row["vowel"] == vowel and row["unit_group"] != "pause"]
        for model in MODEL_SPECS:
            model_rows = [row for row in vowel_rows if row["model"] == model.key]
            if not model_rows:
                continue
            f1_delta = mean_finite([float(row["f1_delta_hz"]) for row in model_rows])
            f2_delta = mean_finite([float(row["f2_delta_hz"]) for row in model_rows])
            if np.isfinite(f1_delta) and np.isfinite(f2_delta):
                ax.scatter(f2_delta, f1_delta, color=model.color, label=model.label, s=42)
                ax.axhline(0, color="gray", linewidth=0.7, alpha=0.35)
                ax.axvline(0, color="gray", linewidth=0.7, alpha=0.35)
        ax.set_title(f"/{vowel}/")
        ax.set_xlabel("F2 delta (Hz)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("F1 delta (Hz)")
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    fig.suptitle("Approximate vowel formant deltas against original H4n (closer to 0 is better)", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def relative_link(from_path: Path, to_path: Path) -> str:
    return Path(os.path.relpath(to_path, start=from_path.parent)).as_posix()


def write_report(
    path: Path,
    targets: list[Target],
    summary_rows: list[dict[str, object]],
    unit_rows: list[dict[str, object]],
    worst_rows: list[dict[str, object]],
    figures_dir: Path,
) -> None:
    fig = lambda name: relative_link(path, figures_dir / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 音素・モーラ単位の追加音響分析\n\n")
        handle.write("Date: 2026-06-18\n\n")
        handle.write("## 目的\n\n")
        handle.write(
            "既存の同一文比較は、MFCC、メルスペクトログラム、F0、フォルマントを文全体または発話セグメント単位で比較していた。"
            "本分析では粒度を上げ、各文の読みをモーラ列に分解し、母音核、子音オンセット、破裂音、摩擦音/破擦音、鼻音、撥音、ポーズに分けて比較した。\n\n"
        )
        handle.write("## 方法\n\n")
        handle.write(
            "Praat/TextGridによる手動または強制アラインメントではなく、Pythonのみで完結する近似アラインメントを使った。"
            "まず日本語テキストに対して読みを定義し、モーラ列を作る。次に元H4n音声上へモーラ境界を時間配分し、RMS谷で軽く補正する。"
            "各モデル音声へはMFCC-DTWの対応から境界を写像する。したがって、これは厳密な音素境界ではなく、同じテキスト位置を比較するための疑似モーラ/疑似音素アラインメントである。\n\n"
        )
        handle.write("抽出した主な指標は以下。\n\n")
        handle.write("| 指標 | 見ているもの | 読み方 |\n")
        handle.write("|---|---|---|\n")
        handle.write("| `unit_similarity` | モーラ全体のMFCC、F0、フォルマント、長さ、オンセットを混ぜた近似スコア | 高いほどそのモーラが元音声に近い |\n")
        handle.write("| `vowel_nucleus_similarity` | 母音核のMFCC、F1/F2/F3、F0 | 高いほど母音の響き・声の高さが近い |\n")
        handle.write("| `consonant_onset_similarity` | 子音立ち上がりのMFCC、高域比、ZCR、スペクトル重心 | 高いほど子音の立ち上がりが近い |\n")
        handle.write("| `formant_relative_distance` | 母音核のフォルマント相対距離 | 低いほど声道共鳴が近い |\n")
        handle.write("| `f0_median_delta_hz` | モーラごとのF0中央値差 | 0Hzに近いほど声の高さが近い |\n\n")

        handle.write("## モデル別・音声カテゴリ別サマリ\n\n")
        handle.write("![Phone class heatmap](" + fig("phone_class_similarity_heatmap.png") + ")\n\n")
        handle.write("| Model | All mora | Vowel nucleus | Consonant onset | Plosive | Fric./Affr. | Nasal | Pause |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for model in MODEL_SPECS:
            values = {}
            for row in summary_rows:
                if row["model"] == model.key:
                    values[str(row["analysis_group"])] = float(row["unit_similarity"])
            handle.write(
                f"| {model.label} | "
                f"{values.get('all_mora', float('nan')):.4f} | "
                f"{values.get('vowel_nucleus', float('nan')):.4f} | "
                f"{values.get('consonant_onset', float('nan')):.4f} | "
                f"{values.get('plosive', float('nan')):.4f} | "
                f"{values.get('fricative_affricate', float('nan')):.4f} | "
                f"{values.get('nasal', float('nan')):.4f} | "
                f"{values.get('pause_timing', float('nan')):.4f} |\n"
            )

        handle.write("\n## 注釈付きスペクトログラムとDTW補正F0\n\n")
        for target in targets:
            text = target.text_file.read_text(encoding="utf-8").strip()
            handle.write(f"### {target.sample}\n\n")
            handle.write(f"{text}\n\n")
            handle.write(f"![Annotated spectrogram {target.sample}]({fig(f'annotated_spectrogram_{target.sample}.png')})\n\n")
            handle.write(f"![Mora heatmap {target.sample}]({fig(f'mora_similarity_heatmap_{target.sample}.png')})\n\n")
            handle.write(f"![Warped F0 {target.sample}]({fig(f'warped_f0_mora_{target.sample}.png')})\n\n")

        handle.write("## 母音フォルマント\n\n")
        handle.write("![Vowel formant delta](" + fig("vowel_formant_delta.png") + ")\n\n")
        handle.write(
            "各点は、元H4n音声に対するF1/F2の平均差である。原点に近いほど、その母音カテゴリの声道共鳴が近い。"
            "ただし短いモーラからLPCで推定しているため、Praatで母音核を手動確認した値ほど安定ではない。\n\n"
        )

        handle.write("## 低スコア単位\n\n")
        handle.write("| Model | Sample | Unit | Group | Similarity | F0 delta | F1 delta | F2 delta | Duration ratio |\n")
        handle.write("|---|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in worst_rows:
            handle.write(
                f"| {row['model_label']} | `{row['sample']}` | {row['unit_index']}:{row['unit_label']} | {row['unit_group']} | "
                f"{float(row['unit_similarity']):.4f} | "
                f"{float(row['f0_median_delta_hz']):.1f} | "
                f"{float(row['f1_delta_hz']):.1f} | "
                f"{float(row['f2_delta_hz']):.1f} | "
                f"{float(row['duration_ratio']):.3f} |\n"
            )

        handle.write("\n## 解釈上の注意\n\n")
        handle.write(
            "- この分析は音素強制アラインメントではなく、疑似モーラ境界とDTW写像である。音節・音素境界の厳密性が必要な場合は、Praat/TextGridまたは日本語対応の強制アラインメントを使うべきである。\n"
            "- 子音の立ち上がりは非常に短いため、サンプルレート、窓幅、DTW写像のズレに敏感である。破裂音・摩擦音の数値は、ランキングよりも異常箇所探索として読む。\n"
            "- フォルマントはLPC近似である。短い母音核、無声化母音、背景ノイズ、TTS特有の倍音構造で大きく揺れる。\n"
            "- それでも、文全体のCompositeでは見えにくい「どの音の種類でズレるか」を把握するには有用である。\n\n"
        )
        handle.write("## Artifacts\n\n")
        handle.write("- `docs/comparisons/phonetic_micro_metrics.tsv`\n")
        handle.write("- `docs/comparisons/phonetic_micro_summary.tsv`\n")
        handle.write("- `docs/comparisons/phonetic_micro_worst_units.tsv`\n")
        handle.write("- `docs/assets/acoustic_phonetic_micro/`\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mora/phone-class micro acoustic analysis for same-text TTS outputs.")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "samples" / "manifests" / "similarity_eval_targets.tsv"))
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "outputs" / "similarity_eval"))
    parser.add_argument("--metrics-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "phonetic_micro_metrics.tsv"))
    parser.add_argument("--summary-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "phonetic_micro_summary.tsv"))
    parser.add_argument("--worst-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "phonetic_micro_worst_units.tsv"))
    parser.add_argument("--report-path", default=str(PROJECT_ROOT / "docs" / "comparisons" / "phonetic_micro_report_ja.md"))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    setup_japanese_font()
    targets = read_manifest(Path(args.manifest))
    generated_root = Path(args.generated_root)
    figures_dir = Path(args.figures_dir)

    unit_rows: list[dict[str, object]] = []
    per_sample_context: dict[str, tuple[list[UnitSpec], dict[str, AudioLite], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    for target in targets:
        units = units_for_sample(target.sample)
        paths = generated_paths(target.sample, generated_root)
        missing = [model.key for model in MODEL_SPECS if model.key not in paths or not paths[model.key].exists()]
        if missing:
            raise FileNotFoundError(f"missing generated audio for {target.sample}: {missing}")

        audio_map: dict[str, AudioLite] = {"original": extract_audio(target.original_wav)}
        boundary_map: dict[str, np.ndarray] = {"original": allocate_original_boundaries(audio_map["original"], units)}
        mapping_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for model in MODEL_SPECS:
            generated = extract_audio(paths[model.key])
            audio_map[model.key] = generated
            generated_bounds, orig_idx, gen_idx, _local = dtw_boundary_mapping(audio_map["original"], generated, boundary_map["original"], units)
            boundary_map[model.key] = generated_bounds
            mapping_map[model.key] = (orig_idx, gen_idx)
            for unit in units:
                unit_rows.append(unit_metrics(target.sample, model.key, unit, audio_map["original"], generated, boundary_map["original"], generated_bounds))

        per_sample_context[target.sample] = (units, audio_map, boundary_map, mapping_map)

        text = target.text_file.read_text(encoding="utf-8").strip()
        plot_mora_heatmap(target.sample, units, unit_rows, figures_dir / f"mora_similarity_heatmap_{target.sample}.png")
        plot_annotated_spectrogram(target.sample, text, units, audio_map, boundary_map, figures_dir / f"annotated_spectrogram_{target.sample}.png")
        plot_warped_f0(
            target.sample,
            text,
            units,
            audio_map["original"],
            audio_map,
            boundary_map,
            mapping_map,
            figures_dir / f"warped_f0_mora_{target.sample}.png",
        )

    summary_rows = summarize(unit_rows)
    worst_rows = worst_units(unit_rows, per_model=8)

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_class_heatmap(summary_rows, figures_dir / "phone_class_similarity_heatmap.png")
    plot_vowel_formant_space(unit_rows, figures_dir / "vowel_formant_delta.png")

    write_tsv(Path(args.metrics_path), unit_rows)
    write_tsv(Path(args.summary_path), summary_rows)
    write_tsv(Path(args.worst_path), worst_rows)
    write_report(Path(args.report_path), targets, summary_rows, unit_rows, worst_rows, figures_dir)

    print(f"wrote\t{args.metrics_path}")
    print(f"wrote\t{args.summary_path}")
    print(f"wrote\t{args.worst_path}")
    print(f"wrote\t{args.report_path}")
    print(f"wrote\t{args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
