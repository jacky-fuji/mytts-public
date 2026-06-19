from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CANDIDATES = [
    ROOT / "docs" / "comparisons" / "generation_time_benchmark_summary.tsv",
    ROOT / "data" / "metrics" / "generation_time_benchmark_summary.tsv",
]
OUT = ROOT / "docs" / "assets" / "generation_time_rtf_bar.png"

W, H = 1600, 880
BG = (247, 250, 253)
PAPER = (255, 255, 255)
NAVY = (18, 32, 55)
INK = (34, 47, 68)
MUTED = (92, 106, 127)
LINE = (220, 229, 240)
TEAL = (20, 152, 145)
BLUE = (55, 125, 213)
PURPLE = (126, 96, 211)
CORAL = (236, 102, 92)
YELLOW = (242, 184, 63)
GREEN = (77, 170, 99)
BROWN = (156, 105, 63)

COLORS = {
    "Irodori-TTS 500M": BLUE,
    "Irodori-TTS 600M VoiceDesign": TEAL,
    "VoxCPM2 ultimate": PURPLE,
    "CosyVoice2": GREEN,
    "Qwen3-TTS 1.7B": YELLOW,
    "Fish Speech S2 Pro": CORAL,
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


FONT_TITLE = font("YuGothB.ttc", 42)
FONT_H = font("YuGothB.ttc", 26)
FONT_M = font("YuGothM.ttc", 22)
FONT_S = font("YuGothM.ttc", 18)
FONT_XS = font("YuGothM.ttc", 15)


def load_rows() -> list[dict[str, object]]:
    summary_path = next((path for path in SUMMARY_CANDIDATES if path.exists()), SUMMARY_CANDIDATES[0])
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "model": row["model"],
                "count": int(row["sample_count"]),
                "audio": float(row["total_audio_duration_sec"]),
                "elapsed": float(row["total_seconds_elapsed"]),
                "wall": float(row["total_wall_seconds"]),
                "rtf": float(row["avg_rtf"]),
                "wall_rtf": float(row["avg_wall_rtf"]),
                "color": COLORS.get(row["model"], BROWN),
            }
        )
    return sorted(out, key=lambda row: float(row["rtf"]))


def text(draw: ImageDraw.ImageDraw, xy, value, fill=INK, font_obj=FONT_M, anchor=None):
    draw.text(xy, value, fill=fill, font=font_obj, anchor=anchor)


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def log_pos(value: float, min_v: float, max_v: float, x0: int, x1: int) -> float:
    value = max(value, min_v)
    return x0 + (math.log10(value) - math.log10(min_v)) / (math.log10(max_v) - math.log10(min_v)) * (x1 - x0)


def tick_values(max_v: float) -> list[float]:
    values = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    while values[-1] < max_v:
        values.append(values[-1] * 2)
    return values


def main() -> None:
    rows = load_rows()
    max_rtf = max(float(row["rtf"]) for row in rows)
    min_v = 0.08
    max_v = max(4.0, 2 ** math.ceil(math.log2(max_rtf)))

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image, "RGBA")

    rounded(draw, (54, 52, W - 54, H - 52), 28, PAPER, outline=LINE, width=2)
    text(draw, (96, 92), "生成時間の比較", fill=NAVY, font_obj=FONT_TITLE)
    text(draw, (98, 150), "同じ3文を生成した平均RTF。短いほど速い。横軸は対数目盛り。", fill=MUTED, font_obj=FONT_M)

    chart_x0, chart_x1 = 590, 1438
    chart_y0, row_h = 250, 80

    direction_label_y = chart_y0 - 34
    for tick in tick_values(max_v):
        if tick < min_v or tick > max_v:
            continue
        x = log_pos(tick, min_v, max_v, chart_x0, chart_x1)
        draw.line((x, chart_y0 - 30, x, chart_y0 + row_h * len(rows) - 8), fill=(229, 235, 244), width=1)
        label = f"{tick:g}x"
        text(draw, (x, chart_y0 - 62), label, fill=MUTED, font_obj=FONT_XS, anchor="ma")
    text(draw, (chart_x0, direction_label_y), "速い", fill=TEAL, font_obj=FONT_S, anchor="ma")
    text(draw, (chart_x1, direction_label_y), "遅い", fill=CORAL, font_obj=FONT_S, anchor="ma")

    for idx, row in enumerate(rows):
        y = chart_y0 + idx * row_h
        model = str(row["model"])
        audio = float(row["audio"])
        elapsed = float(row["elapsed"])
        rtf = float(row["rtf"])
        text(draw, (98, y - 2), model, fill=NAVY, font_obj=FONT_H)
        text(draw, (100, y + 34), f"3文 / 音声 {audio:.1f}s / 生成 {elapsed:.1f}s", fill=MUTED, font_obj=FONT_XS)
        bar_end = log_pos(rtf, min_v, max_v, chart_x0, chart_x1)
        rounded(draw, (chart_x0, y + 6, chart_x1, y + 42), 17, (238, 243, 249), outline=None)
        rounded(draw, (chart_x0, y + 6, bar_end, y + 42), 17, row["color"], outline=None)
        label_x = min(chart_x1 - 8, bar_end + 16)
        text(draw, (label_x, y + 9), f"RTF {rtf:.2f}x", fill=INK, font_obj=FONT_S)

    rounded(draw, (96, H - 142, W - 96, H - 88), 18, (241, 247, 254), outline=(217, 229, 244))
    text(
        draw,
        (124, H - 130),
        "注: 短文、長めナレーション、技術語入り文の3本で測定。RTF=生成時間÷生成音声長。",
        fill=MUTED,
        font_obj=FONT_S,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
