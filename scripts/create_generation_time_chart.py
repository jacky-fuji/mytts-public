from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
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


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


FONT_TITLE = font("YuGothB.ttc", 42)
FONT_H = font("YuGothB.ttc", 27)
FONT_M = font("YuGothM.ttc", 22)
FONT_S = font("YuGothM.ttc", 18)
FONT_XS = font("YuGothM.ttc", 15)
FONT_EN = font("bahnschrift.ttf", 20)


RUNS = [
    {
        "run": "EV launch",
        "model": "Irodori-TTS 600M VoiceDesign",
        "audio": "401.25s",
        "wall": "561.4s",
        "rtf": 1.4,
        "color": TEAL,
    },
    {
        "run": "H4n presentation",
        "model": "Irodori-TTS",
        "audio": "約115.1s",
        "wall": "164.3s",
        "rtf": 1.4,
        "color": BLUE,
    },
    {
        "run": "Same-text",
        "model": "VoxCPM2",
        "audio": "約45.1s",
        "wall": "73.5s",
        "rtf": 1.6,
        "color": PURPLE,
    },
    {
        "run": "Same-text",
        "model": "CosyVoice2",
        "audio": "約23.9s",
        "wall": "44.4s",
        "rtf": 1.9,
        "color": GREEN,
    },
    {
        "run": "H4n presentation",
        "model": "Qwen3-TTS 1.7B",
        "audio": "約48.2s",
        "wall": "112.3s",
        "rtf": 2.3,
        "color": YELLOW,
    },
    {
        "run": "H4n presentation",
        "model": "Fish Speech S2 Pro",
        "audio": "約48.8s",
        "wall": "5253.3s",
        "rtf": 107.7,
        "color": CORAL,
    },
]


def text(draw: ImageDraw.ImageDraw, xy, value, fill=INK, font_obj=FONT_M, anchor=None):
    draw.text(xy, value, fill=fill, font=font_obj, anchor=anchor)


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def log_pos(value: float, x0: int, x1: int) -> float:
    min_v, max_v = 1.0, 128.0
    return x0 + (math.log10(value) - math.log10(min_v)) / (math.log10(max_v) - math.log10(min_v)) * (x1 - x0)


def main() -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image, "RGBA")

    rounded(draw, (54, 52, W - 54, H - 52), 28, PAPER, outline=LINE, width=2)
    text(draw, (96, 92), "生成時間の比較", fill=NAVY, font_obj=FONT_TITLE)
    text(draw, (98, 150), "RTF目安で昇順に並べた。短いほど速い。横軸は対数目盛り。", fill=MUTED, font_obj=FONT_M)

    chart_x0, chart_x1 = 560, 1438
    chart_y0, row_h = 250, 80

    ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    for tick in ticks:
        x = log_pos(tick, chart_x0, chart_x1)
        draw.line((x, chart_y0 - 30, x, chart_y0 + row_h * len(RUNS) - 8), fill=(229, 235, 244), width=1)
        text(draw, (x, chart_y0 - 62), f"{tick}x", fill=MUTED, font_obj=FONT_XS, anchor="ma")
    text(draw, (chart_x0, chart_y0 - 92), "速い", fill=TEAL, font_obj=FONT_S, anchor="ma")
    text(draw, (chart_x1, chart_y0 - 92), "遅い", fill=CORAL, font_obj=FONT_S, anchor="ma")

    for idx, row in enumerate(RUNS):
        y = chart_y0 + idx * row_h
        text(draw, (98, y - 2), row["model"], fill=NAVY, font_obj=FONT_H)
        text(draw, (100, y + 34), f"{row['run']} / 音声 {row['audio']} / Wall {row['wall']}", fill=MUTED, font_obj=FONT_XS)
        bar_end = log_pos(row["rtf"], chart_x0, chart_x1)
        rounded(draw, (chart_x0, y + 6, chart_x1, y + 42), 17, (238, 243, 249), outline=None)
        rounded(draw, (chart_x0, y + 6, bar_end, y + 42), 17, row["color"], outline=None)
        text(draw, (min(chart_x1 - 8, bar_end + 16), y + 9), f"RTF {row['rtf']:.1f}x", fill=INK, font_obj=FONT_S)

    rounded(draw, (96, H - 142, W - 96, H - 88), 18, (241, 247, 254), outline=(217, 229, 244))
    text(
        draw,
        (124, H - 130),
        "注: Runごとに生成音声の合計長が異なるため、Wall timeの単純比較ではなくRTFを主指標にした。",
        fill=MUTED,
        font_obj=FONT_S,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
