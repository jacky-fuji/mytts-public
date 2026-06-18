from __future__ import annotations

import csv
import html
import json
import re
import wave
from collections import defaultdict
from pathlib import Path

from jinja2 import Template


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MD_PATH = DOCS / "local_japanese_tts_voice_clone_public_report.md"
OUT_PATH = DOCS / "local_japanese_tts_voice_clone_interactive_report.html"
METRICS_PATH = DOCS / "comparisons" / "acoustic_similarity_deep_metrics.tsv"
LOUDNESS_PATH = DOCS / "assets" / "report_audio_loudness_manifest.json"
HERO = "assets/tts_voice_clone_project_map_16x9_1920.png"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}


MODEL_ORDER = [
    "Original",
    "Fish Speech S2 Pro",
    "Irodori-TTS 500M",
    "Irodori-TTS 600M VoiceDesign",
    "Qwen3-TTS 1.7B",
    "VoxCPM2 ultimate",
    "CosyVoice2",
]


SAMPLE_LABELS = {
    "similarity_h4n_005": "005: 数字・固有名詞",
    "similarity_h4n_026": "026: 技術語",
    "similarity_h4n_051": "051: ナレーション",
    "similarity_h4n_087": "087: 文脈読み",
}


def rel_from_docs(path: Path) -> str:
    return Path("../").joinpath(path.relative_to(ROOT)).as_posix()


def file_size_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024 / 1024:.1f} MB"


def wav_duration(path: Path) -> str:
    if path.suffix.lower() != ".wav":
        return ""
    try:
        with wave.open(str(path), "rb") as wf:
            return f"{wf.getnframes() / wf.getframerate():.1f}s"
    except Exception:
        return ""


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


TOKEN_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)|\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*"
)


def inline(md: str) -> str:
    out: list[str] = []
    last = 0
    for match in TOKEN_RE.finditer(md):
        out.append(html.escape(md[last : match.start()]))
        if match.group(1) is not None:
            alt = html.escape(match.group(1))
            src = html.escape(match.group(2))
            out.append(
                f'<figure class="figure"><button class="image-button" type="button" data-full="{src}" aria-label="画像を拡大">'
                f'<img src="{src}" alt="{alt}" loading="lazy"></button>'
                f'<figcaption>{alt}</figcaption></figure>'
            )
        elif match.group(3) is not None:
            label = html.escape(match.group(3))
            href = html.escape(match.group(4))
            target = ' target="_blank" rel="noopener"' if href.startswith(("http://", "https://")) else ""
            out.append(f'<a href="{href}"{target}>{label}</a>')
        elif match.group(5) is not None:
            out.append(f"<code>{html.escape(match.group(5))}</code>")
        else:
            out.append(f"<strong>{html.escape(match.group(6))}</strong>")
        last = match.end()
    out.append(html.escape(md[last:]))
    return "".join(out)


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def slug_base(text_value: str, idx: int) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text_value)
    cleaned = re.sub(r"[^\w\-一-龯ぁ-んァ-ヶー]+", "-", cleaned).strip("-").lower()
    return cleaned or f"section-{idx}"


def render_markdown(md: str) -> tuple[str, list[dict[str, str]]]:
    lines = md.splitlines()
    stripped_lines: list[str] = []
    skipped_h1 = False
    skipped_first_image = False
    for line in lines:
        if not skipped_h1 and line.startswith("# "):
            skipped_h1 = True
            continue
        if skipped_h1 and not skipped_first_image and line.startswith("!["):
            skipped_first_image = True
            continue
        stripped_lines.append(line)

    rendered: list[str] = []
    toc: list[dict[str, str]] = []
    i = 0
    section_index = 0
    while i < len(stripped_lines):
        line = stripped_lines[i]
        if not line.strip():
            i += 1
            continue

        if line.startswith("```"):
            lang = html.escape(line[3:].strip())
            code_lines: list[str] = []
            i += 1
            while i < len(stripped_lines) and not stripped_lines[i].startswith("```"):
                code_lines.append(stripped_lines[i])
                i += 1
            i += 1
            rendered.append(
                f'<pre class="code-block"><code class="language-{lang}">{html.escape(chr(10).join(code_lines))}</code></pre>'
            )
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            label_md = heading.group(2).strip()
            label_plain = re.sub(r"`([^`]+)`", r"\1", label_md)
            section_index += 1
            slug = f"{slug_base(label_plain, section_index)}-{section_index}"
            if level <= 3:
                toc.append({"level": str(level), "label": label_plain, "id": slug})
            rendered.append(f'<h{level} id="{slug}">{inline(label_md)}</h{level}>')
            i += 1
            continue

        if line.strip().startswith("<details"):
            raw_lines = [line]
            i += 1
            while i < len(stripped_lines):
                raw_lines.append(stripped_lines[i])
                if stripped_lines[i].strip() == "</details>":
                    i += 1
                    break
                i += 1
            rendered.append("\n".join(raw_lines))
            continue

        if "|" in line and i + 1 < len(stripped_lines) and is_table_separator(stripped_lines[i + 1]):
            headers = split_table_row(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(stripped_lines) and "|" in stripped_lines[i] and stripped_lines[i].strip():
                rows.append(split_table_row(stripped_lines[i]))
                i += 1
            table = ['<div class="table-wrap"><table>']
            table.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in headers) + "</tr></thead>")
            table.append("<tbody>")
            for row in rows:
                table.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            table.append("</tbody></table></div>")
            rendered.append("".join(table))
            continue

        if line.startswith("- "):
            items: list[str] = []
            while i < len(stripped_lines) and stripped_lines[i].startswith("- "):
                items.append(stripped_lines[i][2:].strip())
                i += 1
            rendered.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>")
            continue

        if line.startswith("![") and line.rstrip().endswith(")"):
            rendered.append(inline(line.strip()))
            i += 1
            continue

        paragraph: list[str] = [line.strip()]
        i += 1
        while i < len(stripped_lines):
            nxt = stripped_lines[i]
            if not nxt.strip() or nxt.startswith(("```", "#", "- ")):
                break
            if "|" in nxt and i + 1 < len(stripped_lines) and is_table_separator(stripped_lines[i + 1]):
                break
            paragraph.append(nxt.strip())
            i += 1
        rendered.append(f"<p>{inline(' '.join(paragraph))}</p>")

    return "\n".join(rendered), toc


def audio_category(path: Path) -> tuple[str, str]:
    parts = path.relative_to(ROOT).parts
    path_text = path.as_posix().lower()
    if parts[0] == "samples":
        if "voice_refs" in parts:
            return "参照音声", "reference"
        if "h4n_round2_wav" in parts:
            return "録音チャンク", "source"
        return "入力サンプル", "sample"
    if "similarity_eval" in path_text:
        return "同一文生成", "comparison"
    if "h4n_presentation" in path_text:
        return "長文プレゼン", "presentation"
    if "irodori_ev" in path_text or "ev_launch" in path_text:
        return "EV発表会サンプル", "ev"
    return "生成音声", "output"


def model_from_path(path: Path) -> str:
    if path.relative_to(ROOT).parts[0] == "samples":
        return "Original"
    lowered = path.as_posix().lower()
    if "fish" in lowered:
        return "Fish Speech"
    if "irodori_600" in lowered or "voicedesign" in lowered:
        return "Irodori VoiceDesign"
    if "irodori" in lowered:
        return "Irodori 500M"
    if "qwen" in lowered:
        return "Qwen3-TTS"
    if "voxcpm" in lowered:
        return "VoxCPM2"
    if "cosyvoice" in lowered:
        return "CosyVoice2"
    if "gpt" in lowered:
        return "GPT-SoVITS"
    if "f5" in lowered:
        return "F5-TTS"
    if "aivis" in lowered:
        return "AIVIS"
    return "Other"


def load_loudness_manifest() -> dict:
    if not LOUDNESS_PATH.exists():
        return {"files": {}, "targetRmsDbfs": None, "peakCeilingDbfs": None}
    return json.loads(LOUDNESS_PATH.read_text(encoding="utf-8"))


def loudness_attrs(path: Path, manifest: dict) -> dict[str, str]:
    data = manifest.get("files", {}).get(path.relative_to(ROOT).as_posix(), {})
    gain = float(data.get("gain", 1.0))
    gain_db = float(data.get("gainDb", 0.0))
    mode = data.get("mode", "unmeasured")
    return {
        "gain": f"{gain:.6f}",
        "gainDb": f"{gain_db:+.1f} dB",
        "rmsDbfs": "" if data.get("rmsDbfs") is None else f"{data.get('rmsDbfs'):.1f} dBFS",
        "peakDbfs": "" if data.get("peakDbfs") is None else f"{data.get('peakDbfs'):.1f} dBFS",
        "loudnessMode": mode,
        "normalized": "yes" if mode == "rms_gain_peak_limited" else "no",
    }


def build_audio_library(loudness: dict) -> list[dict[str, str]]:
    files = sorted(
        p for base in [ROOT / "samples", ROOT / "outputs"] if base.exists() for p in base.rglob("*") if p.suffix.lower() in AUDIO_EXTS
    )
    rows: list[dict[str, str]] = []
    for path in files:
        category, category_key = audio_category(path)
        rows.append(
            {
                "name": path.name,
                "path": rel_from_docs(path),
                "displayPath": str(path.relative_to(ROOT)),
                "category": category,
                "categoryKey": category_key,
                "model": model_from_path(path),
                "size": file_size_mb(path),
                "duration": wav_duration(path),
                **loudness_attrs(path, loudness),
            }
        )
    return rows


def build_comparison_groups(loudness: dict) -> list[dict]:
    if not METRICS_PATH.exists():
        return []
    groups: dict[str, dict] = {}
    with METRICS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample = row["sample"]
            original = ROOT / row["original_wav"]
            generated = ROOT / row["generated_wav"]
            if sample not in groups:
                text_path = ROOT / "samples" / "text" / f"{sample}.txt"
                groups[sample] = {
                    "sample": sample,
                    "label": SAMPLE_LABELS.get(sample, sample),
                    "text": read_text(text_path),
                    "audio_items": [
                        {
                            "label": "Original",
                            "modelKey": "original",
                            "src": rel_from_docs(original),
                            "path": str(original.relative_to(ROOT)),
                            "metric": "",
                            "duration": wav_duration(original),
                            **loudness_attrs(original, loudness),
                        }
                    ],
                }
            if generated.exists():
                groups[sample]["audio_items"].append(
                    {
                        "label": row["model_label"],
                        "modelKey": re.sub(r"[^a-z0-9]+", "-", row["model_label"].lower()).strip("-"),
                        "src": rel_from_docs(generated),
                        "path": str(generated.relative_to(ROOT)),
                        "metric": f"Composite {float(row['composite_similarity']):.3f}",
                        "duration": row["generated_duration_sec"] + "s",
                        **loudness_attrs(generated, loudness),
                    }
                )
    ordered_groups = []
    for sample in SAMPLE_LABELS:
        if sample in groups:
            groups[sample]["audio_items"].sort(
                key=lambda item: MODEL_ORDER.index(item["label"]) if item["label"] in MODEL_ORDER else 99
            )
            ordered_groups.append(groups[sample])
    return ordered_groups


def build_reference_clips(library: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred = [
        "h4n_ref10s_neutral_091_093.wav",
        "h4n_ref20s_neutral_091_094.wav",
        "h4n_ref38s_japanese_edge_081_087.wav",
    ]
    lookup = {row["name"]: row for row in library}
    return [lookup[name] for name in preferred if name in lookup]


def build_long_samples(library: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = [
        row
        for row in library
        if row["categoryKey"] in {"presentation", "ev"}
        and ("full" in row["name"].lower() or "concat" in row["name"].lower() or "presentation" in row["name"].lower())
    ]
    return candidates[:24]


def render_inline_audio_panel(groups: list[dict], loudness: dict) -> str:
    template = Template(
        r"""
<section class="audio-inline" id="inline-audio-comparison">
  <div class="section-kicker">Listening Samples</div>
  <h3>試聴サンプル: 同一文で比較</h3>
  <p>元のH4n Pro録音と、各モデルが同じ文章を読んだ音声を並べた。試聴時はRMS {{ target }} dBFSを目安に、ピークが割れない範囲で線形ゲインだけをかけている。元ファイルは変更していない。</p>

  <div class="filter-bar" aria-label="主要比較フィルタ">
    <button class="chip active" type="button" data-filter-sample="all">全サンプル</button>
    {% for group in groups %}
    <button class="chip" type="button" data-filter-sample="{{ group.sample }}">{{ group.label }}</button>
    {% endfor %}
  </div>

  <div class="comparison-grid">
    {% for group in groups %}
    <div class="sample-card" data-sample="{{ group.sample }}">
      <div class="sample-card__head">
        <h3>{{ group.label }}</h3>
        <span>{{ group.sample }}</span>
      </div>
      <p class="sample-text">{{ group.text }}</p>
      <div class="audio-rows">
        {% for item in group.audio_items %}
        <div class="audio-row" data-model="{{ item.modelKey }}">
          <div class="audio-row__meta">
            <strong>{{ item.label }}</strong>
            <span>{{ item.metric or item.duration }} / {{ item.gainDb }}</span>
          </div>
          <audio controls preload="none" src="{{ item.src }}" data-gain="{{ item.gain }}" data-gain-db="{{ item.gainDb }}"></audio>
          <code>{{ item.path }}</code>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>
</section>
"""
    )
    return template.render(groups=groups, target=loudness.get("targetRmsDbfs", -20.0))


def render_audio_appendix(refs: list[dict], long_samples: list[dict], library: list[dict], loudness: dict) -> str:
    template = Template(
        r"""
<section class="audio-appendix" id="audio-reference-library">
  <div class="section-kicker">Appendix</div>
  <h2>音声リファレンス</h2>
  <p>記事本文の試聴以外に、参照音声、長文サンプル、生成済み音声を検索できる付録として残した。ここも測定済みWAVは再生時ゲインで音量をそろえる。</p>

  <div class="audio-columns">
    <section>
      <h3>参照音声</h3>
      {% for row in refs %}
      <div class="compact-audio">
        <div><strong>{{ row.name }}</strong><span>{{ row.duration }} / {{ row.size }} / {{ row.gainDb }}</span></div>
        <audio controls preload="none" src="{{ row.path }}" data-gain="{{ row.gain }}" data-gain-db="{{ row.gainDb }}"></audio>
      </div>
      {% endfor %}
    </section>
    <section>
      <h3>長文・プレゼン系サンプル</h3>
      {% for row in long_samples %}
      <div class="compact-audio">
        <div><strong>{{ row.name }}</strong><span>{{ row.model }} / {{ row.duration or row.size }} / {{ row.gainDb }}</span></div>
        <audio controls preload="none" src="{{ row.path }}" data-gain="{{ row.gain }}" data-gain-db="{{ row.gainDb }}"></audio>
      </div>
      {% endfor %}
    </section>
  </div>

  <section class="library-panel">
    <div class="library-head">
      <div>
        <h3>全音声ライブラリ</h3>
        <p>{{ library|length }}件。検索、分類、モデルで絞り込み、ページ単位でプレイヤーを表示する。測定対象は {{ measured }}件。</p>
      </div>
      <div class="library-controls">
        <input id="audioSearch" type="search" placeholder="ファイル名・パスで検索">
        <select id="categoryFilter" aria-label="分類で絞り込み">
          <option value="all">すべての分類</option>
        </select>
        <select id="modelFilter" aria-label="モデルで絞り込み">
          <option value="all">すべてのモデル</option>
        </select>
      </div>
    </div>
    <div id="libraryStats" class="library-stats"></div>
    <div id="audioLibrary" class="library-list"></div>
    <div class="pager">
      <button type="button" id="prevPage">前へ</button>
      <span id="pageState"></span>
      <button type="button" id="nextPage">次へ</button>
    </div>
  </section>
</section>
"""
    )
    measured = sum(1 for row in library if row.get("normalized") == "yes")
    return template.render(refs=refs, long_samples=long_samples, library=library, measured=measured)


def inject_after_heading(article_html: str, toc: list[dict[str, str]], label: str, block: str) -> str:
    target = next((item for item in toc if item["label"] == label), None)
    if not target:
        return block + "\n" + article_html
    pattern = re.compile(rf'(<h[1-6] id="{re.escape(target["id"])}">.*?</h[1-6]>)')
    replaced, count = pattern.subn(lambda match: match.group(1) + "\n" + block, article_html, count=1)
    return replaced if count else block + "\n" + article_html


HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ローカルPCで自分の声の日本語TTSを作る</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --paper: #ffffff;
      --ink: #172033;
      --muted: #5b667a;
      --line: #dfe7f1;
      --navy: #102038;
      --teal: #109c95;
      --blue: #367fd6;
      --coral: #ef675c;
      --yellow: #f6bd3f;
      --purple: #7f64d5;
      --green: #51ad67;
      --shadow: 0 18px 48px rgba(18, 31, 52, .11);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Yu Gothic", "Meiryo", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: linear-gradient(180deg, #eef5fb 0, var(--bg) 34rem);
      line-height: 1.82;
      letter-spacing: 0;
    }
    .progress { position: fixed; inset: 0 0 auto 0; height: 4px; background: transparent; z-index: 20; }
    .progress__bar { width: 0; height: 100%; background: linear-gradient(90deg, var(--teal), var(--blue), var(--coral)); }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    a { color: #1764bd; text-decoration-thickness: .08em; text-underline-offset: .18em; }
    code { padding: .12rem .34rem; border-radius: 6px; background: #eef3f8; color: #263d5d; font-family: "Cascadia Mono", Consolas, monospace; font-size: .92em; }
    pre.code-block {
      overflow-x: auto;
      padding: 1rem 1.1rem;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #0f1b2e;
      color: #e9f0fa;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
    }
    pre.code-block code { padding: 0; background: transparent; color: inherit; }
    .hero {
      max-width: 1180px;
      margin: 0 auto;
      padding: 42px 24px 18px;
    }
    .hero h1 {
      margin: 0;
      color: var(--navy);
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 1.18;
      font-weight: 800;
      letter-spacing: 0;
    }
    .hero .lead {
      max-width: 820px;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.85;
    }
    .hero-cover {
      margin: 26px 0 0;
      max-width: 900px;
    }
    .hero-cover img {
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid rgba(16, 32, 56, .12);
      background: #fff;
    }
    .hero-cover figcaption { margin-top: .5rem; color: var(--muted); font-size: .88rem; }
    .layout {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 24px 72px;
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      gap: 32px;
      align-items: start;
    }
    .toc {
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow: auto;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,.86);
      backdrop-filter: blur(10px);
    }
    .toc h2 {
      margin: 0 0 12px;
      font-size: .9rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .toc a {
      display: block;
      padding: 7px 0;
      color: var(--ink);
      text-decoration: none;
      font-size: .92rem;
      line-height: 1.45;
      border-bottom: 1px solid rgba(223,231,241,.7);
    }
    .toc a.level-3 { padding-left: 14px; color: var(--muted); font-size: .86rem; }
    .article, .audio-appendix {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--paper);
      box-shadow: 0 8px 28px rgba(18, 31, 52, .07);
    }
    .article { padding: 34px; }
    .article h2, .audio-appendix h2 {
      margin: 2.6rem 0 1rem;
      color: var(--navy);
      font-size: 1.72rem;
      line-height: 1.35;
    }
    .article h2:first-child { margin-top: 0; }
    .article h3 { margin: 2rem 0 .7rem; color: var(--navy); font-size: 1.28rem; }
    .article h4 { margin: 1.4rem 0 .4rem; color: var(--ink); font-size: 1.06rem; }
    .article p, .article li { font-size: 1rem; }
    .article ul { padding-left: 1.4rem; }
    .table-wrap { overflow-x: auto; margin: 1.15rem 0 1.6rem; border: 1px solid var(--line); border-radius: 12px; }
    table { width: 100%; border-collapse: collapse; min-width: 620px; background: #fff; }
    th, td { padding: .72rem .84rem; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }
    th { background: #f1f5fa; color: #233957; font-weight: 700; }
    tr:last-child td { border-bottom: 0; }
    .figure { margin: 1.6rem 0; }
    .image-button { display: block; width: 100%; padding: 0; border: 0; background: transparent; cursor: zoom-in; }
    .figure img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .figure figcaption { margin-top: .45rem; color: var(--muted); font-size: .9rem; }
    .section-kicker {
      display: inline-flex;
      margin-bottom: -1rem;
      color: var(--teal);
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
      font-size: .78rem;
    }
    .audio-appendix {
      grid-column: 2;
      padding: 30px;
      margin-top: 28px;
    }
    .audio-appendix h2 { margin-top: .2rem; }
    .audio-inline {
      margin: 1.6rem 0 2.2rem;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fbfdff;
    }
    .audio-inline h3 {
      margin-top: .2rem;
      color: var(--navy);
    }
    .filter-bar { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 22px; }
    button, input, select { font: inherit; }
    .chip, .pager button {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: .42rem .8rem;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    .chip.active { background: var(--navy); color: #fff; border-color: var(--navy); }
    .comparison-grid { display: grid; gap: 18px; }
    .sample-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      background: #fbfdff;
    }
    .sample-card__head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
    }
    .sample-card h3 { margin: 0; font-size: 1.12rem; }
    .sample-card__head span { color: var(--muted); font-size: .9rem; }
    .sample-text { margin: .7rem 0 1rem; padding: .8rem 1rem; border-left: 4px solid var(--teal); background: #eefaf8; border-radius: 0 8px 8px 0; }
    .audio-rows { display: grid; gap: 10px; }
    .audio-row {
      display: grid;
      grid-template-columns: minmax(150px, 220px) minmax(220px, 1fr);
      gap: 12px;
      align-items: center;
      padding: 12px;
      border: 1px solid #e5edf6;
      border-radius: 12px;
      background: #fff;
    }
    .audio-row audio, .compact-audio audio { width: 100%; }
    .audio-row code { grid-column: 2; overflow-wrap: anywhere; color: var(--muted); }
    .audio-row__meta strong, .compact-audio strong { display: block; line-height: 1.35; }
    .audio-row__meta span, .compact-audio span { display: block; color: var(--muted); font-size: .85rem; }
    .audio-columns {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 24px;
    }
    .audio-columns section, .library-panel {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      background: #fff;
    }
    .audio-columns h3, .library-panel h3 { margin: 0 0 12px; }
    .compact-audio { display: grid; gap: 8px; padding: 12px 0; border-top: 1px solid var(--line); }
    .library-panel { margin-top: 24px; }
    .library-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 420px);
      gap: 18px;
      align-items: start;
    }
    .library-head p { margin: .2rem 0 0; color: var(--muted); }
    .library-controls { display: grid; gap: 10px; }
    .library-controls input, .library-controls select {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: .45rem .7rem;
      background: #fff;
    }
    .library-stats { margin: 16px 0 10px; color: var(--muted); font-size: .92rem; }
    .library-list { display: grid; gap: 10px; }
    .library-item {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(240px, .9fr);
      gap: 12px;
      align-items: center;
      padding: 12px;
      border: 1px solid #e6edf7;
      border-radius: 12px;
      background: #fbfdff;
    }
    .library-item code { display: block; margin-top: 4px; overflow-wrap: anywhere; }
    .library-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 7px; }
    .badge { padding: .16rem .45rem; border-radius: 999px; background: #edf4fb; color: #37506f; font-size: .78rem; }
    .pager { display: flex; justify-content: center; align-items: center; gap: 14px; margin-top: 18px; }
    .glossary {
      margin: 0 0 28px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }
    .glossary summary { cursor: pointer; font-weight: 700; color: var(--navy); }
    .glossary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
    .term { padding: 12px; border: 1px solid #e5edf6; border-radius: 12px; background: #fbfdff; }
    .term strong { display: block; margin-bottom: 4px; color: var(--navy); }
    .recording-script {
      margin: 1.2rem 0 1.8rem;
      padding: 16px 18px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fbfdff;
    }
    .recording-script summary {
      cursor: pointer;
      color: var(--navy);
      font-weight: 700;
    }
    .script-list {
      margin: 16px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 8px;
    }
    .script-list li {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      padding: 8px 10px;
      border: 1px solid #e8eef6;
      border-radius: 10px;
      background: #fff;
      font-size: .94rem;
      line-height: 1.65;
    }
    .script-list code { text-align: center; }
    .lightbox {
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      padding: 28px;
      background: rgba(6, 14, 28, .82);
      z-index: 40;
    }
    .lightbox.open { display: grid; }
    .lightbox img { max-width: min(1280px, 96vw); max-height: 88vh; border-radius: 10px; background: #fff; }
    .lightbox button {
      position: fixed;
      top: 18px;
      right: 18px;
      border: 1px solid rgba(255,255,255,.35);
      background: rgba(255,255,255,.12);
      color: #fff;
      border-radius: 999px;
      padding: .5rem .9rem;
      cursor: pointer;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .toc { position: static; max-height: none; }
      .audio-appendix { grid-column: auto; }
      .audio-columns, .library-head, .glossary-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .hero { padding: 22px 14px; }
      .layout { padding: 0 14px 44px; gap: 18px; }
      .article, .audio-appendix { padding: 20px; }
      .audio-row, .library-item { grid-template-columns: 1fr; }
      .audio-row code { grid-column: auto; }
      .article h2, .audio-appendix h2 { font-size: 1.42rem; }
    }
  </style>
</head>
<body>
  <div class="progress" aria-hidden="true"><div class="progress__bar" id="progressBar"></div></div>
  <header class="hero">
    <h1>ローカルPCで自分の声の日本語TTSを作る</h1>
    <p class="lead">録音、モデル比較、音響分析、試聴サンプルまでをひとつにまとめた検証レポート。本文を読みながら、必要な箇所で元音声と生成音声を比較できる。</p>
    <figure class="hero-cover">
      <img src="{{ hero }}" alt="ローカル日本語TTS / Voice Clone検証 プロジェクト全体図">
      <figcaption>ローカル日本語TTS / Voice Clone検証のプロジェクト全体図</figcaption>
    </figure>
  </header>
  <main class="layout">
    <aside class="toc" aria-label="目次">
      <h2>Contents</h2>
      <a href="#inline-audio-comparison">試聴サンプル</a>
      <a href="#audio-reference-library">音声リファレンス</a>
      {% for item in toc %}
      <a class="level-{{ item.level }}" href="#{{ item.id }}">{{ item.label }}</a>
      {% endfor %}
    </aside>
    <article class="article">
      <details class="glossary" open>
        <summary>指標の読み方ミニグロッサリ</summary>
        <div class="glossary-grid">
          <div class="term"><strong>F0</strong>声帯振動に対応する基本周波数。中央値や輪郭が近いほど、話者の高さや抑揚が近いと見なせる。</div>
          <div class="term"><strong>フォルマント</strong>母音の響き方を示す共鳴周波数。F1/F2/F3の距離が小さいほど、母音の声質が近い。</div>
          <div class="term"><strong>スペクトログラム</strong>時間ごとの周波数成分の分布。濃淡パターンが似ているほど、音色や発音の構造が近い。</div>
          <div class="term"><strong>DTW</strong>Dynamic Time Warping。話速の違いを吸収して系列を対応付ける。距離は小さいほど近い。</div>
          <div class="term"><strong>ASR</strong>自動音声認識。文字起こしが元文に近いほど、内容保持と発音明瞭性が高い。</div>
          <div class="term"><strong>Composite</strong>複数指標を正規化して合成した類似度。本レポートでは1に近いほど元音声に近い。</div>
        </div>
      </details>
      {{ article_html }}
    </article>
    {{ audio_appendix }}
  </main>
  <div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="拡大画像">
    <button type="button" id="closeLightbox">閉じる</button>
    <img alt="">
  </div>
  <script>
    const AUDIO_LIBRARY = {{ audio_library_json }};
    const pageSize = 24;
    let page = 0;
    const search = document.getElementById('audioSearch');
    const categoryFilter = document.getElementById('categoryFilter');
    const modelFilter = document.getElementById('modelFilter');
    const list = document.getElementById('audioLibrary');
    const stats = document.getElementById('libraryStats');
    const pageState = document.getElementById('pageState');

    function uniqueValues(key) {
      return [...new Set(AUDIO_LIBRARY.map(item => item[key]).filter(Boolean))].sort((a, b) => a.localeCompare(b, 'ja'));
    }
    for (const value of uniqueValues('category')) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      categoryFilter.appendChild(option);
    }
    for (const value of uniqueValues('model')) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      modelFilter.appendChild(option);
    }
    function filteredLibrary() {
      const q = search.value.trim().toLowerCase();
      return AUDIO_LIBRARY.filter(item => {
        const text = `${item.name} ${item.displayPath} ${item.model} ${item.category}`.toLowerCase();
        return (!q || text.includes(q)) &&
          (categoryFilter.value === 'all' || item.category === categoryFilter.value) &&
          (modelFilter.value === 'all' || item.model === modelFilter.value);
      });
    }
    function renderLibrary() {
      const items = filteredLibrary();
      const pages = Math.max(1, Math.ceil(items.length / pageSize));
      if (page >= pages) page = pages - 1;
      const start = page * pageSize;
      const visible = items.slice(start, start + pageSize);
      stats.textContent = `${items.length}件を表示対象にしています`;
      pageState.textContent = `${page + 1} / ${pages}`;
      list.innerHTML = visible.map(item => `
        <div class="library-item">
          <div>
            <strong>${escapeHtml(item.name)}</strong>
            <code>${escapeHtml(item.displayPath)}</code>
            <div class="library-badges">
              <span class="badge">${escapeHtml(item.category)}</span>
              <span class="badge">${escapeHtml(item.model)}</span>
              <span class="badge">${escapeHtml(item.duration || item.size)}</span>
              <span class="badge">gain ${escapeHtml(item.gainDb || '+0.0 dB')}</span>
            </div>
          </div>
          <audio controls preload="none" src="${item.path}" data-gain="${item.gain || '1.000000'}" data-gain-db="${escapeHtml(item.gainDb || '+0.0 dB')}"></audio>
        </div>
      `).join('');
      hydrateAudioLabels();
    }
    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
    }
    search.addEventListener('input', () => { page = 0; renderLibrary(); });
    categoryFilter.addEventListener('change', () => { page = 0; renderLibrary(); });
    modelFilter.addEventListener('change', () => { page = 0; renderLibrary(); });
    document.getElementById('prevPage').addEventListener('click', () => { page = Math.max(0, page - 1); renderLibrary(); });
    document.getElementById('nextPage').addEventListener('click', () => { page += 1; renderLibrary(); });
    renderLibrary();

    let audioContext;
    const audioGraph = new WeakMap();
    function hydrateAudioLabels() {
      document.querySelectorAll('audio[data-gain]').forEach(audio => {
        audio.title = `再生時ゲイン ${audio.dataset.gainDb || '+0.0 dB'}`;
      });
    }
    hydrateAudioLabels();
    async function applyPlaybackGain(audio) {
      const gain = Number(audio.dataset.gain || '1');
      if (!Number.isFinite(gain) || Math.abs(gain - 1) < 0.001) return;
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error('AudioContext unavailable');
        if (!audioContext) audioContext = new AudioContextClass();
        if (audioContext.state === 'suspended') await audioContext.resume();
        let graph = audioGraph.get(audio);
        if (!graph) {
          const source = audioContext.createMediaElementSource(audio);
          const gainNode = audioContext.createGain();
          source.connect(gainNode).connect(audioContext.destination);
          graph = { gainNode };
          audioGraph.set(audio, graph);
        }
        graph.gainNode.gain.value = gain;
      } catch (error) {
        audio.volume = Math.max(0, Math.min(1, gain));
      }
    }
    document.addEventListener('play', event => {
      if (event.target instanceof HTMLAudioElement) {
        applyPlaybackGain(event.target);
      }
    }, true);

    for (const button of document.querySelectorAll('[data-filter-sample]')) {
      button.addEventListener('click', () => {
        document.querySelectorAll('[data-filter-sample]').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        const sample = button.dataset.filterSample;
        document.querySelectorAll('.sample-card').forEach(card => {
          card.hidden = sample !== 'all' && card.dataset.sample !== sample;
        });
      });
    }

    const progressBar = document.getElementById('progressBar');
    window.addEventListener('scroll', () => {
      const max = document.documentElement.scrollHeight - innerHeight;
      progressBar.style.width = `${max > 0 ? (scrollY / max) * 100 : 0}%`;
    }, { passive: true });

    const lightbox = document.getElementById('lightbox');
    const lightboxImage = lightbox.querySelector('img');
    document.querySelectorAll('.image-button').forEach(button => {
      button.addEventListener('click', () => {
        lightboxImage.src = button.dataset.full;
        lightboxImage.alt = button.querySelector('img')?.alt || '';
        lightbox.classList.add('open');
      });
    });
    document.getElementById('closeLightbox').addEventListener('click', () => lightbox.classList.remove('open'));
    lightbox.addEventListener('click', event => {
      if (event.target === lightbox) lightbox.classList.remove('open');
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') lightbox.classList.remove('open');
    });
  </script>
</body>
</html>
"""
)


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    article_html, toc = render_markdown(md)
    loudness = load_loudness_manifest()
    library = build_audio_library(loudness)
    comparison_groups = build_comparison_groups(loudness)
    refs = build_reference_clips(library)
    long_samples = build_long_samples(library)
    inline_audio = render_inline_audio_panel(comparison_groups, loudness)
    article_html = inject_after_heading(article_html, toc, "同一文による音響比較", inline_audio)
    audio_appendix = render_audio_appendix(refs, long_samples, library, loudness)
    html_doc = HTML_TEMPLATE.render(
        hero=HERO,
        toc=toc,
        audio_appendix=audio_appendix,
        article_html=article_html,
        audio_library_json=json.dumps(library, ensure_ascii=False),
    )
    html_doc = "\n".join(line.rstrip() for line in html_doc.splitlines()) + "\n"
    OUT_PATH.write_text(html_doc, encoding="utf-8", newline="\n")
    print(f"wrote={OUT_PATH}")
    print(f"toc={len(toc)}")
    print(f"comparison_groups={len(comparison_groups)}")
    print(f"audio_files={len(library)}")
    print(f"audio_loudness_measured={sum(1 for row in library if row.get('normalized') == 'yes')}")
    print(f"reference_clips={len(refs)}")
    print(f"long_samples={len(long_samples)}")


if __name__ == "__main__":
    main()
