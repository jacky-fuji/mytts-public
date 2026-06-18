# Local Japanese TTS Voice Clone Report

This repository is a public artifact bundle for a personal Japanese TTS / voice clone experiment.

## Contents

- `articles/local-japanese-tts-voice-clone.md`: Zenn article draft. `published` is set to `false` by default.
- `images/local-japanese-tts-voice-clone/`: images referenced from the Zenn article.
- `docs/index.html`: GitHub Pages audio sample page.
- `docs/audio/`: selected generated audio samples, normalized and encoded as MP3 for web listening.
- `data/`: recording script and selected evaluation metrics.
- `scripts/`: selected analysis/report helper scripts.

## Important Notice

This is a personal experiment, not a peer-reviewed benchmark or forensic voice-identification report.
The raw reference recordings and full generated output set are intentionally excluded.

この公開版では、プライバシー保護のため本人の元録音は含めていません。音声サンプルは公開用に選抜した生成音声をMP3化し、試聴しやすい音量へ正規化したものです。

## Suggested Publishing Flow

1. Review every file in this repository before publishing.
2. Create a public GitHub repository and push this directory.
3. Enable GitHub Pages from the repository's `docs/` directory.
4. Connect the same repository to Zenn, or copy `articles/local-japanese-tts-voice-clone.md` into a Zenn-managed repository.
5. After Pages is live, add the Pages URL to the Zenn article.

## Licenses

See `LICENSE` for scripts and `NOTICE.md` for article, image, and audio handling.
