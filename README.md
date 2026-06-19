# Local Japanese TTS Voice Clone Report

This repository is a public artifact bundle for a personal Japanese TTS / voice clone experiment.
It accompanies the Zenn article and GitHub Pages audio sample page.

## Canonical Links

- Zenn article: https://zenn.dev/fujinumagic/articles/local-japanese-tts-voice-clone
- GitHub Pages audio samples: https://jacky-fuji.github.io/mytts-public/
- Public repository: https://github.com/jacky-fuji/mytts-public

## Contents

- `articles/local-japanese-tts-voice-clone.md`: exported copy of the published Zenn article.
- `images/local-japanese-tts-voice-clone/`: images referenced from the Zenn article.
- `docs/index.html`: GitHub Pages audio sample page for same-text comparisons and curated public examples.
- `docs/audio/`: selected generated audio samples, loudness-normalized and encoded as MP3 for web listening.
- `docs/audio_manifest.csv`: public audio manifest. Source WAV paths are listed for reproducibility, but the WAV files themselves are not included.
- `data/`: recording script and selected evaluation metrics.
- `scripts/`: selected analysis/report helper scripts.

## Latest Summary

- The Zenn article topics are tuned for discovery around `tts`, `音声合成`, `ai`, `生成ai`, and `ローカルai`.
- Fish Speech S2 Pro had the strongest subjective naturalness, but the same 3-sentence timing benchmark measured 2994.0 seconds for 24.1 seconds of audio.
- Irodori-TTS 600M VoiceDesign remained the best speed/quality balance for local Japanese generation in this experiment.
- Public audio files are generated samples only; raw speaker recordings are excluded.

## Important Notice

This is a personal experiment, not a peer-reviewed benchmark or forensic voice-identification report.
The raw reference recordings and full generated output set are intentionally excluded.

この公開版では、プライバシー保護のため本人の元録音は含めていません。音声サンプルは公開用に選抜した生成音声を、元WAVからMP3へエンコードし、試聴しやすい音量へ正規化したものです。

## Suggested Publishing Flow

1. Review every file in this repository before publishing.
2. Push this repository to GitHub.
3. Enable GitHub Pages from the repository's `docs/` directory.
4. Manage the canonical Zenn article in the private Zenn content repository.
5. Keep the Zenn article, Pages URL, and public repository link synchronized.

## Licenses

See `LICENSE` for scripts and `NOTICE.md` for article, image, and audio handling.
