# Reproducibility Notes

This document records the local conditions used for the public Japanese TTS / voice clone comparison. It is not a recipe for exact reproduction. The raw speaker recordings and the full generated WAV set are intentionally excluded for privacy and storage reasons.

Snapshot date: 2026-06-19 JST

## Hardware And Driver

| Item | Value |
|---|---|
| OS | Microsoft Windows 11 Home, build family `10.0.26200`; `ver` returned `10.0.26200.8655` |
| CPU | AMD Ryzen 7 7700, 8 cores / 16 threads |
| RAM | about 64 GB |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16,311 MiB reported by `nvidia-smi` |
| NVIDIA Driver | 596.36 |
| CUDA reported by `nvidia-smi` | 13.2 |
| PyTorch CUDA runtime used by most TTS environments | CUDA 12.8 wheels (`+cu128`) |

The important distinction is that `nvidia-smi` reports the driver-supported CUDA version, while PyTorch reports the CUDA runtime that its wheel was built against. In this run, the driver reported CUDA 13.2, but the Python environments used CUDA 12.8 PyTorch wheels.

## Python Environments

The comparison used separate virtual environments because the TTS projects have incompatible dependency ranges.

| Environment | Python | PyTorch | Torch CUDA | Transformers | Notes |
|---|---|---:|---:|---:|---|
| `tools/Irodori-TTS/.venv` | 3.10.20 | 2.10.0+cu128 | 12.8 | 4.57.6 | Irodori-TTS 500M / 600M VoiceDesign |
| `tools/qwen3-tts/.venv` | 3.12.11 | 2.8.0+cu128 | 12.8 | 4.57.3 | `qwen-tts` 0.1.1 |
| `tools/fish-speech/.venv` | 3.12.11 | 2.8.0+cu128 | 12.8 | 4.56.1 | Fish Speech S2 Pro |
| `tools/CosyVoice/.venv` | 3.10.20 | 2.8.0+cu128 | 12.8 | 4.51.3 | CosyVoice2 |
| `tools/voxcpm/.venv` | 3.12.11 | 2.8.0+cu128 | 12.8 | 5.12.1 | VoxCPM2 and most analysis scripts |
| `tools/Style-Bert-VITS2/venv` | 3.10.20 | 2.11.0+cu128 | 12.8 | 4.57.6 | Dataset preparation reached; custom training was not completed |
| `tools/F5-TTS/.venv` | 3.11.15 | 2.8.0+cu128 | 12.8 | 5.12.0 | Rejected for Japanese output quality |

The helper script used to collect these values is `scripts/collect_reproducibility_info.py`.

## Model Inventory

The machine-readable inventory is stored in `data/model_inventory.csv`. The most relevant entries are summarized below.

| Model | Model ID / source | Params | Local revision or snapshot | Runtime / tool |
|---|---|---:|---|---|
| Irodori-TTS 500M | `Aratako/Irodori-TTS-500M-v3` | 500M | `236c1e56591279fc24e3c1bf6609fc06e48dde28` | `tools/Irodori-TTS` commit `eaf74d6a19138f743acb5b71a445fd25a57db987` |
| Irodori-TTS 600M VoiceDesign | `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | 600M | `e863a3a93e652e09afeff3e84823a206a0a60314` | `tools/Irodori-TTS` commit `eaf74d6a19138f743acb5b71a445fd25a57db987` |
| Qwen3-TTS 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `fd4b254389122332181a7c3db7f27e918eec64e3` | `qwen-tts` 0.1.1 |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | not locally verified | `refs/main` = `1de9996b6be38b745688de084d87a5633f714e4e` | `tools/fish-speech` commit `e5e292632cb11e7a27b2b7487f58f612bc101e13` |
| CosyVoice2 | `FunAudioLLM/CosyVoice2-0.5B` | 0.5B | `refs/main` = `eec1ae6c79877dbd9379285cf8789c9e0879293d` | `tools/CosyVoice` commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` |
| VoxCPM2 | `openbmb/VoxCPM2` | 2B | `bffb3df5a29440629464e5e839f4d214c8714c3d` | `tools/voxcpm` Python environment |

For Fish Speech and CosyVoice2, the local Hugging Face cache had `refs/main` values but no `snapshots/` directory at inspection time. Therefore this document records the ref hash and marks the snapshot directory as not present.

## Reference Audio And Prompt Conditions

The final evaluation used Zoom H4n Pro recordings at 48 kHz / 24 bit. The original long recordings and sliced raw voice chunks are not part of the public repository.

| Reference profile | Duration | Used for |
|---|---:|---|
| `h4n_ref10s_neutral_091_093.wav` | 9.460s | Fish Speech S2 Pro |
| `h4n_ref20s_neutral_091_094.wav` | 21.030s | Irodori-TTS, Qwen3-TTS, CosyVoice2, VoxCPM2 |
| `h4n_ref38s_japanese_edge_081_087.wav` | 37.730s | pronunciation / reading checks |

The similarity evaluation used four target sentences that were not part of the reference prompt. The generation-time benchmark used three separate sentences: one short sentence, one narration sentence, and one technical-term sentence.

## Generation Settings

| Model | Main settings used in the final comparison |
|---|---|
| Irodori-TTS 500M | zero-shot reference audio; `num_steps=24`; `model_precision=bf16`; model and codec on CUDA |
| Irodori-TTS 600M VoiceDesign | same as 500M plus VoiceDesign caption; `num_steps=24`; `model_precision=bf16`; model and codec on CUDA |
| Qwen3-TTS 1.7B | `device=cuda:0`; `dtype=bfloat16`; `temperature=0.65`; `top_p=0.75`; `max_new_tokens` varied by sample, usually 192 / 224 / 256 |
| Fish Speech S2 Pro | prompt audio tokenized with DAC; text2semantic inference plus decode; CUDA device; `max_new_tokens` read from the target manifest |
| CosyVoice2 | `inference_cross_lingual`; H4n prompt text and prompt wav supplied; `speed=1.0`; `fp16` was used in some sweeps |
| VoxCPM2 ultimate | `mode=ultimate`; `cfg_value=2.0`; `inference_timesteps=10`; prompt text supplied |

No final six-model comparison entry used a custom speaker fine-tune or LoRA. The voice was brought closer by reference conditioning, zero-shot voice cloning, VoiceDesign, or equivalent prompt/reference mechanisms.

## Evaluation Inputs And Scripts

| Public file | Purpose |
|---|---|
| `data/recording_script_ja_100.txt` | 100-sentence Japanese recording script |
| `data/similarity_eval_targets.tsv` | Same-text acoustic comparison targets |
| `data/generation_time_benchmark_targets.tsv` | Three fair generation-time benchmark sentences |
| `data/metrics/acoustic_similarity_deep_metrics.tsv` | Frame/DTW, timbre, prosody, formant, duration metrics |
| `data/metrics/generation_time_benchmark_results.tsv` | Per-sample generation-time measurements |
| `data/metrics/generation_time_benchmark_summary.tsv` | Aggregated RTF summary |
| `data/metrics/phonetic_micro_summary.tsv` | Phone/mora-level aggregate metrics |
| `scripts/analyze_acoustic_similarity_deep.py` | Main acoustic comparison script |
| `scripts/analyze_phonetic_micro_similarity.py` | Phone/mora-level analysis script |
| `scripts/transcribe_generated_outputs.py` | ASR-based content-retention check |
| `scripts/create_generation_time_chart.py` | Generation-time chart generation |

## Known Limits

The evaluation is intentionally narrow. It is an `n=1` personal voice experiment, and subjective listening was done by the speaker. Model revisions, dependency versions, GPU drivers, prompt text, reference audio length, and text normalization can all change the result. Treat the tables as a reproducibility aid, not as a general benchmark.
