from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


PACKAGES = [
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "praat-parselmouth",
    "openai-whisper",
    "faster-whisper",
    "funasr",
    "modelscope",
    "qwen-tts",
]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"import_error": repr(exc)}

    info: dict[str, object] = {
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name_0"] = torch.cuda.get_device_name(0)
    return info


def git_head(path: Path) -> str | None:
    git_dir = path / ".git"
    if not git_dir.exists():
        return None
    try:
        return subprocess.check_output(
            [
                "git",
                f"--git-dir={git_dir}",
                f"--work-tree={path}",
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = {
        "executable": sys.executable,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "torch": torch_info(),
        "tool_git_heads": {
            "Irodori-TTS": git_head(root / "tools" / "Irodori-TTS"),
            "fish-speech": git_head(root / "tools" / "fish-speech"),
            "CosyVoice": git_head(root / "tools" / "CosyVoice"),
            "F5-TTS": git_head(root / "tools" / "F5-TTS"),
            "GPT-SoVITS": git_head(root / "tools" / "GPT-SoVITS"),
            "Style-Bert-VITS2": git_head(root / "tools" / "Style-Bert-VITS2"),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
