from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description="Peak-normalize WAV files into a separate directory.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--peak-dbfs", type=float, default=-3.0)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        raise ValueError(f"No files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_peak = 10 ** (args.peak_dbfs / 20)

    for path in paths:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        normalized = data if peak <= 0 else data / peak * target_peak
        output_path = output_dir / path.name
        sf.write(str(output_path), normalized, sr)
        print(f"{output_path}\t{20 * math.log10(max(target_peak, 1e-12)):.2f} dBFS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
