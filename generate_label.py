import subprocess
import re
import os
import argparse
from pathlib import Path

ASR_BINARY = "./asr/asr_func"
ASR_CONFIG = "asr/asr-decoder-zh_CN-digital-8.cfg.0.5.2.0"
ASR_OUTPUT_DIR = "asr/output/"
ASR_LICENSE = "asr/license_code"
ASR_SAMPLE_RATE = 44 


def recognize_digits(audio_path: str) -> str:
    """
    Run the ASR binary on the given audio file and return recognized digits as a space-separated string.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        ASR_BINARY,
        ASR_CONFIG,
        ASR_OUTPUT_DIR,
        audio_path,
        str(ASR_SAMPLE_RATE),
        ASR_LICENSE
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ASR binary failed:\n{result.stderr}")

    # Each result line looks like "3:0.45,0.72" — extract only the digit before ':'
    digits = re.findall(r'^(\d+):\d+\.\d+,\d+\.\d+', result.stdout, re.MULTILINE)

    return " ".join(digits)


def recognize_digits_with_timestamps(audio_path: str):
    """
    Run the ASR binary on the given audio file and return (digits, timestamps).
    digits: list of digit strings, e.g. ['3', '5', '3', '9']
    timestamps: list of (start, end) tuples, e.g. [('0.45', '0.72'), ...]
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        ASR_BINARY,
        ASR_CONFIG,
        ASR_OUTPUT_DIR,
        audio_path,
        str(ASR_SAMPLE_RATE),
        ASR_LICENSE
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ASR binary failed:\n{result.stderr}")

    digits = []
    timestamps = []
    for match in re.finditer(r'^(\d+):(\d+\.\d+),(\d+\.\d+)', result.stdout, re.MULTILINE):
        digits.append(match.group(1))
        timestamps.append((match.group(2), match.group(3)))

    return digits, timestamps


def write_lab_file(lab_path: str, digits: list, timestamps: list):
    """
    Write a .lab file with two lines:
    Line 1: digits separated by space
    Line 2: start-end timestamps separated by space
    """
    digit_line = " ".join(digits)
    ts_line = " ".join(f"{s}-{e}" for s, e in timestamps)
    with open(lab_path, "w") as f:
        f.write(digit_line + "\n")
        f.write(ts_line + "\n")


def process_root_dir(root_dir: str):
    """
    Process all audio files in root_dir/speaker_id/audio/ and save .lab files
    to root_dir/speaker_id/lab/.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    speaker_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    total_processed = 0
    total_errors = 0

    for speaker_dir in speaker_dirs:
        audio_dir = speaker_dir / "audio"
        if not audio_dir.is_dir():
            print(f"Skipping {speaker_dir.name}: no 'audio' folder found")
            continue

        lab_dir = speaker_dir / "lab"
        lab_dir.mkdir(exist_ok=True)

        audio_files = sorted(audio_dir.iterdir())
        for audio_file in audio_files:
            if not audio_file.is_file():
                continue
            lab_file = lab_dir / (audio_file.stem + ".lab")
            try:
                digits, timestamps = recognize_digits_with_timestamps(str(audio_file))
                if not digits:
                    print(f"  Warning: no digits recognized for {audio_file.name}")
                    continue
                write_lab_file(str(lab_file), digits, timestamps)
                total_processed += 1
                print(f"  {audio_file.name} -> {lab_file.name}")
            except Exception as e:
                total_errors += 1
                print(f"  Error processing {audio_file.name}: {e}")

    print(f"\nDone. Processed: {total_processed}, Errors: {total_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digit ASR: recognize digits from audio files")
    parser.add_argument("input", help="Path to a single audio file or a root directory containing speaker folders")
    parser.add_argument("--batch", action="store_true", help="Process root directory in batch mode (root_dir/speaker_id/audio/)")
    args = parser.parse_args()

    if args.batch:
        process_root_dir(args.input)
    else:
        print(recognize_digits(args.input))