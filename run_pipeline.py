"""
run_pipeline.py

Run the full political text classification pipeline end-to-end.

Usage:
    python run_pipeline.py --raw_dir data/raw/congressional_speeches

This script orchestrates all pipeline stages sequentially:
    1. Load speeches
    2. Merge speaker metadata
    3. Add party labels
    4. Preprocess text
    5. Train baseline (TF-IDF + Logistic Regression)
    6. Tokenize for transformer
    7. Fine-tune RoBERTa
"""

import argparse
import subprocess
import sys


PIPELINE_STEPS = [
    {
        "script": "Code/01_load_speeches.py",
        "description": "Loading raw speeches",
        "uses_raw_dir": True,
    },
    {
        "script": "Code/02_merge_speaker_map.py",
        "description": "Merging speaker metadata",
        "uses_raw_dir": True,
    },
    {
        "script": "Code/03_add_party_label.py",
        "description": "Adding party labels",
    },
    {
        "script": "Code/04_preprocess_text.py",
        "description": "Preprocessing text",
    },
    {
        "script": "Code/05_train_baseline.py",
        "description": "Training baseline model (TF-IDF + Logistic Regression)",
    },
    {
        "script": "Code/06_tokenize_and_concat.py",
        "description": "Tokenizing for RoBERTa",
    },
    {
        "script": "Code/07_train_roberta.py",
        "description": "Fine-tuning RoBERTa",
    },
]


def run_step(script: str, description: str, extra_args: list[str] | None = None):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Running: {script}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full political text classification pipeline."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to directory containing raw speech and SpeakerMap text files",
    )
    parser.add_argument(
        "--skip_roberta",
        action="store_true",
        help="Skip RoBERTa fine-tuning (useful without GPU)",
    )
    args = parser.parse_args()

    steps = PIPELINE_STEPS
    if args.skip_roberta:
        steps = [s for s in steps if "roberta" not in s["script"].lower()]

    for step in steps:
        extra_args = []
        if step.get("uses_raw_dir"):
            extra_args = ["--raw_dir", args.raw_dir]

        run_step(step["script"], step["description"], extra_args)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print("\nTo launch the demo:")
    print("  streamlit run Code/08_streamlit_app.py")


if __name__ == "__main__":
    main()
