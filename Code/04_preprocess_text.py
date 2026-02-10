"""
04_preprocess_text.py

Preprocess congressional speech data by:
1. Restricting to 1980s speeches (97th-100th Congress)
2. Applying paragraph-level filtering (>=2 sentences, >=200 characters)
3. Cleaning text via Unicode and whitespace normalization

Output:
    data/processed/speeches_clean_1980s_paragraph.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from utils import TextPreprocessor


def main(args):
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    print("Loaded:", df.shape)

    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess(df)

    out_path = out_dir / "speeches_clean_1980s_paragraph.parquet"
    df_clean.to_parquet(out_path)

    print("Saved cleaned dataset:", df_clean.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/speeches_with_party.parquet",
        help="Path to speeches with party labels",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    args = parser.parse_args()
    main(args)
