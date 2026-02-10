"""
01_load_speeches.py

Load Congressional speech text files, parse them safely, and
save the merged dataset as a parquet file.

Input:
    Raw speech text files (speeches_043.txt ~ speeches_111.txt)

Output:
    data/processed/speeches_merged.parquet
"""

import argparse
from pathlib import Path

from utils import CongressionalDataLoader


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = CongressionalDataLoader(args.raw_dir)
    df = loader.load_speeches()

    out_path = out_dir / "speeches_merged.parquet"
    df.to_parquet(out_path)

    print("Done.", df.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw Congressional speech text files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    args = parser.parse_args()
    main(args)
