"""
02_merge_speaker_map.py

Load SpeakerMap files, merge speaker metadata, and
save the combined table as a parquet file.

Input:
    SpeakerMap_043.txt ~ SpeakerMap_111.txt

Output:
    data/processed/speaker_map.parquet
"""

import argparse
from pathlib import Path

from utils import CongressionalDataLoader


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = CongressionalDataLoader(args.raw_dir)
    speaker_map = loader.load_speaker_map()

    out_path = out_dir / "speaker_map.parquet"
    speaker_map.to_parquet(out_path)

    print("Done.", speaker_map.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw SpeakerMap files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    args = parser.parse_args()
    main(args)
