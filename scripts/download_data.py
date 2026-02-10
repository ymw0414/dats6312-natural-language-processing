"""
download_data.py

Download the Congressional Record dataset from the Stanford
Gentzkow-Shapiro data archive.

Source:
    https://data.stanford.edu/congress_text

The dataset includes raw speech text and SpeakerMap metadata
for the 43rd through 111th U.S. Congress.

Usage:
    python scripts/download_data.py --out_dir data/raw/congressional_speeches

Note:
    This script downloads and extracts the public "hein-daily" files.
    Total download size is approximately 3-4 GB.
"""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://stacks.stanford.edu/file/druid:md374tz9962"

FILE_GROUPS = {
    "speeches": {
        "url": f"{BASE_URL}/hein-daily.zip",
        "description": "Daily Congressional speech text files",
    },
    "speaker_map": {
        "url": f"{BASE_URL}/hein-daily_SpeakerMap.zip",
        "description": "Speaker metadata (SpeakerMap) files",
    },
}


def download_file(url: str, dest: Path, description: str):
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=progress_hook)
    print()


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Congressional Record data from Stanford."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/raw/congressional_speeches",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep zip files after extraction",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, info in FILE_GROUPS.items():
        zip_path = out_dir / f"{name}.zip"
        download_file(info["url"], zip_path, info["description"])
        extract_zip(zip_path, out_dir)

        if not args.keep_zip:
            zip_path.unlink()
            print(f"  Removed: {zip_path}")

    print(f"\nDone. Raw data saved to: {out_dir}")
    print("\nNext step:")
    print(f"  python run_pipeline.py --raw_dir {out_dir}")


if __name__ == "__main__":
    main()
