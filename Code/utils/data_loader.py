"""
Data loading utilities for Congressional speech datasets.

Handles loading raw speech text files and SpeakerMap metadata
from the Gentzkow-Shapiro Congressional Record corpus.
"""

import pandas as pd
from pathlib import Path


class CongressionalDataLoader:
    """Load and parse raw Congressional Record text and metadata files."""

    CONGRESS_RANGE = range(43, 112)
    ENCODING = "cp1252"
    DELIMITER = "|"

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def load_speeches(self) -> pd.DataFrame:
        """Parse speech text files (speeches_NNN.txt) into a DataFrame.

        Returns:
            DataFrame with columns: speech_id, speech, file_id
        """
        all_rows = []

        for i in self.CONGRESS_RANGE:
            suffix = f"{i:03d}"
            file = self.raw_dir / f"speeches_{suffix}.txt"

            if not file.exists():
                print(f"skip: {file}")
                continue

            with open(file, "r", encoding=self.ENCODING) as f:
                next(f)  # skip header

                for line in f:
                    parts = line.rstrip("\n").split(self.DELIMITER, 1)
                    if len(parts) != 2:
                        continue

                    speech_id, speech = parts
                    all_rows.append((speech_id.strip(), speech.strip(), suffix))

        return pd.DataFrame(all_rows, columns=["speech_id", "speech", "file_id"])

    def load_speaker_map(self) -> pd.DataFrame:
        """Parse SpeakerMap files (SpeakerMap_NNN.txt) into a DataFrame.

        Returns:
            DataFrame with columns: speech_id, speaker, state, party
        """
        all_frames = []

        for i in self.CONGRESS_RANGE:
            suffix = f"{i:03d}"
            file = self.raw_dir / f"SpeakerMap_{suffix}.txt"

            if not file.exists():
                print(f"skip: {file}")
                continue

            df = pd.read_csv(
                file,
                sep=self.DELIMITER,
                header=None,
                names=["speech_id", "speaker", "state", "party"],
                dtype=str,
                encoding=self.ENCODING,
            )
            all_frames.append(df)

        return pd.concat(all_frames, ignore_index=True)
