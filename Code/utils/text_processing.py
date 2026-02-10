"""
Text preprocessing utilities for Congressional speech data.
"""

import re
import unicodedata

import pandas as pd


class TextPreprocessor:
    """Clean and filter Congressional speech text."""

    CONGRESS_1980S = [97, 98, 99, 100]
    VALID_PARTIES = ["D", "R"]
    MIN_SENTENCES = 2
    MIN_CHARACTERS = 200

    @staticmethod
    def fix_unicode(text: str) -> str:
        """Normalize Unicode to NFKC form."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Collapse consecutive whitespace into a single space."""
        return re.sub(r"\s+", " ", text).strip()

    def preprocess(
        self,
        df: pd.DataFrame,
        congress_sessions: list[int] | None = None,
        min_sentences: int | None = None,
        min_characters: int | None = None,
    ) -> pd.DataFrame:
        """Apply full preprocessing pipeline.

        Args:
            df: DataFrame with columns speech, file_id, party.
            congress_sessions: Congress numbers to keep (default: 1980s).
            min_sentences: Minimum sentence count per speech.
            min_characters: Minimum character count per speech.

        Returns:
            Cleaned and filtered DataFrame.
        """
        sessions = congress_sessions or self.CONGRESS_1980S
        min_sent = min_sentences if min_sentences is not None else self.MIN_SENTENCES
        min_char = min_characters if min_characters is not None else self.MIN_CHARACTERS

        df = df.copy()

        # Filter by Congress session
        df["file_id"] = df["file_id"].astype(int)
        df = df[df["file_id"].isin(sessions)]

        # Keep valid party labels
        df = df[df["party"].isin(self.VALID_PARTIES)]

        # Drop missing or empty text
        df = df.dropna(subset=["speech"])
        df = df[df["speech"].str.strip().str.len() > 0]

        # Paragraph-level filtering
        df["num_sent"] = df["speech"].str.count(r"[\.!?]")
        df = df[(df["num_sent"] >= min_sent) & (df["speech"].str.len() >= min_char)]

        # Text cleaning
        df["speech"] = df["speech"].apply(self.fix_unicode)
        df["speech"] = df["speech"].apply(self.clean_whitespace)

        return df
