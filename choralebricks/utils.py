import numpy as np
import pandas as pd
from pathlib import Path

from choralebricks.constants import Voices, VOICE_STRINGS


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    def __init__(
            self,
            message="The schema of the file does not match the expected format"
        ):
        super().__init__(message)

def validate_schema(
        df,
        expected_columns
    ):
    """Validate if the DataFrame schema matches the expected columns."""
    if list(df.columns) != list(expected_columns):
        raise SchemaValidationError(
            f"Schema mismatch. Expected columns: {expected_columns}, but got: {list(df.columns)}"
        )


def read_f0_sv(
    path_csv: Path,
    rename_cols: bool=True
) -> pd.DataFrame:
    expected_columns = ["t", "f0", "label"]

    if path_csv == None:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if path_csv.exists():
        df = pd.read_csv(path_csv, sep=";")
        validate_schema(df, expected_columns)
    else:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if rename_cols:
        df = df.drop(columns=["label"])
    return df


def read_f0(
    path_csv: Path,
) -> pd.DataFrame:
    expected_columns = ["t", "f0"]

    if path_csv == None:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if path_csv.exists():
        df = pd.read_csv(path_csv, sep=";")
        validate_schema(df, expected_columns)
    else:
        raise FileNotFoundError(f"File not found: {path_csv}")

    return df


def read_notes(
    path_csv: Path,
    rename_cols: bool=True
) -> pd.DataFrame:
    expected_columns = ["start", "end", "duration", "pitch_audio", "f0_median", "velocity", "label"]

    if path_csv == None:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if path_csv.exists():
        df = pd.read_csv(path_csv, sep=";")
        validate_schema(df, expected_columns)
    else:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if rename_cols:
        df = df.drop(columns=["velocity", "label"])

    return df


def read_sheet_music_csv(
    path_csv: Path,
    A4: float=442.0
) -> pd.DataFrame:
    # A4 defaults to 442 Hz: the ensembles tuned to 442, and the audio-derived
    # pitch_audio / f0_median columns use the same reference. Keeping the score
    # `pitch_center_freq` on 442 makes score-vs-audio pitch comparisons unbiased.
    expected_columns = [
        "start_meas",
        "end_meas",
        "duration_quarter",
        "pitch",
        "pitch_name",
        "part",
        "time_sig",
        "articulation",
        "expression",
        "velocity",
        "quarter_note_offset",
        "quarter_note_BPM",
        "midiChannel",
    ]

    if path_csv == None:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if path_csv.exists():
        df = pd.read_csv(path_csv, sep=";")
        validate_schema(df, expected_columns)
    else:
        raise FileNotFoundError(f"File not found: {path_csv}")

    df["dur_meas"] = df["end_meas"] - df["start_meas"]
    df["pitch_center_freq"] = A4 * 2**((df["pitch"] - 69) / 12)

    return df


def read_chords(path_csv: Path) -> pd.DataFrame:
    expected_columns = ['start_meas', 'end_meas', 'chord']

    if path_csv == None:
        raise FileNotFoundError(f"File not found: {path_csv}")

    if path_csv.exists():
        df = pd.read_csv(path_csv, sep=";")
        validate_schema(df, expected_columns)
    else:
        raise FileNotFoundError(f"File not found: {path_csv}")

    return df


def voice_to_name(voice_value: int) -> str:
    # Mapping from Voices enum to strings
    try:
        voice_enum = Voices(voice_value)  # Convert value to enum
        return VOICE_STRINGS[voice_enum]  # Get the corresponding string
    except (ValueError, KeyError):
        return "Unknown"  # Handle invalid values


def get_voice_from_int(value):
    try:
        return Voices(value)
    except ValueError:
        return None


def midi2hz(p, f_ref=440):
    """ Returns center frequency in Hz for a given (optionally fractional) MIDI pitch
    """
    return f_ref * np.power(2, (p - 69) / 12)

def hz2midi(f, f_ref=440):
    """ Returns (optionally fractional) MIDI pitch for a given frequency in Hz
    """
    return np.log2(f/f_ref) * 12 + 69
