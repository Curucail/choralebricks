"""Specification class for the ChoraleBricks v1.1 CSV data contract.

This module centralises the column layouts, numeric formatting, and derived-column
formulas used by the conformance test-suite, and serves as a reference for 
implementors of the data contract.
"""

from __future__ import annotations

import bisect
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# --------------------------------------------------------------------------- #
# Column layouts (v1.1 data contract)
# --------------------------------------------------------------------------- #
SCORE_FIELDS = [
    "start_meas",
    "end_meas",
    "start_quarter",
    "dur_quarter",
    "time_sig",
    "pitch",
    "pitch_name",
    "part",
    "instrument",
    "articulation",
    "expression",
    "dynamic",
    "tempo_qpm",
    "start_sec",
    "end_sec",
    "dur_sec",
    "midi_velocity",
]
ALIGNMENT_FIELDS = [
    "start_meas",
    "end_meas",
    "start_quarter",
    "dur_quarter",
    "time_sig",
    "pitch",
    "pitch_name",
    "pitch_written",
    "part",
    "instrument",
    "articulation",
    "expression",
    "dynamic",
    "tempo_qpm",
    "start_sec",
    "end_sec",
    "dur_sec",
    "pitch_dev_cents",
    "midi_velocity",
]
NOTES_FIELDS = ["start_sec", "end_sec", "dur_sec", "pitch", "pitch_name", "pitch_written", "pitch_dev_cents", "midi_velocity"]
RAW_F0_FIELDS = ["t", "f0", "label"]
FILLED_F0_FIELDS = ["t", "f0"]
CHORD_FIELDS = ["start_meas", "end_meas", "chord"]
METADATA_SONG_FIELDS = ["song_id", "composer", "title", "year"]
METADATA_TRACK_FIELDS = [
    "song_id",
    "part",
    "instrument",
    "path_audio",
    "path_f0",
    "path_notes",
    "date",
    "performer",
    "room",
    "microphone",
    "comments",
]

SCORE_PART_ORDER = {"S": 0, "A": 1, "T": 2, "B": 3}
QUARTER_VALUE_FIELDS = {"duration_quarter", "quarter_note_offset", "quarter_note_BPM"}

# --------------------------------------------------------------------------- #
# Numeric constants
# --------------------------------------------------------------------------- #
A4_HZ = 440.0
MEASURE_STEP = Decimal("0.001")
QUARTER_VALUE_STEP = Decimal("0.001")
SECONDS_STEP = Decimal("0.000000001")
F0_NOTE_STEP = Decimal("0.001")
INTEGER_STEP = Decimal("1")
MIDI_VELOCITY_MIN = 0
MIDI_VELOCITY_MAX = 127
# `format_measure_value` operates on computed floats, so a near-integer measure
# position is snapped to an exclusive boundary within this tolerance.
MEASURE_BOUNDARY_OFFSET = 0.001
MEASURE_INTEGER_TOLERANCE = 1e-9

# --------------------------------------------------------------------------- #
# String-format patterns (used by the conformance test-suite)
# --------------------------------------------------------------------------- #
MEASURE_PATTERN = re.compile(r"^-?\d{3,}\.\d{3}$")
QUARTER_VALUE_PATTERN = re.compile(r"^-?\d{3,}\.\d{3}$")
SECONDS_PATTERN = re.compile(r"^-?\d+\.\d{9}$")
F0_NOTE_PATTERN = re.compile(r"^\d+\.\d{3}$")
INTEGER_PATTERN = re.compile(r"^-?\d+$")
PITCH_NAME_PATTERN = re.compile(r"^([A-G])([#-]*)(-?\d+)$")
_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTALS = {"": 0, "#": 1, "##": 2, "-": -1, "--": -2}


# --------------------------------------------------------------------------- #
# Numeric formatting
# --------------------------------------------------------------------------- #
def quantize_decimal(value: Any, step: Decimal) -> Decimal:
    return Decimal(str(value)).quantize(step, rounding=ROUND_HALF_UP)


def format_decimal(value: Decimal, places: int) -> str:
    return f"{value:.{places}f}"


def _coerce_semicolon_csv_types(df: pd.DataFrame) -> pd.DataFrame:
    typed = df.copy()
    for column in typed.columns:
        series = typed[column]
        numeric = pd.to_numeric(series, errors="coerce")
        non_empty = series != ""
        if non_empty.any() and numeric[non_empty].notna().all():
            typed[column] = numeric
    return typed


def format_measure_column_value(value: Any, exclusive_end: bool = False) -> str:
    """Fixed-width three-decimal measure position.

    With ``exclusive_end`` a position that lands on (or numerically rounds to) an
    integer measure is made exclusive, e.g. ``005.000 -> 004.999``.
    """
    numeric_value = float(value)
    if exclusive_end:
        nearest_integer = round(numeric_value)
        if math.isclose(numeric_value, nearest_integer, abs_tol=MEASURE_INTEGER_TOLERANCE):
            numeric_value = nearest_integer - MEASURE_BOUNDARY_OFFSET
        elif math.isclose(round(numeric_value, 3), round(round(numeric_value, 3)), abs_tol=MEASURE_INTEGER_TOLERANCE):
            numeric_value = round(round(numeric_value, 3)) - MEASURE_BOUNDARY_OFFSET
    sign = "-" if numeric_value < 0 else ""
    return f"{sign}{abs(numeric_value):.3f}"


def format_quarter_value(value: Any) -> str:
    numeric_value = quantize_decimal(value, QUARTER_VALUE_STEP)
    sign = "-" if numeric_value < 0 else ""
    return f"{sign}{abs(numeric_value):07.3f}"


def format_seconds(value: Any) -> str:
    return format_decimal(quantize_decimal(value, SECONDS_STEP), 9)


def format_f0_note(value: Any) -> str:
    """Format an already-computed F0 value to three decimals."""
    return format_decimal(quantize_decimal(value, F0_NOTE_STEP), 3)


def midi_velocity(value: Any) -> int:
    velocity = int(quantize_decimal(value, INTEGER_STEP))
    return max(MIDI_VELOCITY_MIN, min(MIDI_VELOCITY_MAX, velocity))


def velocity_from_scalar(value: Any) -> str:
    """Convert a [0, 1] loudness scalar (volume/level) to MIDI velocity in [0, 127]."""
    velocity = int((Decimal(str(value)) * MIDI_VELOCITY_MAX).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return str(max(MIDI_VELOCITY_MIN, min(MIDI_VELOCITY_MAX, velocity)))


# --------------------------------------------------------------------------- #
# Derived columns
# --------------------------------------------------------------------------- #
def pitch_audio_from_f0_note(f0_note: Any) -> int:
    """Audio MIDI pitch ``round(12*log2(f0_note / 440) + 69)`` (A4 = 440 Hz)."""
    midi = Decimal(12) * Decimal(math.log2(float(f0_note) / A4_HZ)) + Decimal(69)
    return int(midi.quantize(INTEGER_STEP, rounding=ROUND_HALF_UP))


def pitch_name_to_midi(pitch_name: str) -> int:
    """MIDI pitch for a spelled pitch name such as ``F#4`` or ``B-3``."""
    match = PITCH_NAME_PATTERN.fullmatch(pitch_name)
    if not match:
        raise ValueError(f"Unsupported pitch name: {pitch_name}")
    pitch_class, accidental, octave = match.groups()
    return (int(octave) + 1) * 12 + _PITCH_CLASSES[pitch_class] + _ACCIDENTALS[accidental]


def note_intervals(notes: list[dict[str, str]]) -> list[tuple[float, float]]:
    return [
        (float(note["start_sec"]), float(note["start_sec"]) + float(note["dur_sec"]))
        for note in notes
    ]


def time_in_intervals(time: float, sorted_intervals: list[tuple[float, float]]) -> bool:
    """Whether ``time`` falls in any interval; ``sorted_intervals`` must be sorted by start."""
    starts = [start for start, _ in sorted_intervals]
    index = bisect.bisect_right(starts, time) - 1
    return index >= 0 and time <= sorted_intervals[index][1]


# --------------------------------------------------------------------------- #
# CSV IO (semicolon-delimited, UTF-8, LF line endings)
# --------------------------------------------------------------------------- #
def read_semicolon_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    df = pd.read_csv(Path(path), sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = _coerce_semicolon_csv_types(df)
    return list(df.columns), df.to_dict(orient="records")


def write_semicolon_csv(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    df = pd.DataFrame.from_records(list(rows), columns=fields)
    df.to_csv(Path(path), sep=";", index=False, lineterminator="\n", encoding="utf-8")
