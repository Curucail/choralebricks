"""Specification class for the ChoraleBricks v1.1 CSV data contract.

This module centralises the column layouts, numeric formatting, and derived-column
formulas used by the conformance test-suite, and serves as a reference for 
implementors of the data contract.
"""

from __future__ import annotations

import bisect
import csv
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Column layouts (v1.1 data contract)
# --------------------------------------------------------------------------- #
SCORE_FIELDS = [
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
ALIGNMENT_FIELDS = [
    "start_meas",
    "end_meas",
    "duration_quarter",
    "pitch",
    "pitch_name",
    "part",
    "time_sig",
    "velocity",
    "start",
    "end",
    "duration",
    "pitch_audio",
    "f0_median",
]
NOTES_FIELDS = ["start", "end", "duration", "pitch_audio", "f0_median", "velocity", "label"]
RAW_F0_FIELDS = ["t", "f0", "label"]
FILLED_F0_FIELDS = ["t", "f0"]
CHORD_FIELDS = ["start_meas", "end_meas", "chord"]
METADATA_SONG_FIELDS = ["song_id", "composer", "title", "year"]
METADATA_TRACK_FIELDS = [
    "song_id",
    "voice",
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
A4_HZ = 442.0
MEASURE_STEP = Decimal("0.001")
QUARTER_VALUE_STEP = Decimal("0.001")
SECONDS_STEP = Decimal("0.000000001")
F0_MEDIAN_STEP = Decimal("0.001")
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
F0_MEDIAN_PATTERN = re.compile(r"^\d+\.\d{3}$")
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


def format_measure_value(value: Any, exclusive_end: bool = False) -> str:
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
    return f"{sign}{abs(numeric_value):07.3f}"


def format_quarter_value(value: Any) -> str:
    numeric_value = quantize_decimal(value, QUARTER_VALUE_STEP)
    sign = "-" if numeric_value < 0 else ""
    return f"{sign}{abs(numeric_value):07.3f}"


def format_seconds(value: Any) -> str:
    return format_decimal(quantize_decimal(value, SECONDS_STEP), 9)


def format_f0_median(value: Any) -> str:
    """Format an already-computed F0 value to three decimals."""
    return format_decimal(quantize_decimal(value, F0_MEDIAN_STEP), 3)


def f0_median_of_window(values: Iterable[Decimal]) -> str:
    """Median of raw F0 ``Decimal`` values in a note window, formatted to 3 decimals."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate an F0 median from an empty window.")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / Decimal(2)
    return f"{median.quantize(F0_MEDIAN_STEP, rounding=ROUND_HALF_UP):.3f}"


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
def pitch_audio_from_f0_median(f0_median: Any) -> int:
    """Audio MIDI pitch ``round(12*log2(f0_median / 442) + 69)`` (A4 = 442 Hz)."""
    midi = Decimal(12) * Decimal(math.log2(float(f0_median) / A4_HZ)) + Decimal(69)
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
        (float(note["start"]), float(note["start"]) + float(note["duration"]))
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
def read_semicolon_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader.fieldnames or []), list(reader)


def write_semicolon_csv(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
