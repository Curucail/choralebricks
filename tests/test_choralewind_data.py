import csv
import os
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from choralebricks import spec
from choralebricks.constants import INSTRUMENTS_WOODWIND, INSTRUMENT_STRINGS, Instrument
from choralebricks.dataset import SongDB
from choralebricks.utils import read_notes, read_sheet_music_csv


CHORALEWIND_PATH = Path(os.getenv("CHORALEWIND_PATH", "ChoraleWind"))
SECONDS_PATTERN = spec.SECONDS_PATTERN
F0_NOTE_PATTERN = spec.F0_NOTE_PATTERN
INTEGER_PATTERN = spec.INTEGER_PATTERN
MSM_PARTS = {"11": "S", "12": "A", "21": "T", "22": "B"}

pytestmark = pytest.mark.skipif(
    not CHORALEWIND_PATH.is_dir(),
    reason="ChoraleWind dataset path not found.",
)


@pytest.fixture(scope="module", name="choralewind")
def choralewind_db():
    """ChoraleWind Dataset"""
    yield SongDB(CHORALEWIND_PATH)


@pytest.fixture(scope="module")
def tracks(choralewind):
    """All ChoraleWind Dataset Tracks"""
    return [track for song in choralewind.songs for track in song.tracks]


def test_choralewind_number_of_songs_and_tracks(choralewind, tracks):
    """Test number of ChoraleWind songs and tracks."""
    assert len(choralewind.songs) == 311
    assert len(tracks) == 8397


def test_choralewind_score_paths_and_optional_chords(tracks):
    """Test ChoraleWind score fallback paths and missing chord annotations."""
    track = tracks[0]
    assert Path(track.path_audio).exists()
    assert Path(track.path_f0).exists()
    assert Path(track.path_notes).exists()
    assert Path(track.path_sheet_music_csv).name.endswith("_01-preproc.csv")
    assert Path(track.path_sheet_music_csv).exists()
    assert Path(track.path_sheet_music_midi).name.endswith("_01-preproc.mid")
    assert Path(track.path_sheet_music_midi).exists()
    assert Path(track.path_sheet_music_mei).exists()
    assert Path(track.path_sheet_music_pdf).exists()
    assert track.path_sheet_music_mxml is None
    assert track.path_chords is None
    assert track.date is None
    assert track.performer is None
    assert track.room is None
    assert track.microphone is None


def test_choralewind_notes_schema(tracks):
    """Test ChoraleWind notes expose audio pitch and median F0."""
    raw_notes = read_notes(tracks[0].path_notes, rename_cols=False)
    notes = read_notes(tracks[0].path_notes)
    assert list(raw_notes.columns) == [
        "start_sec",
        "end_sec",
        "duration_sec",
        "pitch_audio",
        "f0_note",
        "velocity",
        "label",
    ]
    assert list(notes.columns) == ["start_sec", "end_sec", "duration_sec", "pitch_audio", "f0_note"]


def test_choralewind_score_schema(tracks):
    """Test top-level score CSVs use the spec pitch column name."""
    score = read_sheet_music_csv(tracks[0].path_sheet_music_csv)
    assert "pitch" in score.columns
    assert "pitch_sheet_music" not in score.columns


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def expressive_velocities(song):
    path = song.song_dir / f"{song.id}_02-expr.msm"
    root = ET.parse(path).getroot()
    velocities = {part: [] for part in MSM_PARTS.values()}
    for part_node in root:
        part = MSM_PARTS.get(part_node.attrib.get("number", ""))
        if part is None:
            continue
        for note in part_node.iter():
            if note.tag.rsplit("}", 1)[-1] != "note":
                continue
            velocity = int(
                Decimal(note.attrib["velocity"]).quantize(
                    Decimal("1"),
                    rounding=ROUND_HALF_UP,
                )
            )
            velocities[part].append(max(0, min(127, velocity)))
    return velocities


def test_choralewind_numeric_format_and_expressive_velocities(choralewind):
    """Representative tracks and every score use expressive MSM velocity."""
    voice_parts = {1: "S", 2: "A", 3: "T", 4: "B"}
    for song in choralewind.songs:
        expected = expressive_velocities(song)
        score_rows = read_csv_rows(song.tracks[0].path_sheet_music_csv)
        for part in voice_parts.values():
            assert [
                int(row["velocity"])
                for row in sorted(
                    (row for row in score_rows if row["part"] == part),
                    key=lambda row: float(row["quarter_note_offset"]),
                )
            ] == expected[part]

        for voice, part in voice_parts.items():
            track = next(track for track in song.tracks if track.voice == voice)
            notes = read_csv_rows(track.path_notes)
            alignment = read_csv_rows(
                song.song_dir / "alignments" / f"{Path(track.path_audio).stem}.csv"
            )
            assert len(notes) == len(alignment) == len(expected[part])
            assert [int(row["velocity"]) for row in notes] == expected[part]

            for note, aligned in zip(notes, alignment):
                for field in ("start_sec", "end_sec", "duration_sec"):
                    assert SECONDS_PATTERN.fullmatch(note[field])
                    assert note[field] == aligned[field]
                assert Decimal(note["start_sec"]) + Decimal(note["duration_sec"]) == Decimal(
                    note["end_sec"]
                )
                assert INTEGER_PATTERN.fullmatch(note["pitch_audio"])
                assert F0_NOTE_PATTERN.fullmatch(note["f0_note"])
                assert int(note["pitch_audio"]) == spec.pitch_audio_from_f0_note(
                    note["f0_note"]
                )
                for field in ("pitch_audio", "f0_note", "velocity"):
                    assert note[field] == aligned[field]


def test_choralewind_extra_instruments_are_woodwinds():
    """Test ChoraleWind extra instrument abbreviations."""
    assert Instrument.SAX_SOPRANO.value == "ss"
    assert Instrument.BASSOON.value == "bsn"
    assert INSTRUMENT_STRINGS[Instrument.SAX_SOPRANO] == "Soprano Saxophone"
    assert INSTRUMENT_STRINGS[Instrument.BASSOON] == "Bassoon"
    assert Instrument.SAX_SOPRANO in INSTRUMENTS_WOODWIND
    assert Instrument.BASSOON in INSTRUMENTS_WOODWIND
