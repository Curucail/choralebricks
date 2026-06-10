import os
from pathlib import Path

import pytest

from choralebricks.constants import INSTRUMENTS_WOODWIND, INSTRUMENT_STRINGS, Instrument
from choralebricks.dataset import PACKAGE_VERSION, SongDB
from choralebricks.utils import read_notes, read_sheet_music_csv


CHORALEWIND_PATH = Path(os.getenv("CHORALEWIND_PATH", "ChoraleWind"))

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
    assert choralewind.version == PACKAGE_VERSION
    assert (CHORALEWIND_PATH / "VERSION").read_text(encoding="utf-8").strip() == PACKAGE_VERSION


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
        "t_start",
        "t_dur",
        "pitch_audio",
        "f0_median",
        "level",
        "label",
    ]
    assert list(notes.columns) == ["t_start", "t_dur", "pitch_audio", "f0_median"]


def test_choralewind_score_schema(tracks):
    """Test top-level score CSVs use the explicit score-pitch name."""
    score = read_sheet_music_csv(tracks[0].path_sheet_music_csv)
    assert "pitch_sheet_music" in score.columns
    assert "pitch" not in score.columns


def test_choralewind_extra_instruments_are_woodwinds():
    """Test ChoraleWind extra instrument abbreviations."""
    assert Instrument.SAX_SOPRANO.value == "ss"
    assert Instrument.BASSOON.value == "bsn"
    assert INSTRUMENT_STRINGS[Instrument.SAX_SOPRANO] == "Soprano Saxophone"
    assert INSTRUMENT_STRINGS[Instrument.BASSOON] == "Bassoon"
    assert Instrument.SAX_SOPRANO in INSTRUMENTS_WOODWIND
    assert Instrument.BASSOON in INSTRUMENTS_WOODWIND
