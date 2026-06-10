"""
All tests related to dataset.py and the involved logic.
"""
import tomllib
from pathlib import Path

import pytest

from choralebricks.constants import Instrument
from choralebricks.dataset import (
    PACKAGE_VERSION,
    EnsemblePermutations,
    Song,
    SongDB,
    Track,
)


@pytest.fixture
def mockupdb():
    """Mockup database with a single song."""

    def mocktrack(
        voice: int,
        instrument: Instrument
    ):

        track = Track(
            song_id="test_song_01",
            path_audio=f"{instrument.value}_{voice}.wav",
            num_channels=1,
            sample_rate=44100,
            min_samples=441000,
            voice=voice,
            instrument=instrument
        )

        return track

    song01_tracks = [
        mocktrack(1, Instrument.TRUMPET),
        mocktrack(1, Instrument.CLARINET),
        mocktrack(2, Instrument.TRUMPET),
        mocktrack(2, Instrument.CLARINET),
        mocktrack(3, Instrument.BARITONE),
        mocktrack(4, Instrument.BARITONE),
        mocktrack(4, Instrument.TUBA),
    ]

    song_01 = Song(Path("song_01"))
    song_01.tracks = song01_tracks

    return [song_01, ]


@pytest.fixture
def ensembles(mockupdb):
    """All Possible Ensemble Permutations"""
    return [ens for song in mockupdb for ens in EnsemblePermutations(song)]


def test_number_of_songs(mockupdb):
    """Test number of songs"""
    assert len(mockupdb) == 1


def test_number_of_ensembles(ensembles):
    """Test number of ensembles"""
    assert len(ensembles) == 2 * 2 * 1 * 2


def test_instrument_type(mockupdb):
    """Test number of songs"""

    for cur_song in mockupdb:
        for cur_track in cur_song.tracks:
            assert cur_track.instrument_type != None


def test_import_my_module():
    # Attempt to import the module
    try:
        import choralebricks.dataset
    except ImportError:
        pytest.fail("Importing my_module failed")


def test_songdb_requires_version_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset VERSION file not found"):
        SongDB(tmp_path)


def test_songdb_rejects_mismatched_version(tmp_path):
    (tmp_path / "VERSION").write_text("0.0.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        SongDB(tmp_path)


def test_songdb_accepts_matching_version(tmp_path):
    (tmp_path / "VERSION").write_text(f"{PACKAGE_VERSION}\n", encoding="utf-8")
    (tmp_path / "metadata_songs.csv").write_text(
        "song_id;composer;title;year\n",
        encoding="utf-8",
    )

    song_db = SongDB(tmp_path)

    assert song_db.version == PACKAGE_VERSION
    assert song_db.songs == []


def test_package_version_matches_pyproject():
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as handle:
        pyproject_version = tomllib.load(handle)["project"]["version"]

    assert PACKAGE_VERSION == pyproject_version
