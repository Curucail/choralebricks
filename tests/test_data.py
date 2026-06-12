import bisect
import csv
import math
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from choralebricks.dataset import EnsemblePermutations, SongDB
from choralebricks.utils import read_f0_sv, read_f0, read_notes, read_chords
from choralebricks import ChordSequence


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
NOTES_FIELDS = [
    "start",
    "end",
    "duration",
    "pitch_audio",
    "f0_median",
    "velocity",
    "label",
]
RAW_F0_FIELDS = ["t", "f0", "label"]
FILLED_F0_FIELDS = ["t", "f0"]
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
SCORE_PART_ORDER = {"S": 0, "A": 1, "T": 2, "B": 3}
CHORD_FIELDS = ["start_meas", "end_meas", "chord"]
ALIGNMENT_SCORE_FIELDS = [
    "start_meas",
    "end_meas",
    "pitch_name",
    "time_sig",
    "part",
]
NUMERIC_ALIGNMENT_SCORE_FIELDS = [
    "duration_quarter",
    "pitch",
]
MEASURE_PATTERN = re.compile(r"^-?\d{3,}\.\d{3}$")
QUARTER_VALUE_PATTERN = re.compile(r"^-?\d{3,}\.\d{3}$")
F0_MEDIAN_PATTERN = re.compile(r"^\d+\.\d{3}$")
A4_HZ = 442.0
PITCH_NAME_PATTERN = re.compile(r"^([A-G])([#-]*)(-?\d+)$")
PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTALS = {"": 0, "#": 1, "##": 2, "-": -1, "--": -2}


# Check for the environment variable CHORALEDB_PATH
choraledb_path = os.getenv('CHORALEDB_PATH')

if choraledb_path:
    TRACKS = [track for song in SongDB().songs for track in song.tracks]
    tr_ids = [f"{track.song_id}-{track.path_audio.stem}" for track in TRACKS]
else:
    TRACKS = []
    tr_ids = []


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader.fieldnames or []), list(reader)


def raw_f0_path(track):
    return Path(track.path_f0).with_name(
        Path(track.path_f0).name.replace("_f0_filled.csv", "_f0.csv")
    )


def alignment_path(track):
    return (
        Path(track.path_notes).parent.parent
        / "alignments"
        / Path(track.path_notes).name.replace("_notes.csv", ".csv")
    )


def note_intervals(notes):
    return [
        (
            Decimal(note["start"]),
            Decimal(note["start"]) + Decimal(note["duration"]),
        )
        for note in notes
    ]


def time_in_intervals(time, intervals):
    starts = [start for start, _ in intervals]
    index = bisect.bisect_right(starts, time) - 1
    return index >= 0 and time <= intervals[index][1]


def format_f0_median(values):
    ordered = sorted(values)
    assert ordered
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / Decimal(2)
    )
    return f"{median.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP):.3f}"


def pitch_audio_from_f0_median(f0_median):
    midi = Decimal(12) * Decimal(math.log2(float(f0_median) / A4_HZ)) + Decimal(69)
    return int(midi.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def pitch_name_to_midi(pitch_name):
    match = PITCH_NAME_PATTERN.fullmatch(pitch_name)
    assert match, f"Unsupported pitch name: {pitch_name}"
    pitch_class, accidental, octave = match.groups()
    return (
        (int(octave) + 1) * 12
        + PITCH_CLASSES[pitch_class]
        + ACCIDENTALS[accidental]
    )


"""
    Test Fixtures
"""
@pytest.fixture(name="choralebricks")
def songdb():
    """ChoraleBricks Dataset"""
    choralebricks = SongDB()
    yield choralebricks


@pytest.fixture
def tracks(choralebricks):
    """All Dataset Tracks"""
    return [track for song in choralebricks.songs for track in song.tracks]


@pytest.fixture
def songs(choralebricks):
    """All Dataset Songs"""
    return choralebricks.songs


@pytest.fixture
def ensembles(choralebricks):
    """All Possible Ensemble Permutations"""
    return [ens for song in choralebricks.songs for ens in EnsemblePermutations(song)]


"""
    Data Integration Tests
"""
def test_number_of_songs(songs):
    """Test number of songs"""
    assert len(songs) == 10


def test_track_count(choralebricks, tracks):
    assert len(tracks) == 193


def test_dataset_has_no_orphan_annotations(choralebricks):
    referenced = {
        Path(track.path_notes).resolve()
        for song in choralebricks.songs
        for track in song.tracks
    }
    actual = {
        path.resolve()
        for path in choralebricks.root_dir.rglob("*_notes.csv")
    }
    assert actual == referenced


def test_number_of_ensembles(ensembles):
    """Test number of ensembles"""
    assert len(ensembles) == 4582


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_paths_audio_not_none(track):
    """Test paths for each track"""
    assert track.path_audio is not None


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_paths_f0_not_none(track):
    """Test paths for each track"""
    assert track.path_f0 is not None


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_paths_notes_not_none(track):
    """Test paths for each track"""
    assert track.path_notes is not None


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_files_exist(track):
    """Test files exist"""
    assert Path(track.path_audio).exists()
    assert Path(track.path_f0).exists()
    assert Path(track.path_notes).exists()


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_files_not_empty(track):
    """Test files are not empty"""
    assert Path(track.path_audio).stat().st_size != 0
    assert Path(track.path_f0).stat().st_size != 0
    assert Path(track.path_notes).stat().st_size != 0


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_track_suffix(track):
    """Test track suffix."""
    assert Path(track.path_audio).suffix == ".wav"
    assert Path(track.path_f0).suffix == ".csv"
    assert Path(track.path_notes).suffix == ".csv"


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_csv_headers(track):
    """Test CSV file headers from Sonic Visualizer."""
    path_sv_f0 = Path(str(track.path_f0).replace("_filled", ""))
    f0_head = read_f0_sv(path_sv_f0, rename_cols=False).columns if track.path_f0 else []
    notes_head = read_notes(track.path_notes, rename_cols=False).columns if track.path_notes else []
    assert list(f0_head) == ["t", "f0", "label"]
    assert list(notes_head) == ["start", "end", "duration", "pitch_audio", "f0_median", "velocity", "label"]


def test_all_csv_files_are_semicolon_delimited(choralebricks):
    for path in choralebricks.root_dir.rglob("*.csv"):
        header, _ = read_csv_rows(path)
        assert len(header) > 1, f"{path} is not semicolon-delimited"


def test_v11_csv_schemas(songs, tracks):
    for song in songs:
        score_header, _ = read_csv_rows(song.tracks[0].path_sheet_music_csv)
        chord_header, _ = read_csv_rows(song.tracks[0].path_chords)
        assert score_header == SCORE_FIELDS
        assert chord_header == CHORD_FIELDS

    for track in tracks:
        notes_header, _ = read_csv_rows(track.path_notes)
        raw_header, _ = read_csv_rows(raw_f0_path(track))
        filled_header, _ = read_csv_rows(track.path_f0)
        alignment_header, _ = read_csv_rows(alignment_path(track))
        assert notes_header == NOTES_FIELDS
        assert raw_header == RAW_F0_FIELDS
        assert filled_header == FILLED_F0_FIELDS
        assert alignment_header == ALIGNMENT_FIELDS


def test_score_rows_are_sorted_by_start_and_satb(songs):
    for song in songs:
        _, rows = read_csv_rows(song.tracks[0].path_sheet_music_csv)
        keys = [
            (Decimal(row["start_meas"]), SCORE_PART_ORDER[row["part"]])
            for row in rows
        ]
        assert keys == sorted(keys)


def test_measure_format_and_exclusive_ends(choralebricks):
    affected_rows = 0
    for path in choralebricks.root_dir.rglob("*.csv"):
        header, rows = read_csv_rows(path)
        if "start_meas" not in header and "end_meas" not in header:
            continue
        assert "start_meas" in header and "end_meas" in header
        for row in rows:
            assert MEASURE_PATTERN.fullmatch(row["start_meas"])
            assert MEASURE_PATTERN.fullmatch(row["end_meas"])
            end = Decimal(row["end_meas"])
            assert end != end.to_integral_value()
        affected_rows += len(rows)
    assert affected_rows == 11447


def test_quarter_value_format(choralebricks):
    score_rows = 0
    alignment_rows = 0
    for song in choralebricks.songs:
        _, score = read_csv_rows(song.tracks[0].path_sheet_music_csv)
        for row in score:
            for field in (
                "duration_quarter",
                "quarter_note_offset",
                "quarter_note_BPM",
            ):
                assert QUARTER_VALUE_PATTERN.fullmatch(row[field])
        score_rows += len(score)

        for path in (song.song_dir / "alignments").glob("*.csv"):
            _, alignment = read_csv_rows(path)
            for row in alignment:
                assert QUARTER_VALUE_PATTERN.fullmatch(row["duration_quarter"])
            alignment_rows += len(alignment)

    assert score_rows == 1887
    assert alignment_rows == 9097


def test_score_pitch_names_match_midi(songs):
    score_rows = 0
    for song in songs:
        _, rows = read_csv_rows(song.tracks[0].path_sheet_music_csv)
        for row in rows:
            assert Decimal(row["pitch"]) == pitch_name_to_midi(
                row["pitch_name"]
            )
        score_rows += len(rows)
    assert score_rows == 1887


def test_alignments_match_notes_and_scores(songs):
    note_rows = 0
    for song in songs:
        _, score_rows = read_csv_rows(song.tracks[0].path_sheet_music_csv)
        score_by_part = {
            part: sorted(
                (row for row in score_rows if row["part"] == part),
                key=lambda row: Decimal(row["quarter_note_offset"]),
            )
            for part in ("S", "A", "T", "B")
        }
        for track in song.tracks:
            _, notes = read_csv_rows(track.path_notes)
            _, alignment = read_csv_rows(alignment_path(track))
            assert len(notes) == len(alignment)
            assert [row["pitch_audio"] for row in notes] == [
                row["pitch_audio"] for row in alignment
            ]
            assert [row["f0_median"] for row in notes] == [
                row["f0_median"] for row in alignment
            ]
            assert [row["velocity"] for row in notes] == [
                row["velocity"] for row in alignment
            ]
            part = alignment[0]["part"]
            score_voice = score_by_part[part]
            assert len(score_voice) == len(alignment)
            for score_row, alignment_row in zip(score_voice, alignment):
                assert {
                    field: alignment_row[field]
                    for field in ALIGNMENT_SCORE_FIELDS
                } == {
                    field: score_row[field]
                    for field in ALIGNMENT_SCORE_FIELDS
                }
                for field in NUMERIC_ALIGNMENT_SCORE_FIELDS:
                    assert Decimal(alignment_row[field]) == Decimal(
                        score_row[field]
                    )
                assert Decimal(
                    alignment_row["pitch"]
                ) == pitch_name_to_midi(alignment_row["pitch_name"])
            note_rows += len(notes)
    assert note_rows == 9097


def test_f0_annotations_and_note_medians(tracks):
    note_rows = 0
    for track in tracks:
        _, notes = read_csv_rows(track.path_notes)
        _, raw = read_csv_rows(raw_f0_path(track))
        _, filled = read_csv_rows(track.path_f0)
        intervals = note_intervals(notes)

        raw_times = [Decimal(row["t"]) for row in raw]
        raw_values = [Decimal(row["f0"]) for row in raw]
        assert raw_times == sorted(raw_times)
        assert all(value > 0 for value in raw_values)
        assert all(time_in_intervals(time, intervals) for time in raw_times)

        filled_times = [Decimal(row["t"]) for row in filled]
        assert filled_times == sorted(filled_times)
        assert len(filled_times) == len(set(filled_times))
        filled_steps = [
            later - earlier
            for earlier, later in zip(filled_times, filled_times[1:])
        ]
        assert all(step > 0 for step in filled_steps)
        median_step = sorted(filled_steps)[len(filled_steps) // 2]
        assert max(filled_steps) <= median_step * 2
        for row in filled:
            time = Decimal(row["t"])
            value = Decimal(row["f0"])
            if value == 0:
                assert row["f0"] == "0.0"
            else:
                assert time_in_intervals(time, intervals)

        for note in notes:
            median = note["f0_median"]
            assert F0_MEDIAN_PATTERN.fullmatch(median)
            assert Decimal(median) > 0
            start = Decimal(note["start"])
            end = start + Decimal(note["duration"])
            first = bisect.bisect_left(raw_times, start)
            last = bisect.bisect_right(raw_times, end)
            assert median == format_f0_median(raw_values[first:last])
        note_rows += len(notes)
    assert note_rows == 9097


def test_pitch_audio_derived_from_f0_median(tracks):
    """pitch_audio is round(12*log2(f0_median / 442) + 69), A4 = 442 Hz."""
    note_rows = 0
    for track in tracks:
        _, notes = read_csv_rows(track.path_notes)
        _, alignment = read_csv_rows(alignment_path(track))
        for note in notes:
            assert int(note["pitch_audio"]) == pitch_audio_from_f0_median(
                note["f0_median"]
            )
        for aligned in alignment:
            assert int(aligned["pitch_audio"]) == pitch_audio_from_f0_median(
                aligned["f0_median"]
            )
        note_rows += len(notes)
    assert note_rows == 9097


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_track_samplerate(track):
    """Test if all tracks have the same samplerate."""
    if track.path_audio:
        _, sr = sf.read(track.path_audio)
        assert sr == track.sample_rate


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_track_min_samples(track):
    """Test if track has the given min_samples."""
    if track.path_audio:
        data, _ = sf.read(track.path_audio)
        assert len(data) == track.min_samples


def test_track_len_per_song(songs):
    """Test if all songs have tracks with same number of samples."""
    for song in songs:
        track_lengths = []
        for track in song.tracks:
            if track.path_audio:
                data, _ = sf.read(track.path_audio)
                track_lengths.append(len(data))
        assert len(set(track_lengths)) <= 1, f"Not all audio files of {song.id} have the same length."


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_dur_f0_audio(track):
    """Audio and F0-annotations should have similar length (+-1 seconds)"""
    dur_audio = track.min_samples / track.sample_rate
    path_sv_f0 = Path(str(track.path_f0).replace("_filled", ""))
    dur_f0 = read_f0_sv(path_sv_f0).tail(1)["t"].values[0]
    assert np.abs(dur_audio - dur_f0) < 1.0


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_f0_trajectory_uniqueness(track):
    """All F0-trajectories should have only one entry per time instance"""
    df = read_f0(track.path_f0)
    assert df.shape[0] == df.drop_duplicates("t").shape[0]


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_dur_note_audio(track):
    """Audio and note annotations should have similar length (+-1 seconds)"""
    dur_audio = track.min_samples / track.sample_rate
    last_note = read_notes(track.path_notes).tail(1)
    dur_notes = (last_note["start"] + last_note["duration"]).values[0]
    assert np.abs(dur_audio - dur_notes) < 1.0


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_chord_csv(track):
    """Test chord CSV."""
    csv_header = read_chords(track.path_chords).columns if track.path_chords else []
    assert list(csv_header) == ["start_meas", "end_meas", "chord"]


@pytest.mark.parametrize("track", TRACKS, ids=tr_ids)
def test_chord_annotations_sequence(track):
    """Test if CSV chord annotations can be parsed into a ChordSequence"""

    # all tracks link to the same chord annotations, so we only take one
    cs = ChordSequence.from_csv(track.path_chords)
    # all songs should have a chord in the second measure
    assert cs.get_chord_at(2.25).root is not None, f"Chord for song {track.song_id} not parsed correctly."
    for i in range(len(cs.bounds)-1):
        if cs.bounds[i,1] > cs.bounds[i+1,0]:
            print(track.song_id, i, cs.bounds[i,1], cs.bounds[i+1,0])
    assert np.all(cs.bounds[:-1,1] <= cs.bounds[1:,0]), f"Overlapping chord annotations for song {track.song_id}."
