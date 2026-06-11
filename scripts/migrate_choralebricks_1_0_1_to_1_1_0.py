"""Migrate a ChoraleBricks 1.0.1 release to the 1.1 data contract."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import shutil
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


DEFAULT_SOURCE = Path(r"C:\datasets\choralebricks\1.0.1")
DEFAULT_TARGET = Path(r"C:\datasets\choralebricks\1.1.0")
CHORALEBRICKS_VERSION = "1.1.0"
RELEASE_DATE = "10.06.2026"
MEASURE_STEP = Decimal("0.001")
F0_MEDIAN_STEP = Decimal("0.001")
A4_HZ = 442.0
VELOCITY_MAX = 127

ALIGNMENT_1_0_FIELDS = [
    "t_start",
    "f0_mean",
    "t_dur",
    "pitch_audio",
    "start_meas",
    "end_meas",
    "duration_quarterLength",
    "pitch_sheet_music",
    "pitchName",
    "timeSig",
    "part",
]
ALIGNMENT_1_1_FIELDS = [
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
NOTES_1_0_FIELDS = ["TIME", "VALUE", "DURATION", "LEVEL", "LABEL"]
NOTES_1_1_FIELDS = [
    "start",
    "end",
    "duration",
    "pitch_audio",
    "f0_median",
    "velocity",
    "label",
]
RAW_F0_1_0_FIELDS = ["TIME", "VALUE", "LABEL"]
RAW_F0_1_1_FIELDS = ["t", "f0", "label"]
FILLED_F0_FIELDS = ["t", "f0"]
SCORE_1_0_FIELDS = [
    "start_meas",
    "end_meas",
    "duration_quarterLength",
    "pitch",
    "pitchName",
    "timeSig",
    "articulation",
    "expression",
    "grace",
    "part",
    "midiChannel",
    "midiProgram",
    "volume",
    "pitchWritten",
    "pitchNameWritten",
    "quarternoteoffset",
    "quarterNoteBPM",
]
SCORE_1_1_FIELDS = [
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
ORPHAN_NOTES_RELATIVE_PATH = Path(
    "01_AudioAndAnnotations"
    "/Gesius_DuFriedensfuerstHerrJesuChrist"
    "/annotations"
    "/03_eh_notes.csv"
)
CRUEGER_SONG = "Crueger_AufAufMeinHerzMitFreuden"
CRUEGER_MEI_NAME = f"{CRUEGER_SONG}.mei"
MEI_NAMESPACE = "http://www.music-encoding.org/ns/mei"
MEI_TAG = f"{{{MEI_NAMESPACE}}}"
PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
PREVIOUS_DIATONIC = {
    "c": "b",
    "d": "c",
    "e": "d",
    "f": "e",
    "g": "f",
    "a": "g",
    "b": "a",
}
ACCIDENTAL_TO_OFFSET = {"ff": -2, "f": -1, "n": 0, "s": 1, "ss": 2}
OFFSET_TO_ACCIDENTAL = {value: key for key, value in ACCIDENTAL_TO_OFFSET.items()}
SOURCE_KEY_OFFSETS = {"f": 1, "c": 1}


def read_dict_rows(
    path: Path,
    delimiter: str,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def write_dict_rows(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(format_measure_fields(row, fields) for row in rows)


def normalize_row(row: dict[str, str], fields: list[str]) -> dict[str, str]:
    return {field: row.get(field) or "" for field in fields}


def sort_score_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            Decimal(row["start_meas"]),
            SCORE_PART_ORDER[row["part"]],
        ),
    )


def format_measure_value(value: str, *, exclusive_end: bool = False) -> str:
    numeric_value = Decimal(value)
    if exclusive_end and numeric_value == numeric_value.to_integral_value():
        numeric_value -= MEASURE_STEP
    sign = "-" if numeric_value < 0 else ""
    return f"{sign}{abs(numeric_value):07.3f}"


def format_measure_fields(
    row: dict[str, str],
    fields: list[str],
) -> dict[str, str]:
    formatted = dict(row)
    if "start_meas" in fields and formatted.get("start_meas", "") != "":
        formatted["start_meas"] = format_measure_value(formatted["start_meas"])
    if "end_meas" in fields and formatted.get("end_meas", "") != "":
        formatted["end_meas"] = format_measure_value(
            formatted["end_meas"],
            exclusive_end=True,
        )
    return formatted


def note_intervals(notes: list[dict[str, str]]) -> list[tuple[float, float]]:
    return [
        (
            float(note["start"]),
            float(note["start"]) + float(note["duration"]),
        )
        for note in notes
    ]


def time_in_note_intervals(
    time: float,
    intervals: list[tuple[float, float]],
) -> bool:
    return any(start <= time <= end for start, end in intervals)


def format_f0_median(values: list[Decimal]) -> str:
    if not values:
        raise ValueError("Cannot calculate an F0 median from an empty window.")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / Decimal(2)
    return f"{median.quantize(F0_MEDIAN_STEP, rounding=ROUND_HALF_UP):.3f}"


def pitch_audio_from_f0_median(f0_median: str) -> str:
    midi = Decimal(12) * Decimal(math.log2(float(f0_median) / A4_HZ)) + Decimal(69)
    return str(int(midi.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


def velocity_from_scalar(value: str) -> str:
    """Convert a [0, 1] loudness scalar (e.g. volume/level) to MIDI velocity."""
    velocity = int(
        (Decimal(value) * VELOCITY_MAX).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    return str(max(0, min(VELOCITY_MAX, velocity)))


def recompute_note_f0_medians(
    notes_path: Path,
    alignment_path: Path,
    raw_path: Path,
) -> int:
    notes_header, notes = read_dict_rows(notes_path, ";")
    alignment_header, alignment = read_dict_rows(alignment_path, ";")
    raw_header, raw = read_dict_rows(raw_path, ";")
    if notes_header != NOTES_1_1_FIELDS:
        raise ValueError(f"{notes_path}: unexpected notes header {notes_header}.")
    if alignment_header != ALIGNMENT_1_1_FIELDS:
        raise ValueError(
            f"{alignment_path}: unexpected alignment header {alignment_header}."
        )
    if raw_header != RAW_F0_1_1_FIELDS:
        raise ValueError(f"{raw_path}: unexpected raw F0 header {raw_header}.")
    if len(notes) != len(alignment):
        raise ValueError(f"{notes_path}: alignment row count mismatch.")

    timestamps = [Decimal(row["t"]) for row in raw]
    if timestamps != sorted(timestamps):
        raise ValueError(f"{raw_path}: raw F0 timestamps are not sorted.")
    values = [Decimal(row["f0"]) for row in raw]

    for row_number, (note, aligned) in enumerate(
        zip(notes, alignment),
        start=2,
    ):
        start = Decimal(note["start"])
        end = start + Decimal(note["duration"])
        first = bisect.bisect_left(timestamps, start)
        last = bisect.bisect_right(timestamps, end)
        if first == last:
            raise ValueError(
                f"{notes_path}:{row_number}: note interval contains no raw F0."
            )
        median = format_f0_median(values[first:last])
        pitch_audio = pitch_audio_from_f0_median(median)
        note["f0_median"] = median
        aligned["f0_median"] = median
        note["pitch_audio"] = pitch_audio
        aligned["pitch_audio"] = pitch_audio

    write_dict_rows(notes_path, NOTES_1_1_FIELDS, notes)
    write_dict_rows(alignment_path, ALIGNMENT_1_1_FIELDS, alignment)
    return len(notes)


def clean_f0_annotations(
    notes_path: Path,
    raw_path: Path,
    filled_path: Path,
) -> tuple[int, int]:
    notes_header, notes = read_dict_rows(notes_path, ";")
    raw_header, raw = read_dict_rows(raw_path, ";")
    filled_header, filled = read_dict_rows(filled_path, ";")
    if notes_header != NOTES_1_1_FIELDS:
        raise ValueError(f"{notes_path}: unexpected notes header {notes_header}.")
    if raw_header != RAW_F0_1_1_FIELDS:
        raise ValueError(f"{raw_path}: unexpected raw F0 header {raw_header}.")
    if filled_header != FILLED_F0_FIELDS:
        raise ValueError(f"{filled_path}: unexpected filled F0 header {filled_header}.")

    intervals = note_intervals(notes)
    cleaned_raw = [
        row
        for row in raw
        if float(row["f0"]) != 0
        and time_in_note_intervals(float(row["t"]), intervals)
    ]
    cleaned_filled = []
    zeroed = 0
    for row in filled:
        cleaned = normalize_row(row, FILLED_F0_FIELDS)
        if (
            float(cleaned["f0"]) != 0
            and not time_in_note_intervals(float(cleaned["t"]), intervals)
        ):
            cleaned["f0"] = "0.0"
            zeroed += 1
        cleaned_filled.append(cleaned)

    removed = len(raw) - len(cleaned_raw)
    write_dict_rows(raw_path, RAW_F0_1_1_FIELDS, cleaned_raw)
    write_dict_rows(filled_path, FILLED_F0_FIELDS, cleaned_filled)
    return removed, zeroed


def mei_note_midi(note: ET.Element, key_offsets: dict[str, int]) -> int:
    pname = note.attrib["pname"]
    accidental = key_offsets.get(pname, 0)
    for child in note:
        if child.tag == f"{MEI_TAG}accid":
            encoded = child.attrib.get("accid.ges") or child.attrib.get("accid")
            if encoded:
                accidental = ACCIDENTAL_TO_OFFSET[encoded]
    return (int(note.attrib["oct"]) + 1) * 12 + PITCH_CLASSES[pname] + accidental


def target_spelling(source_pname: str, target_midi: int) -> tuple[str, int, int]:
    pname = PREVIOUS_DIATONIC[source_pname]
    candidates = []
    for octave in range(0, 9):
        natural_midi = (octave + 1) * 12 + PITCH_CLASSES[pname]
        alteration = target_midi - natural_midi
        if alteration in OFFSET_TO_ACCIDENTAL:
            candidates.append((abs(alteration), pname, octave, alteration))
    if not candidates:
        raise ValueError(f"Cannot spell MIDI pitch {target_midi} from {source_pname}.")
    _, pname, octave, alteration = min(candidates)
    return pname, octave, alteration


def transpose_crueger_mei(mei_path: Path) -> int:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(mei_path, parser=parser)
    root = tree.getroot()
    score_defs = [
        score_def
        for score_def in root.iter(f"{MEI_TAG}scoreDef")
        if "key.pname" in score_def.attrib or "keysig" in score_def.attrib
    ]
    if len(score_defs) != 1:
        raise ValueError(
            f"{mei_path}: expected one keyed scoreDef, found {len(score_defs)}."
        )
    score_def = score_defs[0]
    if (
        score_def.attrib.get("keysig") != "2s"
        or score_def.attrib.get("key.pname") != "d"
        or score_def.attrib.get("key.mode") != "major"
    ):
        raise ValueError(f"{mei_path}: expected a D-major score definition.")
    score_def.attrib.pop("keysig")
    score_def.attrib["key.pname"] = "c"

    changed = 0
    for note in root.iter(f"{MEI_TAG}note"):
        if "pname" not in note.attrib or "oct" not in note.attrib:
            continue
        source_midi = mei_note_midi(note, SOURCE_KEY_OFFSETS)
        target_midi = source_midi - 2
        pname, octave, alteration = target_spelling(
            note.attrib["pname"],
            target_midi,
        )
        note.attrib["pname"] = pname
        note.attrib["oct"] = str(octave)

        accidental_nodes = [
            child for child in note if child.tag == f"{MEI_TAG}accid"
        ]
        if alteration == 0:
            for child in accidental_nodes:
                note.remove(child)
        else:
            accidental = (
                accidental_nodes[0]
                if accidental_nodes
                else ET.SubElement(note, f"{MEI_TAG}accid")
            )
            accidental.attrib.clear()
            accidental.attrib["accid.ges"] = OFFSET_TO_ACCIDENTAL[alteration]
            for child in accidental_nodes[1:]:
                note.remove(child)

        if mei_note_midi(note, {}) != target_midi:
            raise RuntimeError(f"{mei_path}: failed to transpose a note.")
        changed += 1

    ET.register_namespace("", MEI_NAMESPACE)
    tree.write(mei_path, encoding="utf-8", xml_declaration=True)
    return changed


def migrate_annotation_pair(notes_path: Path, alignment_path: Path) -> int:
    notes_header, old_notes = read_dict_rows(notes_path, ",")
    alignment_header, old_alignment = read_dict_rows(alignment_path, ";")
    if notes_header != NOTES_1_0_FIELDS:
        raise ValueError(f"{notes_path}: unexpected notes header {notes_header}.")
    if alignment_header != ALIGNMENT_1_0_FIELDS:
        raise ValueError(
            f"{alignment_path}: unexpected alignment header {alignment_header}."
        )
    if len(old_notes) != len(old_alignment):
        raise ValueError(
            f"{notes_path}: {len(old_notes)} notes do not match "
            f"{len(old_alignment)} alignment rows."
        )

    new_notes: list[dict[str, str]] = []
    new_alignment: list[dict[str, str]] = []
    for note, alignment in zip(old_notes, old_alignment):
        velocity = velocity_from_scalar(note["LEVEL"])
        note_end = str(Decimal(note["TIME"]) + Decimal(note["DURATION"]))
        alignment_end = str(
            Decimal(alignment["t_start"]) + Decimal(alignment["t_dur"])
        )
        new_notes.append(
            {
                "velocity": velocity,
                "start": note["TIME"],
                "end": note_end,
                "duration": note["DURATION"],
                "pitch_audio": "",
                "f0_median": "",
                "label": note.get("LABEL") or "",
            }
        )
        new_alignment.append(
            {
                "start_meas": alignment["start_meas"],
                "end_meas": alignment["end_meas"],
                "duration_quarter": alignment["duration_quarterLength"],
                "pitch": alignment["pitch_sheet_music"],
                "pitch_name": alignment["pitchName"],
                "part": alignment["part"],
                "time_sig": alignment["timeSig"],
                "velocity": velocity,
                "start": alignment["t_start"],
                "end": alignment_end,
                "duration": alignment["t_dur"],
                "pitch_audio": "",
                "f0_median": "",
            }
        )

    write_dict_rows(notes_path, NOTES_1_1_FIELDS, new_notes)
    write_dict_rows(alignment_path, ALIGNMENT_1_1_FIELDS, new_alignment)
    return len(new_notes)


def migrate_raw_f0(path: Path) -> int:
    header, rows = read_dict_rows(path, ",")
    if header != RAW_F0_1_0_FIELDS:
        raise ValueError(f"{path}: unexpected raw F0 header {header}.")
    migrated = [
        {
            "t": row["TIME"],
            "f0": row["VALUE"],
            "label": row.get("LABEL") or "",
        }
        for row in rows
    ]
    write_dict_rows(path, RAW_F0_1_1_FIELDS, migrated)
    return len(migrated)


def migrate_semicolon_csv(path: Path) -> int:
    header, rows = read_dict_rows(path, ";")
    if len(header) == 1 and "," in header[0]:
        header, rows = read_dict_rows(path, ",")
    if header == SCORE_1_0_FIELDS:
        migrated = []
        for row in rows:
            old = normalize_row(row, SCORE_1_0_FIELDS)
            migrated.append(
                {
                    "start_meas": old["start_meas"],
                    "end_meas": old["end_meas"],
                    "duration_quarter": old["duration_quarterLength"],
                    "pitch": old["pitch"],
                    "pitch_name": old["pitchName"],
                    "part": old["part"],
                    "time_sig": old["timeSig"],
                    "articulation": old["articulation"],
                    "expression": old["expression"],
                    "velocity": velocity_from_scalar(old["volume"]),
                    "quarter_note_offset": old["quarternoteoffset"],
                    "quarter_note_BPM": old["quarterNoteBPM"],
                    "midiChannel": old["midiChannel"],
                }
            )
        write_dict_rows(path, SCORE_1_1_FIELDS, sort_score_rows(migrated))
    elif header == SCORE_1_1_FIELDS:
        normalized = [normalize_row(row, SCORE_1_1_FIELDS) for row in rows]
        write_dict_rows(path, SCORE_1_1_FIELDS, sort_score_rows(normalized))
    else:
        write_dict_rows(
            path,
            header,
            [normalize_row(row, header) for row in rows],
        )
    return len(rows)


def migrate_filled_f0(path: Path) -> int:
    header, rows = read_dict_rows(path, ",")
    if header != FILLED_F0_FIELDS:
        raise ValueError(f"{path}: unexpected filled F0 header {header}.")
    write_dict_rows(
        path,
        FILLED_F0_FIELDS,
        [normalize_row(row, FILLED_F0_FIELDS) for row in rows],
    )
    return len(rows)


def update_changelog(path: Path, version: str) -> None:
    old = path.read_text(encoding="utf-8")
    old = old.replace(
        "Vulpius_DieHelleSonnLeuchtJetztHerfuer siwtched with "
        "Vulpius_ChristusDerIstMeinLeben",
        "Vulpius_DieHelleSonnLeuchtJetztHerfuer switched with "
        "Vulpius_ChristusDerIstMeinLeben",
    )
    section = (
        f"## {version}\n\n"
        f"- Release date: {RELEASE_DATE}\n"
        "- Breaking release with no backwards compatibility for version 1.0.x\n"
        "- Added VERSION file\n"
        "- Changed all CSV files to semicolon separation\n"
        "- Renamed and reordered alignment, notes, and score csv columns\n"
        "- Sorted score rows by start_meas and SATB part order\n"
        "- Dropped uninformative score columns (grace, midiProgram, pitchWritten, "
        "pitchNameWritten)\n"
        "- Added a seconds-based end column (start + duration) to the alignment "
        "and notes csv files\n"
        "- Added a velocity column to the alignment csv files (taken from the "
        "paired note) and stored velocity as an integer in [0, 127], converted "
        "from the former score volume and note level scalars via "
        "round(scalar * 127)\n"
        "- Recomputed note and alignment f0_median from raw F0 export in each "
        "closed note interval, rounded to three decimals.\n"
        "- Recomputed note and alignment pitch_audio from f0_median as "
        "round(12*log2(f0_median / 442) + 69) (A4 = 442 Hz).\n"
        "- Removed stale file "
        "Gesius_DuFriedensfuerstHerrJesuChrist/annotations/03_eh_notes.csv\n"
        "- Restricted non-zero (voiced) F0 values to be only allowed during note events\n"
        "- Raw F0 csv rows outside note events are removed; filled F0 csv rows "
        "outside note events are retained and their values are written as 0.0\n"
        "- Transposed Crueger_AufAufMeinHerzMitFreuden.mei down by two semitones "
        "to match the rest of the assets\n"
        "- Standardized measure positions to fixed-width three-decimal formatting\n"
        "- Made end measure annotations with exact integer end positions exclusive, "
        "e.g. 005.000 --> 004.999\n\n"
        "### CSV column migration v1.0.1 -> v1.1.0\n\n"
        "| CSV type | 1.0.1 columns | 1.1.0 columns |\n"
        "| --- | --- | --- |\n"
        "| Alignment | t_start, f0_mean, t_dur, pitch_audio, start_meas, "
        "end_meas, duration_quarterLength, pitch_sheet_music, pitchName, "
        "timeSig, part | start_meas, end_meas, duration_quarter, pitch, "
        "pitch_name, part, time_sig, velocity, start, end, duration, "
        "pitch_audio, f0_median |\n"
        "| Notes | TIME, VALUE, DURATION, LEVEL, LABEL | start, end, duration, "
        "pitch_audio, f0_median, velocity, label |\n"
        "| Raw F0 | TIME, VALUE, LABEL | t, f0, label |\n"
        "| Filled F0 | t, f0 | t, f0 |\n"
        "| Score | start_meas, end_meas, duration_quarterLength, pitch, "
        "pitchName, timeSig, articulation, expression, grace, part, "
        "midiChannel, midiProgram, volume, pitchWritten, pitchNameWritten, "
        "quarternoteoffset, quarterNoteBPM | start_meas, end_meas, "
        "duration_quarter, pitch, pitch_name, part, time_sig, articulation, "
        "expression, velocity, quarter_note_offset, quarter_note_BPM, "
        "midiChannel |\n"
        "| Chords | start_meas, end_meas, chord | "
        "start_meas, end_meas, chord |\n\n"
        "Notes to the csv changes:\n\n"
        "- Alignment csv files: `f0_mean` is now called `f0_median` since it "
        "represents the median value, not the mean, and is recomputed from raw "
        "F0 csv inside the closed note interval. Renamed `t_start` -> `start`, "
        "`t_dur` -> `duration`, `duration_quarterLength` -> `duration_quarter`, "
        "`pitch_sheet_music` -> `pitch`, `pitchName` -> `pitch_name`, `timeSig` "
        "-> `time_sig`. Added `end` (= `start` + `duration`) and `velocity` "
        "(from the paired note).\n"
        "- *_notes.csv: `TIME` -> `start`, `DURATION` -> `duration`, `LEVEL` -> "
        "`velocity` (integer in [0, 127]), and `LABEL` -> `label`. Added `end` "
        "(= `start` + `duration`) and `pitch_audio`, derived from `f0_median` "
        "(A4 = 442 Hz). `f0_median` replaces `VALUE` and is recomputed from raw "
        "F0 csv inside the closed note interval.\n"
        "- Raw F0: `TIME` -> `t`, `VALUE` -> `f0`, and `LABEL` -> `label`.\n"
        "- Score: `duration_quarterLength` -> `duration_quarter`, `pitchName` "
        "-> `pitch_name`, `timeSig` -> `time_sig`, `volume` -> `velocity` "
        "(integer in [0, 127]), `quarternoteoffset` -> `quarter_note_offset`, "
        "`quarterNoteBPM` -> `quarter_note_BPM`. Dropped `grace`, `midiProgram`, "
        "`pitchWritten`, and `pitchNameWritten`.\n\n"
    )
    heading = "# ChoraleBricks Changelog\n\n"
    if not old.startswith(heading):
        raise ValueError(f"{path}: unexpected changelog heading.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write((heading + section + old[len(heading):]).rstrip() + "\n")


def migrate_release(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source release not found: {source}.")
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}.")

    version = CHORALEBRICKS_VERSION

    shutil.copytree(source, target, copy_function=shutil.copy2)
    try:
        orphan = target / ORPHAN_NOTES_RELATIVE_PATH
        if not orphan.is_file():
            raise FileNotFoundError(f"Expected stale orphan file not found: {orphan}.")
        orphan.unlink()

        audio_root = target / "01_AudioAndAnnotations"
        for component in (audio_root, target / "02_ConductingVideos"):
            (component / "VERSION").write_text(f"{version}\n", encoding="utf-8")

        alignment_paths = sorted(
            path
            for path in audio_root.rglob("*.csv")
            if path.parent.name == "alignments"
        )
        notes_paths = sorted(audio_root.rglob("*_notes.csv"))
        if len(alignment_paths) != 193 or len(notes_paths) != 193:
            raise RuntimeError(
                f"Expected 193 alignments and notes, found "
                f"{len(alignment_paths)} and {len(notes_paths)}."
            )

        for notes_path in notes_paths:
            alignment_path = (
                notes_path.parent.parent
                / "alignments"
                / notes_path.name.replace("_notes.csv", ".csv")
            )
            migrate_annotation_pair(notes_path, alignment_path)

        for path in sorted(audio_root.rglob("*_f0.csv")):
            if not path.name.endswith("_f0_filled.csv"):
                migrate_raw_f0(path)
        for path in sorted(audio_root.rglob("*_f0_filled.csv")):
            migrate_filled_f0(path)

        for notes_path in notes_paths:
            stem = notes_path.name.removesuffix("_notes.csv")
            raw_path = notes_path.with_name(f"{stem}_f0.csv")
            clean_f0_annotations(
                notes_path,
                raw_path,
                notes_path.with_name(f"{stem}_f0_filled.csv"),
            )
            recompute_note_f0_medians(
                notes_path,
                notes_path.parent.parent
                / "alignments"
                / notes_path.name.replace("_notes.csv", ".csv"),
                raw_path,
            )

        transpose_crueger_mei(
            audio_root / CRUEGER_SONG / CRUEGER_MEI_NAME
        )

        migrated_paths = set(notes_paths) | set(alignment_paths)
        migrated_paths.update(audio_root.rglob("*_f0.csv"))
        migrated_paths.update(audio_root.rglob("*_f0_filled.csv"))
        for path in sorted(target.rglob("*.csv")):
            if path not in migrated_paths:
                migrate_semicolon_csv(path)

        update_changelog(target / "CHANGELOG.md", version)
    except Exception:
        shutil.rmtree(target)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_release(args.source, args.target)
    print(f"Migrated ChoraleBricks {args.source} to {args.target}.")


if __name__ == "__main__":
    main()
