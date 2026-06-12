"""Migrate a ChoraleBricks 1.0.1 release to the 1.1 data contract."""

from __future__ import annotations

import argparse
import bisect
import csv
import shutil
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

from choralebricks import spec
from choralebricks.spec import (
    format_measure_value,
    format_quarter_value,
    velocity_from_scalar,
)


DEFAULT_SOURCE = Path(r"C:\datasets\choralebricks\1.0.1")
DEFAULT_TARGET = Path(r"C:\datasets\choralebricks\1.1.0")
CHORALEBRICKS_VERSION = "1.1.0"
QUARTER_VALUE_FIELDS = {
    "duration_quarter",
    "quarter_note_offset",
    "quarter_note_BPM",
}

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
        writer.writerows(format_numeric_fields(row, fields) for row in rows)


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


def format_numeric_fields(
    row: dict[str, str],
    fields: list[str],
) -> dict[str, str]:
    formatted = format_measure_fields(row, fields)
    for field in QUARTER_VALUE_FIELDS:
        if field in fields and formatted.get(field, "") != "":
            formatted[field] = format_quarter_value(formatted[field])
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
        median = spec.f0_median_of_window(values[first:last])
        pitch_audio = str(spec.pitch_audio_from_f0_median(median))
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
