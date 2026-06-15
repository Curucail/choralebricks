"""Migrate a ChoraleBricks 1.0.1 release to the 1.1 data contract."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

import pandas as pd

from choralebricks.spec import format_measure_value, velocity_from_scalar


# Shared MEI music-theory lookups (used by the transposition helpers below).
MEI_NAMESPACE = "http://www.music-encoding.org/ns/mei"
MEI_TAG = f"{{{MEI_NAMESPACE}}}"
PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
ACCIDENTAL_TO_OFFSET = {"ff": -2, "f": -1, "n": 0, "s": 1, "ss": 2}
OFFSET_TO_ACCIDENTAL = {value: key for key, value in ACCIDENTAL_TO_OFFSET.items()}


def read_table(path: Path, sep: str) -> pd.DataFrame:
    """Read a CSV as all-string columns, preserving values and empty cells verbatim."""
    return pd.read_csv(
        path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    ).fillna("")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", index=False, lineterminator="\n", encoding="utf-8")


def add_durations(starts, durations) -> list[str]:
    """Exact ``start + duration`` per row, kept as strings."""
    return [str(Decimal(start) + Decimal(dur)) for start, dur in zip(starts, durations)]


def format_measure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Format measure positions and make integer end positions exclusive."""
    formatted = frame.copy()
    if "start_meas" in formatted.columns:
        formatted["start_meas"] = formatted["start_meas"].map(format_measure_value)
    if "end_meas" in formatted.columns:
        formatted["end_meas"] = formatted["end_meas"].map(
            lambda value: format_measure_value(value, exclusive_end=True)
        )
    return formatted


def sort_score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    part_order = {"S": 0, "A": 1, "T": 2, "B": 3}
    keyed = frame.assign(
        _meas=frame["start_meas"].map(Decimal),
        _part=frame["part"].map(part_order),
    ).sort_values(["_meas", "_part"], kind="stable")
    return keyed.drop(columns=["_meas", "_part"])


def migrate_annotation_pair(notes_path: Path, alignment_path: Path) -> int:
    notes = read_table(notes_path, ",")
    alignment = read_table(alignment_path, ";")
    if list(notes.columns) != ["TIME", "VALUE", "DURATION", "LEVEL", "LABEL"]:
        raise ValueError(f"{notes_path}: unexpected notes header {list(notes.columns)}.")
    if list(alignment.columns) != [
        "t_start", "f0_mean", "t_dur", "pitch_audio", "start_meas", "end_meas",
        "duration_quarterLength", "pitch_sheet_music", "pitchName", "timeSig", "part",
    ]:
        raise ValueError(
            f"{alignment_path}: unexpected alignment header {list(alignment.columns)}."
        )
    if len(notes) != len(alignment):
        raise ValueError(
            f"{notes_path}: {len(notes)} notes do not match "
            f"{len(alignment)} alignment rows."
        )

    velocity = notes["LEVEL"].map(velocity_from_scalar).to_numpy()
    # The 1.1 audio columns rename existing 1.0.1 values: keep the measured F0
    # (notes ``VALUE`` / alignment ``f0_mean``) and the audio pitch verbatim.
    pitch_audio = alignment["pitch_audio"].to_numpy()

    new_notes = pd.DataFrame({
        "start": notes["TIME"].to_numpy(),
        "end": add_durations(notes["TIME"], notes["DURATION"]),
        "duration": notes["DURATION"].to_numpy(),
        "pitch_audio": pitch_audio,
        "f0_note": notes["VALUE"].to_numpy(),
        "velocity": velocity,
        "label": notes["LABEL"].to_numpy(),
    })
    new_alignment = pd.DataFrame({
        "start_meas": alignment["start_meas"].to_numpy(),
        "end_meas": alignment["end_meas"].to_numpy(),
        "duration_quarter": alignment["duration_quarterLength"].to_numpy(),
        "pitch": alignment["pitch_sheet_music"].to_numpy(),
        "pitch_name": alignment["pitchName"].to_numpy(),
        "part": alignment["part"].to_numpy(),
        "time_sig": alignment["timeSig"].to_numpy(),
        "velocity": velocity,
        "start": alignment["t_start"].to_numpy(),
        "end": add_durations(alignment["t_start"], alignment["t_dur"]),
        "duration": alignment["t_dur"].to_numpy(),
        "pitch_audio": pitch_audio,
        "f0_note": alignment["f0_mean"].to_numpy(),
    })
    write_table(new_notes, notes_path)
    write_table(format_measure_columns(new_alignment), alignment_path)
    return len(new_notes)


def migrate_raw_f0(path: Path) -> int:
    raw = read_table(path, ",")
    if list(raw.columns) != ["TIME", "VALUE", "LABEL"]:
        raise ValueError(f"{path}: unexpected raw F0 header {list(raw.columns)}.")
    renamed = raw.rename(columns={"TIME": "t", "VALUE": "f0", "LABEL": "label"})
    write_table(renamed, path)
    return len(renamed)


def migrate_filled_f0(path: Path) -> int:
    filled = read_table(path, ",")
    if list(filled.columns) != ["t", "f0"]:
        raise ValueError(f"{path}: unexpected filled F0 header {list(filled.columns)}.")
    write_table(filled, path)
    return len(filled)


def migrate_semicolon_csv(path: Path) -> int:
    rows = read_table(path, ";")
    if rows.shape[1] == 1 and "," in rows.columns[0]:
        rows = read_table(path, ",")
    columns = list(rows.columns)
    score_1_0 = [
        "start_meas", "end_meas", "duration_quarterLength", "pitch", "pitchName",
        "timeSig", "articulation", "expression", "grace", "part", "midiChannel",
        "midiProgram", "volume", "pitchWritten", "pitchNameWritten",
        "quarternoteoffset", "quarterNoteBPM",
    ]
    score_1_1 = [
        "start_meas", "end_meas", "duration_quarter", "pitch", "pitch_name", "part",
        "time_sig", "articulation", "expression", "velocity", "quarter_note_offset",
        "quarter_note_BPM", "midiChannel",
    ]
    if columns == score_1_0:
        migrated = pd.DataFrame({
            "start_meas": rows["start_meas"].to_numpy(),
            "end_meas": rows["end_meas"].to_numpy(),
            "duration_quarter": rows["duration_quarterLength"].to_numpy(),
            "pitch": rows["pitch"].to_numpy(),
            "pitch_name": rows["pitchName"].to_numpy(),
            "part": rows["part"].to_numpy(),
            "time_sig": rows["timeSig"].to_numpy(),
            "articulation": rows["articulation"].to_numpy(),
            "expression": rows["expression"].to_numpy(),
            "velocity": rows["volume"].map(velocity_from_scalar).to_numpy(),
            "quarter_note_offset": rows["quarternoteoffset"].to_numpy(),
            "quarter_note_BPM": rows["quarterNoteBPM"].to_numpy(),
            "midiChannel": rows["midiChannel"].to_numpy(),
        })
        write_table(sort_score_rows(format_measure_columns(migrated)), path)
    elif columns == score_1_1:
        write_table(sort_score_rows(format_measure_columns(rows)), path)
    else:
        write_table(format_measure_columns(rows), path)
    return len(rows)


def clean_f0_annotations(
    notes_path: Path,
    raw_path: Path,
    filled_path: Path,
) -> tuple[int, int]:
    notes = read_table(notes_path, ";")
    raw = read_table(raw_path, ";")
    filled = read_table(filled_path, ";")
    if list(notes.columns) != [
        "start", "end", "duration", "pitch_audio", "f0_note", "velocity", "label",
    ]:
        raise ValueError(f"{notes_path}: unexpected notes header {list(notes.columns)}.")
    if list(raw.columns) != ["t", "f0", "label"]:
        raise ValueError(f"{raw_path}: unexpected raw F0 header {list(raw.columns)}.")
    if list(filled.columns) != ["t", "f0"]:
        raise ValueError(f"{filled_path}: unexpected filled F0 header {list(filled.columns)}.")

    starts = notes["start"].astype(float).to_numpy()
    ends = starts + notes["duration"].astype(float).to_numpy()

    def in_any_note(times: pd.Series) -> pd.Series:
        flags = [bool(((starts <= t) & (t <= ends)).any()) for t in times.astype(float)]
        return pd.Series(flags, index=times.index)

    raw_keep = (raw["f0"].astype(float) != 0) & in_any_note(raw["t"])
    removed = int((~raw_keep).sum())
    write_table(raw[raw_keep], raw_path)

    outside = (filled["f0"].astype(float) != 0) & ~in_any_note(filled["t"])
    zeroed = int(outside.sum())
    filled.loc[outside, "f0"] = "0.0"
    write_table(filled, filled_path)
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
    previous_diatonic = {
        "c": "b", "d": "c", "e": "d", "f": "e", "g": "f", "a": "g", "b": "a",
    }
    pname = previous_diatonic[source_pname]
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
    source_key_offsets = {"f": 1, "c": 1}
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
        source_midi = mei_note_midi(note, source_key_offsets)
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


def migrate_release(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source release not found: {source}.")
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}.")

    shutil.copytree(source, target, copy_function=shutil.copy2)
    try:
        # There is a stale file in v1.0.1 which was a overseen copy/rename artifact of 03_bcl_notes.csv in v1.0.1
        # It will be removed for v1.1.0
        orphan = target / (
            "01_AudioAndAnnotations/Gesius_DuFriedensfuerstHerrJesuChrist"
            "/annotations/03_eh_notes.csv"
        )
        if not orphan.is_file():
            raise FileNotFoundError(f"Expected stale orphan file not found: {orphan}.")
        orphan.unlink()

        audio_root = target / "01_AudioAndAnnotations"
        for component in (audio_root, target / "02_ConductingVideos"):
            # Mandatory VERSION tagging starting from v1.1.0
            (component / "VERSION").write_text("1.1.0\n", encoding="utf-8")

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
                # Only rename columns, 
                migrate_raw_f0(path)
        for path in sorted(audio_root.rglob("*_f0_filled.csv")):
            # Only rename columns
            migrate_filled_f0(path)

        for notes_path in notes_paths:
            stem = notes_path.name.removesuffix("_notes.csv")
            raw_path = notes_path.with_name(f"{stem}_f0.csv")
            clean_f0_annotations(
                notes_path,
                raw_path,
                notes_path.with_name(f"{stem}_f0_filled.csv"),
            )

        # The piece Crueger_AufAufMeinHerzMitFreuden is notated in D major but should be in C major to match the rest of the dataset. 
        # We transpose it down 2 semitones by editing the MEI file.
        crueger = "Crueger_AufAufMeinHerzMitFreuden"
        transpose_crueger_mei(audio_root / crueger / f"{crueger}.mei")

        migrated_paths = set(notes_paths) | set(alignment_paths)
        migrated_paths.update(audio_root.rglob("*_f0.csv"))
        migrated_paths.update(audio_root.rglob("*_f0_filled.csv"))
        for path in sorted(target.rglob("*.csv")):
            if path not in migrated_paths:
                # All csv files will be semicolon separated starting from v1.1.0
                migrate_semicolon_csv(path)

    except Exception:
        shutil.rmtree(target)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(r"C:\datasets\choralebricks\1.0.1")
    )
    parser.add_argument(
        "--target", type=Path, default=Path(r"C:\datasets\choralebricks\1.1.0")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_release(args.source, args.target)
    print(f"Migrated ChoraleBricks {args.source} to {args.target}.")


if __name__ == "__main__":
    main()
