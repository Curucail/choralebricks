"""Migrate a ChoraleBricks 1.0.1 release to the 1.1 data contract."""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

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


def write_legacy_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=",", index=False, lineterminator="\n", encoding="utf-8")


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


def f0_to_midi(f0_hz: float, a4: float = 442.0) -> float:
    return 69.0 + 12.0 * math.log2(f0_hz / a4) if f0_hz > 0.0 else 0.0


def migrate_annotation_pair(
    notes_path: Path,
    alignment_path: Path,
    raw_f0_cleaned: pd.DataFrame,
) -> pd.DataFrame:
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

    raw_t = raw_f0_cleaned["t"].astype(float)
    raw_f0 = raw_f0_cleaned["f0"].astype(float)
    note_starts = notes["TIME"].astype(float)
    note_ends = note_starts + notes["DURATION"].astype(float)

    def note_f0_median(start: float, end: float) -> float:
        in_window = raw_f0[(raw_t >= start) & (raw_t <= end) & (raw_f0 != 0.0)]
        return float(in_window.median()) if len(in_window) > 0 else 0.0

    f0_medians = [note_f0_median(s, e) for s, e in zip(note_starts, note_ends)]
    f0_medians_str = [str(f) for f in f0_medians]
    pitch_audio = [str(round(f0_to_midi(f))) for f in f0_medians]

    velocity = notes["LEVEL"].map(velocity_from_scalar).to_numpy()

    new_notes = pd.DataFrame({
        "start_sec": notes["TIME"].to_numpy(),
        "end_sec": add_durations(notes["TIME"], notes["DURATION"]),
        "duration_sec": notes["DURATION"].to_numpy(),
        "pitch_audio": pitch_audio,
        "f0_median": f0_medians_str,
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
        "start_sec": alignment["t_start"].to_numpy(),
        "end_sec": add_durations(alignment["t_start"], alignment["t_dur"]),
        "duration_sec": alignment["t_dur"].to_numpy(),
        "pitch_audio": pitch_audio,
        "f0_median": f0_medians_str,
    })
    write_table(new_notes, notes_path)
    write_table(format_measure_columns(new_alignment), alignment_path)
    return new_notes


def migrate_piece_csv(path: Path) -> int:
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
) -> pd.DataFrame:
    """Remove raw F0 values outside note events, zero filled F0 outside note events.

    Returns the cleaned raw F0 DataFrame (columns: t, f0, label) for downstream use.
    """
    notes = read_table(notes_path, ",")
    if list(notes.columns) != ["TIME", "VALUE", "DURATION", "LEVEL", "LABEL"]:
        raise ValueError(f"{notes_path}: unexpected notes header {list(notes.columns)}.")
    raw = read_table(raw_path, ",")
    if list(raw.columns) != ["TIME", "VALUE", "LABEL"]:
        raise ValueError(f"{raw_path}: unexpected raw F0 header {list(raw.columns)}.")
    raw = raw.rename(columns={"TIME": "t", "VALUE": "f0", "LABEL": "label"})

    filled = read_table(filled_path, ",")
    if list(filled.columns) != ["t", "f0"]:
        raise ValueError(f"{filled_path}: unexpected filled F0 header {list(filled.columns)}.")

    starts = notes["TIME"].astype(float).to_numpy()
    ends = starts + notes["DURATION"].astype(float).to_numpy()

    def in_any_note(times: pd.Series) -> pd.Series:
        flags = [bool(((starts <= t) & (t <= ends)).any()) for t in times.astype(float)]
        return pd.Series(flags, index=times.index)

    raw_keep = (raw["f0"].astype(float) != 0) & in_any_note(raw["t"])
    cleaned_raw = raw[raw_keep].reset_index(drop=True)
    write_table(cleaned_raw, raw_path)

    outside = (filled["f0"].astype(float) != 0) & ~in_any_note(filled["t"])
    filled.loc[outside, "f0"] = "0.0"
    write_table(filled, filled_path)
    return cleaned_raw


def read_svl_pitch_track(svl_path: Path) -> pd.DataFrame:
    """Read a Sonic Visualiser sparse pitch track into the legacy raw-F0 schema."""
    root = ET.parse(svl_path).getroot()
    model = root.find(".//model")
    if model is None:
        raise ValueError(f"{svl_path}: missing <model> definition.")

    sample_rate = float(model.attrib["sampleRate"])
    points = root.findall(".//point")
    if not points:
        raise ValueError(f"{svl_path}: no pitch points found.")

    raw = pd.DataFrame({
        "TIME": [float(point.attrib["frame"]) / sample_rate for point in points],
        "VALUE": [float(point.attrib["value"]) for point in points],
        "LABEL": [point.attrib.get("label", "") for point in points],
    })
    return raw


def interpolate_f0_to_template(
    raw_pitch: pd.DataFrame,
    filled_template_path: Path,
) -> pd.DataFrame:
    """Project sparse raw F0 onto the copied filled-F0 time axis."""
    filled_template = read_table(filled_template_path, ",")
    if list(filled_template.columns) != ["t", "f0"]:
        raise ValueError(
            f"{filled_template_path}: unexpected filled F0 header "
            f"{list(filled_template.columns)}."
        )

    target_times = filled_template["t"].astype(float).to_numpy()
    f0_orig = raw_pitch[["TIME", "VALUE"]].astype(float).to_numpy()

    _, unique_idx = np.unique(f0_orig[:, 0], return_index=True)
    f0_orig = f0_orig[np.sort(unique_idx), :]

    f0_new = np.zeros_like(target_times)
    if len(f0_orig) < 2:
        return pd.DataFrame({"t": filled_template["t"].to_numpy(), "f0": f0_new})

    dt = np.diff(f0_orig[:, 0])
    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        return pd.DataFrame({"t": filled_template["t"].to_numpy(), "f0": f0_new})
    dt_max_allowed = float(np.min(positive_dt)) * 1.0001

    idxs = np.searchsorted(f0_orig[:, 0], target_times) - 1
    idxs[idxs < 0] = 0
    idxs[idxs >= len(f0_orig) - 1] = len(f0_orig) - 2

    t_idx = np.take(f0_orig[:, 0], idxs)
    t_diff = target_times - t_idx
    mask = (t_diff >= 0) & (t_diff <= dt_max_allowed)

    h = t_diff[mask] / np.take(dt, idxs[mask])
    f0_new[mask] = (
        (1 - h) * np.take(f0_orig[:, 1], idxs[mask])
        + h * np.take(f0_orig[:, 1], idxs[mask] + 1)
    )
    return pd.DataFrame({"t": filled_template["t"].to_numpy(), "f0": f0_new})


def replace_known_bad_04_bar_f0_annotations(target: Path) -> None:
    """Replace a known copied F0 artifact with the checked-in corrected pitch track."""
    annotations_dir = (
        target
        / "01_AudioAndAnnotations"
        / "Vulpius_ChristusDerIstMeinLeben"
        / "annotations"
    )
    raw_path = annotations_dir / "04_bar_f0.csv"
    filled_path = annotations_dir / "04_bar_f0_filled.csv"
    replacement_svl = Path(__file__).resolve().parent / "04_bar.svl"

    if not replacement_svl.is_file():
        raise FileNotFoundError(
            f"Expected replacement Sonic Visualiser file not found: {replacement_svl}."
        )

    raw_pitch = read_svl_pitch_track(replacement_svl)
    filled_pitch = interpolate_f0_to_template(raw_pitch, filled_path)
    write_legacy_table(raw_pitch, raw_path)
    write_legacy_table(filled_pitch, filled_path)


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

        replace_known_bad_04_bar_f0_annotations(target)

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

        for notes_path in tqdm.tqdm(notes_paths, desc="Migrating ChoraleBricks v1.0.1 to v1.1.0"):
            alignment_path = (
                notes_path.parent.parent
                / "alignments"
                / notes_path.name.replace("_notes.csv", ".csv")
            )
            stem = notes_path.name.removesuffix("_notes.csv")
            raw_path = notes_path.with_name(f"{stem}_f0.csv")
            filled_path = notes_path.with_name(f"{stem}_f0_filled.csv")
            cleaned_raw = clean_f0_annotations(notes_path, raw_path, filled_path)
            migrate_annotation_pair(notes_path, alignment_path, cleaned_raw)

        # The piece Crueger_AufAufMeinHerzMitFreuden is notated in D major but should be in C major to match the rest of the dataset. 
        # We transpose it down 2 semitones by editing the MEI file.
        crueger = "Crueger_AufAufMeinHerzMitFreuden"
        transpose_crueger_mei(audio_root / crueger / f"{crueger}.mei")

        migrated_paths = set(notes_paths) | set(alignment_paths)
        migrated_paths.update(audio_root.rglob("*_f0.csv"))
        migrated_paths.update(audio_root.rglob("*_f0_filled.csv"))
        for path in sorted(target.rglob("*.csv")):
            if path not in migrated_paths:
                # Migrate the per chorale top-level csvs
                migrate_piece_csv(path)

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
