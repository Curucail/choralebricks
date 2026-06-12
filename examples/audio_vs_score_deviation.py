"""Audio-vs-score pitch deviation statistics for ChoraleBricks and ChoraleWind.

For every aligned note the alignment CSV carries both the notated score pitch
(``pitch``) and the audio pitch derived from the tracked F0 (``pitch_audio`` /
``f0_median``). This script quantifies and plots how the *performed/synthesised*
pitch deviates from the *notated* pitch:

  * per-voice overlay histograms of notated vs. audio MIDI pitch,
  * the per-note semitone-difference distribution, with the +/-12 octave-error
    rate called out as a headline sanity number,
  * the fine intonation deviation in cents (octave errors excluded),
  * the cents deviation broken down by instrument and by family (brass/woodwind).

All comparisons use A4 = 442 Hz, the tuning the ensembles targeted and the
reference of the audio-derived columns.

Run against either dataset by pointing the env vars at the
``01_AudioAndAnnotations`` folders::

    CHORALEDB_PATH=...\\choralebricks\\1.1.0\\01_AudioAndAnnotations
    CHORALEWIND_PATH=...\\choralewind\\1.0.0\\01_AudioAndAnnotations
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from choralebricks.dataset import SongDB
from choralebricks.constants import (
    Voices,
    VOICE_COLORS,
    VOICE_STRINGS,
    InstrumentType,
)
from choralebricks.utils import midi2hz
from choralebricks.spec import read_semicolon_csv

A4_HZ = 442.0
OUTPUT_DIR = Path(__file__).resolve().parent / "output_plots"
DATASETS = {
    "choralebricks": os.getenv("CHORALEDB_PATH"),
    "choralewind": os.getenv("CHORALEWIND_PATH"),
}


def collect_deviations(root: Path) -> pd.DataFrame:
    """One row per aligned note with notated pitch, audio pitch, and deviations."""
    songdb = SongDB(root)
    records = []
    for song in songdb.songs:
        for track in song.tracks:
            alignment = song.song_dir / "alignments" / f"{Path(track.path_audio).stem}.csv"
            _, rows = read_semicolon_csv(alignment)
            for row in rows:
                pitch_score = int(row["pitch"])
                pitch_audio = int(row["pitch_audio"])
                f0_median = float(row["f0_median"])
                records.append(
                    {
                        "voice": int(track.voice),
                        "instrument": track.instrument.value,
                        "family": track.instrument_type.value,
                        "pitch_score": pitch_score,
                        "pitch_audio": pitch_audio,
                        "semitone_diff": pitch_audio - pitch_score,
                        "cents": 1200.0 * np.log2(f0_median / midi2hz(pitch_score, f_ref=A4_HZ)),
                    }
                )
    return pd.DataFrame.from_records(records)


def print_summary(name: str, df: pd.DataFrame) -> None:
    octave = df["semitone_diff"].abs().eq(12)
    mismatch = df["semitone_diff"].ne(0)
    fine = df.loc[df["semitone_diff"].eq(0), "cents"]
    print("=" * 60)
    print(f"{name}: {len(df)} aligned notes across {df['voice'].nunique()} voices")
    print(f"  pitch_audio != notated pitch : {mismatch.mean() * 100:5.2f} %  ({mismatch.sum()})")
    print(f"  +/-12 octave errors          : {octave.mean() * 100:5.2f} %  ({octave.sum()})")
    counts = df["semitone_diff"].value_counts().sort_index()
    print("  semitone-diff distribution   : "
          + ", ".join(f"{k:+d}:{v}" for k, v in counts.items() if v))
    print(f"  fine cents (|diff|=0)        : mean={fine.mean():+.1f}  std={fine.std():.1f}  "
          f"p5={np.percentile(fine, 5):+.1f}  p50={np.percentile(fine, 50):+.1f}  "
          f"p95={np.percentile(fine, 95):+.1f}")


def figure_pitch_overlay(name: str, df: pd.DataFrame) -> None:
    """Per-voice overlay of notated vs. audio MIDI pitch distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, voice in zip(axes.ravel(), Voices):
        sub = df[df["voice"] == voice.value]
        if sub.empty:
            ax.set_visible(False)
            continue
        bins = np.arange(sub[["pitch_score", "pitch_audio"]].min().min() - 1,
                         sub[["pitch_score", "pitch_audio"]].max().max() + 2) - 0.5
        ax.hist(sub["pitch_score"], bins=bins, alpha=0.5, color="0.4", label="notated")
        ax.hist(sub["pitch_audio"], bins=bins, alpha=0.5,
                color=VOICE_COLORS[voice], label="audio")
        ax.set_title(f"{VOICE_STRINGS[voice]}", fontsize=12)
        ax.set_xlabel("MIDI pitch")
        ax.legend()
    fig.suptitle(f"{name}: notated vs. audio pitch per voice", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_pitch_overlay.pdf")
    plt.close(fig)


def figure_semitone_diff(name: str, df: pd.DataFrame) -> None:
    """Per-note semitone difference (audio - notated); octave errors highlighted."""
    counts = df["semitone_diff"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:red" if abs(k) == 12 else "tab:blue" for k in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_yscale("log")
    ax.set_xlabel("audio pitch - notated pitch (semitones)")
    ax.set_ylabel("#notes (log)")
    octave_rate = df["semitone_diff"].abs().eq(12).mean() * 100
    ax.set_title(f"{name}: semitone deviation  (red = +/-12 octave errors, "
                 f"{octave_rate:.2f} % of notes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_semitone_diff.pdf")
    plt.close(fig)


def figure_cents(name: str, df: pd.DataFrame) -> None:
    """Fine intonation deviation in cents for non-octave-error notes."""
    fine = df[df["semitone_diff"].abs() < 6]
    fig, ax = plt.subplots(figsize=(10, 5))
    for voice in Voices:
        sub = fine[fine["voice"] == voice.value]
        if sub.empty:
            continue
        sns.kdeplot(np.clip(sub["cents"], -100, 100), ax=ax, fill=True, alpha=0.25,
                    color=VOICE_COLORS[voice], label=VOICE_STRINGS[voice])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlim(-100, 100)
    ax.set_xlabel("cents deviation from notated pitch (A4 = 442 Hz)")
    ax.set_title(f"{name}: fine intonation deviation per voice", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_cents_deviation.pdf")
    plt.close(fig)


def figure_cents_by_instrument(name: str, df: pd.DataFrame) -> None:
    """Cents deviation per instrument and per family (brass vs. woodwind)."""
    fine = df[df["semitone_diff"].abs() < 6].copy()
    fine["cents"] = fine["cents"].clip(-100, 100)
    order = fine.groupby("instrument")["cents"].median().sort_values().index
    fig, (ax_inst, ax_family) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 1]}
    )
    sns.boxplot(data=fine, x="cents", y="instrument", order=order, ax=ax_inst,
                color="0.8", fliersize=0)
    ax_inst.axvline(0, color="k", lw=0.8, ls="--")
    ax_inst.set_title(f"{name}: cents deviation by instrument")
    ax_inst.set_xlabel("cents")

    palette = {InstrumentType.BRASS.value: "tab:orange",
               InstrumentType.WOODWIND.value: "tab:green"}
    sns.violinplot(data=fine, x="family", y="cents", ax=ax_family,
                   hue="family", palette=palette, legend=False,
                   order=[f.value for f in InstrumentType if f.value in set(fine["family"])])
    ax_family.axhline(0, color="k", lw=0.8, ls="--")
    ax_family.set_title("by family")
    ax_family.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_cents_by_instrument.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    available = {name: path for name, path in DATASETS.items() if path and Path(path).is_dir()}
    if not available:
        raise SystemExit("Set CHORALEDB_PATH and/or CHORALEWIND_PATH to the "
                         "01_AudioAndAnnotations folders.")
    for name, path in available.items():
        df = collect_deviations(Path(path))
        print_summary(name, df)
        figure_pitch_overlay(name, df)
        figure_semitone_diff(name, df)
        figure_cents(name, df)
        figure_cents_by_instrument(name, df)
        print(f"  figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
