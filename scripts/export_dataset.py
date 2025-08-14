"""
    Export ChoraleBricks to an NPZ file.
    We use this as a starting point for machine learning experiments.
"""

import numpy as np
import logging
from pathlib import Path
import soundfile as sf
import random
import resampy
import matplotlib.pyplot as plt
import librosa

from choralebricks.dataset import SongDB, EnsemblePermutations
from choralebricks.utils import read_f0

logger = logging.getLogger(__name__)


def main():
    cbdb = SongDB()

    output = []

    for cur_song in cbdb.songs:
        logger.info(f"Processing {cur_song}...")
        cur_ensemble_permutations = EnsemblePermutations(cur_song)
        all_ensembles = list(cur_ensemble_permutations)

        output_song = []

        for cur_ensemble in random.sample(all_ensembles, k=5):
            output_ensemble = []

            # Collect data for export
            for cur_track in cur_ensemble:
                # cur_track.path_fo
                audio, sample_rate = sf.read(cur_track.path_audio)
                target_sample_rate = 22050
                audio = resampy.resample(audio, sample_rate, target_sample_rate)
                output_ensemble.append({
                    "audio": audio,
                    "sample_rate": target_sample_rate,
                    "f0": read_f0(cur_track.path_f0),
                    "voice": cur_track.voice,
                    "instrument": cur_track.instrument,
                })

            output_song.append(output_ensemble)

        output.append(output_song)

    # Save to NPZ file
    output_path = Path("chorale_bricks_export.npz")
    np.savez(output_path, data=output)
    logger.info(f"Exported data to {output_path}")


def get_sample(ensemble, win_len_sec, display=True):
    """" Sample a window of audio and F0 from the ensemble.

    This function samples a window of audio and F0 from the ensemble,
    ensuring that the sampled audio is centered around a random point in the audio.
    It also interpolates the F0 to match the sampled audio time axis.

    Parameters
    ----------
    ensemble : list of dict
        List where each element represents a voice in the ensemble.
    win_len_sec : float
        Length of the window to sample, in seconds.
    display : bool, optional
        If True, displays a spectrogram with F0 overlay for each voice. Defaults to True.

    Returns
    -------
    audio_sample : np.ndarray
        Array of shape (num_voices, win_len_samples) containing the sampled audio for each voice.
    f0_sample : np.ndarray
        Array of shape (num_voices, win_len_samples) containing the interpolated F0 for each voice.

    Notes
    -----
        - The function avoids sampling from the very beginning and end of the audio to reduce edge effects.
        - If `display` is True, a matplotlib figure is shown with the spectrogram and F0 overlay for each voice.
    """

    cur_voice_idx = np.random.choice(np.arange(len(ensemble)), size=1, replace=False)[0]

    # sample win_len seconds of audio from the first ensemble
    cur_sample_rate = ensemble[0]['sample_rate']
    win_len = int(win_len_sec * cur_sample_rate)  # convert to samples

    # let's not use the first window and the last window to avoid edge effects
    valid_sample_idcs = np.arange(win_len,
                                len(ensemble[cur_voice_idx]['audio']) - win_len)
    center_sample_idx = np.random.choice(valid_sample_idcs)
    print("Window center sample index:", center_sample_idx, center_sample_idx - int(win_len / 2), center_sample_idx + int(win_len / 2))
    audio_sample = []
    f0_sample = []

    for cur_voice in ensemble:
        cur_audio_sample = cur_voice['audio'][center_sample_idx - int(win_len / 2): center_sample_idx + int(win_len / 2)]
        audio_sample.append(cur_audio_sample)

        # get the F0 for the sampled audio
        cur_f0_sample = cur_voice['f0']
        cur_f0_sample = cur_f0_sample[(cur_f0_sample['t'] >= (center_sample_idx - int(win_len / 2)) / cur_sample_rate) &
                                      (cur_f0_sample['t'] <= (center_sample_idx + int(win_len / 2)) / cur_sample_rate)]

        # interpolate the F0 to the sampled audio time axis
        # only for visualization purposes, messes up the un-voiced frames in the interpolation
        cur_f0_sample = np.interp(np.arange(0, len(cur_audio_sample)) / cur_sample_rate,
                                  cur_f0_sample['t'] - cur_f0_sample["t"].min(),  # make relative to the start of the sample
                                  cur_f0_sample['f0'])
        f0_sample.append(cur_f0_sample)

    # convert to numpy arrays
    audio_sample = np.array(audio_sample)
    f0_sample = np.array(f0_sample)

    audio_sample = np.array(audio_sample)
    print(f"Sampled audio shape: {audio_sample.shape}")
    print(f"Sampled F0 shape: {f0_sample.shape}")

    if display:
        # figure with subfigures for each voice
        plt.figure(figsize=(10, 10))
        plt.suptitle(f"Spectrogram of Ensemble {cur_ensemble_idx[0]}")

        for cur_voice_idx in range(audio_sample.shape[0]):
            plt.subplot(audio_sample.shape[0], 1, cur_voice_idx + 1)

            # make stft with librosa and add a subplot
            D = librosa.stft(audio_sample[cur_voice_idx], n_fft=N_FFT, hop_length=HOP_LENGTH)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            plt.imshow(S_db, aspect='auto', origin='lower', cmap="Grays")
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")

            # add F0 line to the spectrogram
            win_centers = HOP_LENGTH * np.arange(S_db.shape[1]) - HOP_LENGTH / 2
            cur_f0_sample_plot = np.interp(win_centers,
                                      np.arange(0, len(f0_sample[cur_voice_idx])),
                                      f0_sample[cur_voice_idx])

            plt.plot(cur_f0_sample_plot,
                     color='red',
                     linewidth=2,
                     label='F0')

            plt.legend()
            plt.tight_layout()
            plt.grid()

        plt.show()

    return audio_sample, f0_sample

if __name__ == "__main__":
    random.seed(42)
    # np.random.seed(42)

    logging.basicConfig(level=logging.INFO)
    # main()

    WIN_LEN_SEC = 2  # seconds
    N_FFT = 2048
    HOP_LENGTH = 512

    # Example of how to read the exported data
    with np.load("scripts/chorale_bricks_export_v20240812.npz", allow_pickle=True) as data:
        songs = data["data"]

        # put all ensembles into a single list
        all_ensembles = [ensemble for song in songs for ensemble in song]
        print(f"Total ensembles: {len(all_ensembles)}")
        print(f"First ensemble audio shape: {all_ensembles[0][0]['audio'].shape}")
        print(f"First ensemble F0 shape: {all_ensembles[0][0]['f0'].shape}")

        # sample an ensemble and voice
        all_ensembles_idcs = np.arange(len(all_ensembles))
        cur_ensemble_idx = np.random.choice(all_ensembles_idcs, size=1, replace=False)
        sampled_ensemble = all_ensembles[cur_ensemble_idx[0]]
        print(f"Sampled ensemble audio shape: {sampled_ensemble[0]['audio'].shape}")
        print(f"Sampled ensemble F0 shape: {sampled_ensemble[0]['f0'].shape}")

        X, y = get_sample(sampled_ensemble, win_len_sec=WIN_LEN_SEC, display=True)
