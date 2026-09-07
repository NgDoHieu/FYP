# Music Genre Audio CNN

This is a separate GTZAN audio project. It trains a CNN from WAV audio, not pre-generated PNG images.

Pipeline:

1. Load GTZAN tracks from `../raw/gtzan/genres`.
2. Split complete tracks into train, validation, and test sets.
3. Convert each five-second audio window to a 128-bin mel spectrogram.
4. Apply training-only brightness, noise, and time-mask augmentation.
5. Train a CNN and save evaluation results and graphs.

## Setup

From `NEW FYP`:

```powershell
uv venv --python 3.12 .venv
uv pip install --python ".venv\\Scripts\\python.exe" -r "music_genre_audio_cnn\\requirements.txt"
```

## Run

```powershell
& ".venv\\Scripts\\python.exe" "music_genre_audio_cnn\\train.py"
```

Outputs are written to `music_genre_audio_cnn/results`: `results.json`, `training_curves.png`, `confusion_matrix.png`, and the saved Keras model.

Accuracy is measured on held-out tracks and is not guaranteed to exceed a fixed threshold. The important distinction from the earlier image project is that this model learns from audio-derived mel spectrograms.

## Artist20 classifier

The CNN uses the MP3 archive. From `NEW FYP`, extract it once:

```powershell
tar -xzf "artist20\artist20-mp3s-32k.tgz" -C "artist20"
```

This produces the expected one-artist-per-directory layout:

```text
artist20/artist20/mp3s-32k/
	artist_name_1/
		album_name/track_01.mp3
	artist_name_2/
		album_name/track_01.mp3
```

Then run:

```powershell
& ".venv\Scripts\python.exe" "music_genre_audio_cnn\train_artist20.py"
```

The MP3 archive is sufficient. `artist20-mfccs.tgz` and `artist20-chromftrs.tgz` are precomputed features for other model types and are not used by this mel-spectrogram CNN. Use `--data-dir` to select a different Artist20 folder, and `--epochs` or `--batch-size` to adjust training. The model, discovered artist labels, and evaluation report are saved in `music_genre_audio_cnn/results_artist20`.
