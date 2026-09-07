# Music Genre Audio CNN

This project trains convolutional neural networks directly from audio files. Each track is loaded with `librosa`, converted to a 128-bin mel spectrogram, divided into ten three-second excerpts, and classified by a CNN. The project supports GTZAN, FMA Small, and Artist20-style datasets.

The training pipeline uses:

- 22,050 Hz mono audio
- 30 seconds per track
- ten three-second excerpts per track
- training-only brightness, Gaussian-noise, and time-masking augmentation
- stratified train, validation, and test splits
- Keras models with batch normalization, dropout, L2 regularization, and early stopping

Large audio datasets are not included in the Git repository. See the root `.gitignore` file for the excluded local folders.

## Project layout

```text
music_genre_audio_cnn/
	train.py                 # GTZAN, ten genres
	train_fma.py             # FMA Small, eight genres
	train_artist20.py        # Artist20, one class per artist
	requirements.txt
	results/                 # GTZAN model and evaluation outputs
	results_fma/             # FMA outputs created after training
	results_artist20/        # Artist20 model and evaluation outputs
```

## Setup

From the repository root (`NEW FYP`), create the environment and install the dependencies:

```powershell
uv venv --python 3.12 .venv
uv pip install --python ".venv\Scripts\python.exe" -r "music_genre_audio_cnn\requirements.txt"
```

The scripts require Python 3.12, TensorFlow, librosa, NumPy, Matplotlib, and scikit-learn. The FMA-to-GTZAN conversion script also uses `soundfile`; install it if you use that workflow and it is not already available in your environment.

## Dataset preparation

### GTZAN

The GTZAN trainer expects this structure:

```text
raw/gtzan/genres/
	blues/       classical/   country/     disco/       hiphop/
	jazz/        metal/       pop/         reggae/      rock/
```

To create GTZAN-style folders from FMA metadata, run the labeling utility from the repository root:

```powershell
& ".venv\Scripts\python.exe" "fma_to_gtzan_labeling\label_fma_to_gtzan.py" --limit 500
```

Use `--dry-run` to preview the mapping. The default source and metadata paths are `raw/fma_small` and `raw/fma_metadata/raw_tracks.csv`; the default output is `raw/gtzan/genres`.

### FMA Small

The FMA trainer uses eight top-level genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. First organize the FMA Small files:

```powershell
& ".venv\Scripts\python.exe" "fma_to_gtzan_labeling\organize_fma_small.py"
```

The script reads `raw/fma_small` and `raw/fma_metadata/tracks.csv`, then creates `raw/fma_small_organized/genres`.

### Artist20

Extract the Artist20 MP3 archive once:

```powershell
tar -xzf "artist20\artist20-mp3s-32k.tgz" -C "artist20"
```

The expected layout is:

```text
artist20/artist20/mp3s-32k/
	artist_name_1/album_name/track_01.mp3
	artist_name_2/album_name/track_01.mp3
```

The MP3 archive is sufficient. The precomputed MFCC and chroma archives are not used by this mel-spectrogram CNN.

## Training

Run commands from the repository root:

```powershell
# GTZAN
& ".venv\Scripts\python.exe" "music_genre_audio_cnn\train.py"

# FMA Small
& ".venv\Scripts\python.exe" "music_genre_audio_cnn\train_fma.py"

# Artist20
& ".venv\Scripts\python.exe" "music_genre_audio_cnn\train_artist20.py"
```

Artist20 accepts optional arguments:

```powershell
& ".venv\Scripts\python.exe" "music_genre_audio_cnn\train_artist20.py" --data-dir "path\to\artist20" --epochs 40 --batch-size 32
```

## Outputs and evaluation

The GTZAN and FMA scripts save their results under `results` and `results_fma`, respectively:

- `*.keras`: best model checkpoint selected by validation accuracy
- `results.json`: test loss, test accuracy, per-class report, split sizes, and training details
- `training_curves.png`: training and validation loss/accuracy
- `confusion_matrix.png`: test-set confusion matrix

The Artist20 script saves its best model, `results.json`, and the discovered artist order in `results_artist20`. Artist20 evaluation averages predictions across the ten excerpts from each track before calculating track-level accuracy.

Reported accuracy is measured on held-out tracks and depends on the dataset version, available audio files, and training run. It should not be interpreted as a guaranteed threshold.
