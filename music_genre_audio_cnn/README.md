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

## How it works

### Dataset organization

Each dataset is represented as labeled directories. GTZAN uses ten genre directories, FMA Small uses eight genre directories, and Artist20 uses one directory for each artist. The FMA utilities prepare these directory structures from the original FMA metadata and audio files. The classifier scripts then discover the available files and assign an integer label to each directory.

The expected GTZAN structure is:

```text
raw/gtzan/genres/
	blues/       classical/   country/     disco/       hiphop/
	jazz/        metal/       pop/         reggae/      rock/
```

The FMA Small classifier uses these eight labels:


- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

Artist20 follows an artist and album hierarchy beneath its dataset root:

```text
artist20/artist20/mp3s-32k/
	artist_name_1/album_name/track_01.mp3
	artist_name_2/album_name/track_01.mp3
```

The Artist20 model uses the audio archive rather than the precomputed MFCC or chroma feature archives.

### Audio preprocessing

For every track, the pipeline:

1. Loads up to 30 seconds of mono audio at 22,050 Hz.
2. Computes a 128-bin mel spectrogram using a 2,048-sample FFT and a 512-sample hop length.
3. Converts mel power to decibels and normalizes the values to the range 0 to 1.
4. Splits the spectrogram into ten three-second excerpts, padding short excerpts when necessary.
5. Applies augmentation only to training excerpts: brightness changes, Gaussian noise, and random time masking.

This produces a consistent four-dimensional input tensor for the CNN while retaining short-term time and frequency information from the original audio.

### CNN architecture

The classifiers share the same general architecture. Three convolutional blocks learn increasingly complex spectrogram patterns. Each block uses a convolution, batch normalization, max pooling, and dropout. Global average pooling then converts the feature maps into a compact representation, followed by a 128-unit dense layer and a softmax classification layer.

The convolutional and dense layers use L2 regularization. Adam optimization trains the network with sparse categorical cross-entropy. Early stopping restores the weights from the best validation-accuracy epoch, while the learning rate is reduced when validation loss stops improving.

### Dataset-specific training

GTZAN and FMA Small classify individual excerpts during evaluation. Their datasets are split by complete track before excerpts are generated, preventing excerpts from the same track from appearing in both training and evaluation data.

Artist20 also splits complete tracks, but combines the ten excerpt predictions by averaging their class probabilities. The final Artist20 result is therefore a track-level artist prediction rather than an individual-excerpt prediction.

## Outputs and evaluation

The GTZAN and FMA scripts save their results under `results` and `results_fma`, respectively:

- `*.keras`: best model checkpoint selected by validation accuracy
- `results.json`: test loss, test accuracy, per-class report, split sizes, and training details
- `training_curves.png`: training and validation loss/accuracy
- `confusion_matrix.png`: test-set confusion matrix

The Artist20 script saves its best model, `results.json`, and the discovered artist order in `results_artist20`. Artist20 evaluation averages predictions across the ten excerpts from each track before calculating track-level accuracy.

Reported accuracy is measured on held-out tracks and depends on the dataset version, available audio files, and training run. It should not be interpreted as a guaranteed threshold.
