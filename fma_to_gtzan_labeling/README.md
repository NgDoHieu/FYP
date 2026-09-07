# FMA to GTZAN Labeling Project

This project converts music from the FMA dataset into a GTZAN-style genre folder structure.

## Goal

The raw FMA dataset has many more genre labels than the GTZAN dataset. This script:

1. reads the FMA metadata,
2. maps each FMA genre to one of the 10 GTZAN labels,
3. copies matching audio files into `raw/gtzan/genres/<label>/`.

## GTZAN labels

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

## Folder layout

```text
raw/
  gtzan/
    genres/
      blues/
      classical/
      country/
      disco/
      hiphop/
      jazz/
      metal/
      pop/
      reggae/
      rock/
```

## Run

```powershell
py -3 label_fma_to_gtzan.py --fma-root ..\raw\fma_small --metadata ..\raw\fma_metadata\raw_tracks.csv --output ..\raw\gtzan\genres --limit 500
```

Use `--dry-run` to preview the mapping without copying files.

## Notes

This is a best-effort genre mapping. FMA has a broader taxonomy than GTZAN, so the script uses a practical mapping rule set rather than a perfect one-to-one label translation.
