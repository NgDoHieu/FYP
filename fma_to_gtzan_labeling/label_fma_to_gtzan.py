import ast
import csv
import subprocess
import shutil
from collections import Counter
from pathlib import Path

import librosa
import soundfile as sf

GTZAN_LABELS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# Approximate mapping from FMA labels to GTZAN labels.
# Many FMA categories are broader than GTZAN, so this is a practical mapping.
FMA_TO_GTZN = {
    "blues": "blues",
    "classical": "classical",
    "country": "country",
    "disco": "disco",
    "funk": "disco",
    "soul": "disco",
    "soul-rnb": "disco",
    "rnb": "disco",
    "rhythm and blues": "disco",
    "hip-hop": "hiphop",
    "hip hop": "hiphop",
    "rap": "hiphop",
    "jazz": "jazz",
    "jazz: vocal": "jazz",
    "vocal jazz": "jazz",
    "metal": "metal",
    "hard rock": "metal",
    "heavy metal": "metal",
    "punk": "metal",
    "rock": "rock",
    "indie-rock": "rock",
    "post-rock": "rock",
    "krautrock": "rock",
    "loud-rock": "rock",
    "noise": "rock",
    "drone": "rock",
    "experimental": "rock",
    "alternative": "rock",
    "garage": "rock",
    "psychedelic": "rock",
    "lo-fi": "rock",
    "electronic": "hiphop",
    "ambient electronic": "hiphop",
    "electronica": "hiphop",
    "pop": "pop",
    "easy listening": "pop",
    "adult contemporary": "pop",
    "reggae": "reggae",
    "dancehall": "reggae",
    "roots reggae": "reggae",
}


def normalize_label(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("&", "and")
        .replace("-", " ")
        .replace("_", " ")
    )


def parse_track_genres(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except Exception:
        return []

    if isinstance(parsed, list):
        return [str(item.get("genre_title", "")).strip() for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        title = parsed.get("genre_title")
        return [str(title).strip()] if title else []
    return []


def map_fma_to_gtzan(genres: list[str]) -> str | None:
    for genre in genres:
        normalized = normalize_label(genre)
        mapped = FMA_TO_GTZN.get(normalized)
        if mapped:
            return mapped

    # Fallback: scan for containing keywords from the mapping dictionary.
    text = " ".join(normalize_label(g) for g in genres)
    for key, value in FMA_TO_GTZN.items():
        if key in text:
            return value
    return None


def find_track_file(track_id: str, fma_root: Path) -> Path | None:
    normalized = str(track_id).strip()
    if not normalized:
        return None

    candidates = {normalized, normalized.zfill(6)}
    for file_path in sorted(fma_root.rglob("*.mp3")):
        file_stem = file_path.stem
        if file_stem in candidates or file_stem.lstrip("0") in candidates:
            return file_path
    return None


def convert_mp3_to_wav(source: Path, destination: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        error_output = result.stderr.strip() or "ffmpeg decode failed"
        print(f"Skipping invalid audio file: {source} ({error_output})")
        return False

    try:
        audio, sample_rate = librosa.load(str(source), sr=None, mono=True)
        sf.write(str(destination), audio, sample_rate)
        return True
    except Exception as exc:
        print(f"Skipping invalid audio file: {source} ({exc})")
        return False


def build_manifest(metadata_path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            track_id = (row.get("track_id") or "").strip()
            if not track_id:
                continue
            genre_titles = parse_track_genres(row.get("track_genres", ""))
            gtzan_label = map_fma_to_gtzan(genre_titles)
            if gtzan_label:
                manifest[track_id] = gtzan_label
    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    fma_root = project_root / "raw" / "fma_small"
    metadata_path = project_root / "raw" / "fma_metadata" / "raw_tracks.csv"
    output_dir = project_root / "raw" / "gtzan" / "genres"
    limit = 0
    dry_run = False

    if not fma_root.exists():
        raise FileNotFoundError(f"FMA root not found: {fma_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"FMA metadata not found: {metadata_path}")

    manifest = build_manifest(metadata_path)
    counts = Counter()
    copied = 0
    skipped_invalid_audio = 0

    for track_id, label in manifest.items():
        if track_id in {"", None}:
            continue
        if label not in GTZAN_LABELS:
            continue

        source = find_track_file(track_id, fma_root)
        if source is None:
            continue

        destination_dir = output_dir / label
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / f"{source.stem}.wav"
        if destination.exists():
            counts[label] += 0
        else:
            if not dry_run:
                if source.suffix.lower() == ".mp3":
                    if not convert_mp3_to_wav(source, destination):
                        skipped_invalid_audio += 1
                        continue
                else:
                    shutil.copy2(source, destination)
            counts[label] += 1
            copied += 1

        if limit and copied >= limit:
            break

    print("GTZAN labels used:")
    for label in GTZAN_LABELS:
        print(f"  {label}: {counts[label]}")
    print(f"Copied/selected tracks: {copied}")
    print(f"Skipped invalid audio files: {skipped_invalid_audio}")
    if dry_run:
        print("Dry run only: no files were copied.")


if __name__ == "__main__":
    main()
