import csv
import shutil
from pathlib import Path

FMA_GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

def main():
    project_root = Path(__file__).resolve().parents[1]
    fma_root = project_root / "raw" / "fma_small"
    metadata_path = project_root / "raw" / "fma_metadata" / "tracks.csv"
    output_dir = project_root / "raw" / "fma_small_organized" / "genres"
    
    if not fma_root.exists():
        print(f"FMA root not found: {fma_root}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    for genre in FMA_GENRES:
        (output_dir / genre).mkdir(exist_ok=True)

    # Read tracks.csv skipping the headers properly to get track_id and genre_top
    print("Reading metadata...")
    track_to_genre = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for _ in range(3): # Skip the 3 header rows in tracks.csv
            next(reader)
        for row in reader:
            if not row: continue
            track_id = row[0]
            subset = row[32] # column 'set -> subset'
            genre_top = row[40] # column 'track -> genre_top'
            if subset == "small" and genre_top in FMA_GENRES:
                track_to_genre[track_id.zfill(6)] = genre_top

    print(f"Found {len(track_to_genre)} tracks for FMA Small.")
    
    copied = 0
    for file_path in fma_root.rglob("*.mp3"):
        track_id = file_path.stem
        if track_id in track_to_genre:
            genre = track_to_genre[track_id]
            dest = output_dir / genre / file_path.name
            if not dest.exists():
                shutil.copy2(file_path, dest)
            copied += 1
            
    print(f"Successfully organized {copied} files into {output_dir}")

if __name__ == "__main__":
    main()
