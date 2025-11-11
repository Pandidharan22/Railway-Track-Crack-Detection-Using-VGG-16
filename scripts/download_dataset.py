import os
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "salmaneunus/railway-track-fault-detection"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def try_kagglehub() -> Path | None:
    try:
        import kagglehub  # type: ignore
    except Exception as e:
        print(f"kagglehub not available: {e}")
        return None
    try:
        print(f"Attempting download via kagglehub: {DATASET_SLUG}")
        path = kagglehub.dataset_download(DATASET_SLUG)
        print("kagglehub download path:", path)
        return Path(path)
    except Exception as e:
        print(f"kagglehub download failed: {e}")
        return None


def try_kaggle_cli() -> Path | None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as e:
        print(f"kaggle package not available: {e}")
        return None

    # Check credentials
    # Requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Kaggle API authentication failed. Make sure your API token is set up.")
        print("- Place kaggle.json in %USERPROFILE%/.kaggle on Windows")
        print("- Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print(f"Details: {e}")
        return None

    zip_path = RAW_DIR / "dataset.zip"
    try:
        print(f"Downloading {DATASET_SLUG} to {zip_path} ...")
        api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), quiet=False, force=True)
        # Kaggle API saves as <slug>.zip in target path
        # Find the most recent zip in RAW_DIR
        zips = sorted(RAW_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            raise RuntimeError("No zip file found after download.")
        zip_path = zips[0]
        return zip_path
    except Exception as e:
        print(f"Kaggle CLI download failed: {e}")
        return None


def extract_zip(zip_file: Path, target_dir: Path) -> Path:
    print(f"Extracting {zip_file} -> {target_dir}")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(target_dir)
    # Return top-level extracted folder if any
    entries = [p for p in target_dir.iterdir() if p.is_dir()]
    if entries:
        print("Extracted directories:", ", ".join(e.name for e in entries))
        return entries[0]
    return target_dir


def main():
    ensure_dirs()

    # 1) Try kagglehub (fast path)
    hub_path = try_kagglehub()
    extracted_root: Path | None = None
    if hub_path and hub_path.exists():
        # If kagglehub returns a folder, copy/organize under RAW_DIR
        print(f"Copying from {hub_path} to {RAW_DIR}")
        # If hub_path is already inside a temp cache, copy tree
        # Avoid re-copying if RAW_DIR is same source
        if hub_path.resolve() != RAW_DIR.resolve():
            # Copy content without duplicating large files unnecessarily
            # We'll do a simple file-by-file copy to be robust.
            for root, dirs, files in os.walk(hub_path):
                rel = os.path.relpath(root, hub_path)
                dest_dir = RAW_DIR / rel
                Path(dest_dir).mkdir(parents=True, exist_ok=True)
                for f in files:
                    src_f = Path(root) / f
                    dst_f = Path(dest_dir) / f
                    if not dst_f.exists():
                        try:
                            # Use binary copy
                            with open(src_f, 'rb') as s, open(dst_f, 'wb') as d:
                                d.write(s.read())
                        except Exception as e:
                            print(f"Failed to copy {src_f} -> {dst_f}: {e}")
        extracted_root = RAW_DIR

    # 2) Fallback to Kaggle CLI API if needed
    if extracted_root is None:
        zip_path = try_kaggle_cli()
        if zip_path is None:
            print("\nERROR: Failed to download dataset via kagglehub or kaggle API.")
            print("Please ensure internet connectivity and Kaggle credentials (API token) are configured.")
            sys.exit(1)
        extracted_root = extract_zip(zip_path, RAW_DIR)

    print("\nDataset is available under:")
    print(f"  {extracted_root}")
    print("\nNote: The data/ directory is git-ignored by default to avoid large commits.")


if __name__ == "__main__":
    main()
