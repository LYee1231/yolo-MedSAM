import argparse
from pathlib import Path
import os
import shutil

def safe_rename(src: Path, dst: Path):
    """If the target already exists, automatically increment the filename to prevent overwriting: name (1).ext, name (2).ext ..."""
    if not dst.exists():
        src.rename(dst)
        return dst
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem} ({i}){suf}")
        if not cand.exists():
            src.rename(cand)
            return cand
        i += 1

def main():
    ap = argparse.ArgumentParser(
        description="Batch removal of specified substrings from filenames (default: removes _Annotation)"
    )
    ap.add_argument("--root", required=True, help="Root directory (all files within it will be processed recursively)")
    ap.add_argument("--substr", default="_mask", help="The substring to be deleted")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg,.tif,.tiff,.bmp",
                    help="Specified file extensions, comma-separated; leave blank to include all files")
    ap.add_argument("--apply", action="store_true", help="Perform the actual renaming (omitting this indicates a trial run)")
    args = ap.parse_args()

    root = Path(args.root)
    assert root.is_dir(), f"The directory does not exist: {root}"

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    count = 0
    preview = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        stem_new = p.stem.replace(args.substr, "")
        if stem_new == p.stem:
            continue
        dst = p.with_name(f"{stem_new}{p.suffix}")
        preview.append((p, dst))
        count += 1

    if not preview:
        print("No files requiring modification were found.")
        return

    print(f"{count} filenames will be modified (displaying the first 20 previews):")
    for i, (src, dst) in enumerate(preview[:20], 1):
        print(f"{i:>2}. {src.name}  ->  {dst.name}")

    if not args.apply:
        print("\nThis is currently in [trial operation] mode, allowing preview only without making changes. If you confirm everything is correct, please add --apply to execute the changes.")
        return

    done = 0
    for src, dst in preview:
        final_dst = safe_rename(src, dst)
        done += 1
    print(f"Done! A total of {done} files have been renamed.")

if __name__ == "__main__":
    main()
