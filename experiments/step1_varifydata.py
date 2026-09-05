import os
import cv2
from pathlib import Path


# Find the project root: forensic_layer/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REALS_DIR = PROJECT_ROOT / "dataset" / "reals"
FAKES_DIR = PROJECT_ROOT / "dataset" / "fakes"


def verify_folder(directory, label):

    print(f"\nChecking {label}")
    print(f"Directory: {directory}")

    # Check whether folder exists
    if not directory.exists():
        print(f"ERROR: Folder not found: {directory}")
        return []

    # Get image files
    files = [
        f for f in directory.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    print(f"{label}: {len(files)} images found")

    # Check for corrupted/unreadable images
    broken = []

    for f in files:
        img = cv2.imread(str(f))

        if img is None:
            broken.append(f.name)

    if broken:
        print(f"WARNING - {len(broken)} unreadable files:")
        for file in broken:
            print(f"  {file}")
    else:
        print("All files readable - OK")

    return files


# Verify real images
real_files = verify_folder(REALS_DIR, "Reals")

# Verify fake images
fake_files = verify_folder(FAKES_DIR, "Fakes")


# Dataset summary
print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"Real images : {len(real_files)}")
print(f"Fake images : {len(fake_files)}")
print(f"Total images: {len(real_files) + len(fake_files)}")


# Show sample filenames
print("\nSample real filenames:")

for f in real_files[:3]:
    print(" ", f.name)

print("\nSample fake filenames:")

for f in fake_files[:3]:
    print(" ", f.name)