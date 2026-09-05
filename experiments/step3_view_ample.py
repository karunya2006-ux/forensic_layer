import cv2
import os

REALS_DIR = 'dataset/reals'

print("Current working directory:", os.getcwd())
print("Does dataset/reals exist?", os.path.exists(REALS_DIR))

files = sorted(os.listdir(REALS_DIR))
print("First 10 files found:", files[:10])

# Filter to only actual image files
image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print("First 10 IMAGE files:", image_files[:10])

if image_files:
    sample_path = os.path.join(REALS_DIR, image_files[0])
    print("Trying to load:", sample_path)
    img = cv2.imread(sample_path)
    print("Loaded successfully?" , img is not None)