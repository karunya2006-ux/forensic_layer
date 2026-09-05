import os
import re
from collections import Counter

FAKES_DIR = 'dataset/fakes'

def extract_base_id(filename):
    match = re.search(r'passport_(\d+)', filename)
    return match.group(1) if match else None

fake_files = [f for f in os.listdir(FAKES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
base_ids = [extract_base_id(f) for f in fake_files]
counts = Counter(base_ids)

print("Total fake images:", len(fake_files))
print("Number of unique base passport IDs used for fakes:", len(counts))
print("Distribution (base_id: how many fakes from it):")
for base_id, count in counts.most_common(15):
    print(f"  passport_{base_id}: {count} fakes")