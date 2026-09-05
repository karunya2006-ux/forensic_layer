import cv2
import os
from pathlib import Path

# Adjust path to root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALS_DIR = PROJECT_ROOT / 'dataset' / 'reals'
RESULTS_DIR = PROJECT_ROOT / 'results'
BACKGROUND_PATCH = (0.05, 0.01, 0.85, 0.045)  # x%, y%, w%, h%

def visualize():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_file = list(REALS_DIR.glob('*.jpg'))[0]
    print(f"Testing on: {sample_file.name}")
    
    img = cv2.imread(str(sample_file))
    if img is None:
        print("Could not load image.")
        return
        
    h_img, w_img = img.shape[:2]
    
    # Calculate pixel coordinates
    x_pct, y_pct, w_pct, h_pct = BACKGROUND_PATCH
    x, y = int(x_pct * w_img), int(y_pct * h_img)
    w, h = int(w_pct * w_img), int(h_pct * h_img)
    
    # Draw a thick green rectangle
    preview = img.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    # Add text label
    label = "BACKGROUND PATCH"
    cv2.putText(preview, label, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    out_path = RESULTS_DIR / 'patch_visualization.jpg'
    cv2.imwrite(str(out_path), preview)
    print(f"Saved visualization to: {out_path}")

if __name__ == '__main__':
    visualize()
