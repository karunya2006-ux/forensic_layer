import os
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ---------- PATH CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALS_DIR = PROJECT_ROOT / 'dataset' / 'reals'
FAKES_DIR = PROJECT_ROOT / 'dataset' / 'fakes'
RESULTS_DIR = PROJECT_ROOT / 'results'

REFERENCE_HIST_PATH = RESULTS_DIR / 'layer1_reference_histogram.npy'
METADATA_PATH = RESULTS_DIR / 'layer1_metadata.json'
PLOT_PATH = RESULTS_DIR / 'layer1_background_texture.png'

# Background patch: (x_pct, y_pct, w_pct, h_pct)
# Verified against AZE passport top decorative border strip
BACKGROUND_PATCH = (0.05, 0.01, 0.85, 0.045)


# ---------- HELPER FUNCTIONS ----------
def get_image_files(directory):
    if not os.path.exists(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])


def get_background_patch(image, patch_coords=BACKGROUND_PATCH):
    """Crops the specified relative region from the image."""
    h_img, w_img = image.shape[:2]
    x_pct, y_pct, w_pct, h_pct = patch_coords
    x, y = int(x_pct * w_img), int(y_pct * h_img)
    w, h = int(w_pct * w_img), int(h_pct * h_img)
    return image[y:y+h, x:x+w]


def locate_photo_region(image: np.ndarray) -> tuple | None:
    """
    Locates the portrait photo bounding box in a passport image.
    Uses Canny edge detection, dilation, and external contour filtering by area & aspect ratio.
    Returns (x, y, w, h) or None if no candidate matches.
    """
    if image is None or image.size == 0:
        return None

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    if img_area == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    max_area_ratio = 0.0

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        area_ratio = contour_area / img_area
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0.0

        if 0.03 <= area_ratio <= 0.15 and 0.5 <= aspect_ratio <= 1.0:
            if area_ratio > max_area_ratio:
                max_area_ratio = area_ratio
                best_bbox = (x, y, w, h)

    return best_bbox


def get_histogram_from_image(image, patch_coords=BACKGROUND_PATCH):
    """Extracts a normalized grayscale intensity histogram from the background patch."""
    if image is None or image.size == 0:
        return None

    patch = get_background_patch(image, patch_coords)
    if patch.size == 0:
        return None

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_sum = hist.sum()
    if hist_sum == 0:
        return None
    return hist / hist_sum


def get_histogram(image_path, patch_coords=BACKGROUND_PATCH):
    """Loads image from path and extracts normalized patch histogram."""
    img = cv2.imread(str(image_path))
    return get_histogram_from_image(img, patch_coords)


# ---------- CALIBRATION & TRAINING ----------
def calibrate_layer1(reals_dir=REALS_DIR, fakes_dir=FAKES_DIR, save_artifacts=True):
    """
    Calibrates Layer 1 reference histogram and statistical threshold using genuine dataset.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    real_files = get_image_files(reals_dir)
    fake_files = get_image_files(fakes_dir)

    print(f"[Layer 1 Calibration] Found {len(real_files)} real images, {len(fake_files)} fake images.")

    genuine_hists = []
    for f in real_files:
        h = get_histogram(os.path.join(reals_dir, f))
        if h is not None:
            genuine_hists.append(h)

    fake_hists = []
    for f in fake_files:
        h = get_histogram(os.path.join(fakes_dir, f))
        if h is not None:
            fake_hists.append(h)

    if len(genuine_hists) < 5:
        raise ValueError(f"Insufficient genuine images for calibration (found {len(genuine_hists)}).")

    # 1. Compute reference genuine distribution
    reference_genuine = np.mean(genuine_hists, axis=0)

    # 2. Compute intra-class (genuine-to-genuine) distances
    genuine_to_genuine = np.array([wasserstein_distance(h, reference_genuine) for h in genuine_hists])
    gen_mean = float(np.mean(genuine_to_genuine))
    gen_std = float(np.std(genuine_to_genuine))
    gen_max = float(np.max(genuine_to_genuine))

    # 3. Compute fake-to-genuine distances if fakes exist
    fake_to_genuine = np.array([wasserstein_distance(h, reference_genuine) for h in fake_hists]) if fake_hists else np.array([])
    fake_mean = float(np.mean(fake_to_genuine)) if len(fake_to_genuine) > 0 else 0.0

    # 4. Determine calibrated threshold (3-sigma or empirical margin)
    # Default to 3 standard deviations above genuine mean, with minimum guard band
    calibrated_threshold = max(gen_mean + 3 * gen_std, gen_max * 1.5, 0.00004)

    metadata = {
        "layer_name": "Layer 1: Background Texture Analysis",
        "patch_coords": BACKGROUND_PATCH,
        "genuine_samples_count": len(genuine_hists),
        "fake_samples_count": len(fake_hists),
        "genuine_distance_mean": gen_mean,
        "genuine_distance_std": gen_std,
        "genuine_distance_max": gen_max,
        "fake_distance_mean": fake_mean,
        "calibrated_threshold": calibrated_threshold
    }

    if save_artifacts:
        np.save(REFERENCE_HIST_PATH, reference_genuine)
        with open(METADATA_PATH, "w") as mf:
            json.dump(metadata, mf, indent=4)

        # Plot separation distribution
        plt.figure(figsize=(8, 5))
        plt.hist(genuine_to_genuine, alpha=0.6, label=f'Genuine ({len(genuine_to_genuine)})', bins=15, color='green')
        if len(fake_to_genuine) > 0:
            plt.hist(fake_to_genuine, alpha=0.6, label=f'Fake ({len(fake_to_genuine)})', bins=15, color='red')
        plt.axvline(calibrated_threshold, color='black', linestyle='--', label=f'Threshold ({calibrated_threshold:.5f})')
        plt.xlabel('Wasserstein Distance to Genuine Reference')
        plt.ylabel('Count')
        plt.title('Layer 1: Background Texture Distance Distribution')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Layer 1 Calibration] Reference saved to {REFERENCE_HIST_PATH}")
        print(f"[Layer 1 Calibration] Plot saved to {PLOT_PATH}")
        print(f"[Layer 1 Calibration] Metadata saved to {METADATA_PATH}")

    return reference_genuine, metadata


# ---------- SINGLE-IMAGE INFERENCE API ----------
def evaluate_background_texture(image_input, reference_hist=None, threshold=None):
    """
    Evaluates a single passport image against the calibrated genuine background texture baseline.

    Parameters:
        image_input: Either a file path (str, Path) or an OpenCV BGR image (np.ndarray).
        reference_hist: Optional pre-loaded numpy reference histogram.
        threshold: Optional custom threshold.

    Returns:
        dict: Diagnostic results including score, pass/fail status, and distance metrics.
    """
    # 1. Load Reference Histogram
    if reference_hist is None:
        if not os.path.exists(REFERENCE_HIST_PATH):
            calibrate_layer1()
        reference_hist = np.load(REFERENCE_HIST_PATH)

    # 2. Load Threshold
    if threshold is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as mf:
                meta = json.load(mf)
                threshold = meta.get("calibrated_threshold", 0.00004)
        else:
            threshold = 0.00004

    # 3. Read image if file path is provided
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            return {
                "layer_name": "Layer 1: Background Texture Analysis",
                "passed": False,
                "score": 1.0,
                "error": f"Could not read image from {image_input}"
            }
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return {
            "layer_name": "Layer 1: Background Texture Analysis",
            "passed": False,
            "score": 1.0,
            "error": "Invalid input type. Expected file path or numpy image."
        }

    # 4. Extract patch histogram
    doc_hist = get_histogram_from_image(img, BACKGROUND_PATCH)
    if doc_hist is None:
        return {
            "layer_name": "Layer 1: Background Texture Analysis",
            "passed": False,
            "score": 1.0,
            "error": "Failed to extract background patch (image may be too small or cropped improperly)."
        }

    # 5. Compute Wasserstein Distance
    distance = float(wasserstein_distance(doc_hist, reference_hist))

    # 6. Normalize anomaly score (0.0 = completely normal, 1.0 = highly anomalous)
    score = min(distance / (threshold * 2.0), 1.0) if threshold > 0 else 1.0
    passed = distance <= threshold

    return {
        "layer_name": "Layer 1: Background Texture Analysis",
        "passed": bool(passed),
        "score": round(score, 5),
        "distance": round(distance, 7),
        "threshold": round(threshold, 7),
        "verdict": "GENUINE_TEXTURE" if passed else "ANOMALOUS_TEXTURE"
    }


if __name__ == '__main__':
    print("=" * 60)
    print("CALIBRATING AND TESTING LAYER 1")
    print("=" * 60)
    ref, meta = calibrate_layer1()
    print("\nCalibration Summary:")
    print(json.dumps(meta, indent=4))

    # Test sample real and fake
    sample_reals = get_image_files(REALS_DIR)
    sample_fakes = get_image_files(FAKES_DIR)

    if sample_reals:
        real_sample_path = os.path.join(REALS_DIR, sample_reals[0])
        res_real = evaluate_background_texture(real_sample_path)
        print(f"\nSample Real Test ({sample_reals[0]}):")
        print(json.dumps(res_real, indent=4))

    if sample_fakes:
        fake_sample_path = os.path.join(FAKES_DIR, sample_fakes[0])
        res_fake = evaluate_background_texture(fake_sample_path)
        print(f"\nSample Fake Test ({sample_fakes[0]}):")
        print(json.dumps(res_fake, indent=4))