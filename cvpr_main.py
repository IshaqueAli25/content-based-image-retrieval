# cvpr_visualsearch_browser.py
import os
import numpy as np
import scipy.io as sio
import cv2
import tkinter as tk
from tkinter import filedialog
from cvpr_compare import cvpr_compare_batch
from extractSpatialGrid import extractSpatialGrid
from extractGlobalColorHistogram import extractGlobalColorHistogram
from sklearn.decomposition import PCA

# CONFIG#######################

DESCRIPTOR_FOLDER = r'C:\Users\Ishaque\Desktop\0.CV CW\descriptors'
DESCRIPTOR_SUBFOLDER = 'spatial_2x2_4bins_rgb'  # global_8bins_rgb or spatial_2x2_4bins_rgb
DATASET_FOLDER = r'C:\Users\Ishaque\Desktop\0.CV CW\MSRC_ObjCategImageDatabase_v2'
DISTANCE_METHOD = 'chi_square'
NUM_RESULTS = 10
USE_PCA = False
PCA_COMPONENTS = 50

###############################

# Load all descriptors
descriptor_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)
if not os.path.exists(descriptor_path):
    raise FileNotFoundError(f"Descriptor folder not found: {descriptor_path}")

ALLFEAT = []
ALLFILES = []

print("Loading descriptors...")
for fname in sorted(os.listdir(descriptor_path)):
    if not fname.endswith('.mat'):
        continue
    data = sio.loadmat(os.path.join(descriptor_path, fname))
    imgfile = fname.replace('.mat', '.bmp')
    ALLFILES.append(os.path.join(DATASET_FOLDER, 'Images', imgfile))
    ALLFEAT.append(data['F'].flatten())

ALLFEAT = np.array(ALLFEAT, dtype=np.float32)
N, D = ALLFEAT.shape

# L2 normalize
ALLFEAT = ALLFEAT / (np.linalg.norm(ALLFEAT, axis=1, keepdims=True) + 1e-10)

if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS, whiten=False, random_state=42)
    ALLFEAT_PCA = pca.fit_transform(ALLFEAT)
else:
    ALLFEAT_PCA = None

print(f"Loaded {N} descriptors with dimension {D}")


def compute_query_descriptor(image_path):
    """Compute descriptor that matches stored descriptors automatically."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sample_dim = ALLFEAT.shape[1]

    # Decide based on stored descriptor dimension
    if sample_dim == 512:
        # global RGB histogram (8 bins per channel)
        descriptor = extractGlobalColorHistogram(img_rgb, bins_per_channel=8,
                                                 colorspace='rgb', normalize='l2').flatten()
    elif sample_dim == 256:
        # 2x2 grid, 4 bins per channel
        descriptor = extractSpatialGrid(img_rgb, grid_rows=2, grid_cols=2,
                                        bins_per_channel=4, use_texture=False,
                                        colorspace='rgb', normalize='l2').flatten()
    elif sample_dim == 2048:
        # 2x2 grid, 8 bins per channel
        descriptor = extractSpatialGrid(img_rgb, grid_rows=2, grid_cols=2,
                                        bins_per_channel=8, use_texture=False,
                                        colorspace='rgb', normalize='l2').flatten()
    else:
        raise RuntimeError(f"Unsupported descriptor dimension: {sample_dim}")

    descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-10)
    return descriptor


def search_similar_images(query_path):
    """Search using cvpr_compare_batch"""
    print(f"\nSearching for: {query_path}")
    query_desc = compute_query_descriptor(query_path)

    if USE_PCA and ALLFEAT_PCA is not None:
        query_desc = pca.transform(query_desc.reshape(1, -1))[0]
        feats = ALLFEAT_PCA
    else:
        feats = ALLFEAT

    dists = cvpr_compare_batch(query_desc, feats, method=DISTANCE_METHOD)
    order = np.argsort(dists)[:NUM_RESULTS]

    results = []
    for idx in order:
        results.append({
            'path': ALLFILES[idx],
            'distance': dists[idx],
            'class': os.path.basename(ALLFILES[idx]).split('_')[0]
        })
    return results


def display_results_grid(query_path, results):
    """Display query + top 10 results in single window"""
    rows, cols = 3, 4
    cell_width, cell_height = 200, 200
    padding = 10

    canvas_width = cols * (cell_width + padding) + padding
    canvas_height = rows * (cell_height + padding) + padding + 50
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    query_img = cv2.imread(query_path)
    if query_img is not None:
        query_resized = cv2.resize(query_img, (cell_width, cell_height))
        y_start = padding + 30
        x_start = padding
        canvas[y_start:y_start+cell_height, x_start:x_start+cell_width] = query_resized
        cv2.putText(canvas, "QUERY", (x_start+10, y_start-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for i, result in enumerate(results):
        pos = i + 1
        row = pos // cols
        col = pos % cols
        y_start = row * (cell_height + padding) + padding + 30
        x_start = col * (cell_width + padding) + padding
        img = cv2.imread(result['path'])
        if img is not None:
            img_resized = cv2.resize(img, (cell_width, cell_height))
            canvas[y_start:y_start+cell_height, x_start:x_start+cell_width] = img_resized
            label = f"#{i+1} {result['class']}"
            cv2.putText(canvas, label, (x_start+5, y_start-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    title = f"Visual Search Results - Top {NUM_RESULTS} ({DISTANCE_METHOD})"
    cv2.putText(canvas, title, (padding, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("Visual Search Results", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# browsing code

def browse_and_search():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Query Image",
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png"), ("All files", "*.*")],
        initialdir=os.path.join(DATASET_FOLDER, 'Images')
    )
    if file_path:
        try:
            results = search_similar_images(file_path)
            query_name = os.path.basename(file_path)
            query_class = query_name.split('_')[0] if '_' in query_name else 'unknown'
            print(f"\nQuery: {query_name} (class: {query_class})")
            print(f"\n{'Rank':<6}{'File':<30}{'Class':<10}{'Distance':<12}{'Match'}")
            print("-" * 70)
            matches = 0
            for i, res in enumerate(results):
                name = os.path.basename(res['path'])
                match = '✓' if res['class'] == query_class else '✗'
                if res['class'] == query_class:
                    matches += 1
                print(f"{i+1:<6}{name:<30}{res['class']:<10}{res['distance']:<12.6f}{match}")
            precision = matches / NUM_RESULTS
            print(f"\nPrecision @ {NUM_RESULTS}: {precision:.4f}")
            display_results_grid(file_path, results)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No file selected")


if __name__ == "__main__":
    print("=" * 70)
    print("Visual Search with File Browser")
    print("=" * 70)
    print(f"Descriptor: {DESCRIPTOR_SUBFOLDER}")
    print(f"Distance: {DISTANCE_METHOD}")
    print(f"Results: {NUM_RESULTS}")
    print(f"PCA: {'Yes (' + str(PCA_COMPONENTS) + 'D)' if USE_PCA else 'No'}")
    print("=" * 70)
    browse_and_search()

# =========================================================
# EXTRA VISUALIZATION EXAMPLES (for coursework documentation)
# =========================================================
import matplotlib.pyplot as plt
from extractGlobalColorHistogram import extractGlobalColorHistogram
from extractSpatialGrid import extractSpatialGrid

def generate_visual_examples():
    example_image = os.path.join(DATASET_FOLDER, 'Images', '2_13_s.bmp')  # you can pick any
    save_dir = os.path.join(os.getcwd(), 'visual_examples')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating visual examples from: {example_image}")
    img = cv2.imread(example_image)
    if img is None:
        print("❌ Could not load example image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 1️⃣ GLOBAL COLOR HISTOGRAM
    hist = extractGlobalColorHistogram(img_rgb, bins_per_channel=8, colorspace='rgb', normalize='l2').flatten()
    plt.figure(figsize=(8, 4))
    plt.plot(hist)
    plt.title("Global RGB Color Histogram (8 bins/channel)")
    plt.xlabel("Bin Index")
    plt.ylabel("Normalized Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_histogram_example.png"), dpi=300)
    plt.close()

    cv2.imwrite(os.path.join(save_dir, "global_histogram_input.png"), cv2.cvtColor(img_rgb * 255, cv2.COLOR_RGB2BGR))

    # 2️⃣ SPATIAL 2x2 GRID OVERLAY
    grid_img = img.copy()
    h, w = grid_img.shape[:2]
    cv2.line(grid_img, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)
    cv2.line(grid_img, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, "spatial_grid_overlay.png"), grid_img)

    print(f"✅ Saved images in: {save_dir}")
    print(" - global_histogram_input.png")
    print(" - global_histogram_example.png")
    print(" - spatial_grid_overlay.png")

# Uncomment next line to run automatically once
generate_visual_examples()
