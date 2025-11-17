# cvpr_pca_analysis_final.py
"""
Robust PCA analysis for Visual Search coursework.

- Loads descriptors from DESCRIPTOR_FOLDER/DESCRIPTOR_SUBFOLDER (expects .mat files with 'F')
- Computes baseline MAP (no PCA) using DISTANCE_METHOD
- Runs PCA for PCA_COMPONENTS and computes MAP using Euclidean and Mahalanobis (diagonal approx)
- Regularizes tiny eigenvalues, guards against Inf/NaN, and saves plot + results
Outputs:
 - pca_performance_comparison.png
 - pca_results.txt
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
import warnings

# Try import from your compare module
try:
    from cvpr_compare import cvpr_compare_batch
    try:
        from cvpr_compare import score_from_distance as _score_from_distance
    except Exception:
        _score_from_distance = None
except Exception as e:
    raise ImportError("Could not import cvpr_compare.cvpr_compare_batch. Make sure cvpr_compare.py is present.") from e

# ----------------- CONFIG - EDIT THESE -----------------
DESCRIPTOR_FOLDER = r'./descriptors'                   # base descriptors folder (relative or absolute)
DESCRIPTOR_SUBFOLDER = 'spatial_2x2_4bins_rgb'         # e.g. 'spatial_2x2_4bins_rgb' or 'global_8bins_rgb'
DISTANCE_METHOD = 'chi_square'                         # baseline distance used in evaluation
PCA_COMPONENTS = [20, 38, 50, 100, 150, 200]          # dims to test
OUT_PNG = 'pca_performance_comparison.png'
OUT_TXT = 'pca_results.txt'
# Numerical safety
EIG_REG_EPS = 1e-4        # add to eigenvalues before inversion (increase if numerical issues)
MAX_INV = 1e5             # clip inverse eigenvalues to avoid overflow
SCORE_REPLACE = -1e12     # replacement for non-finite scores (similarity scores higher=better)
# ------------------------------------------------------

def distance_to_scores(distances, method=DISTANCE_METHOD):
    """Convert distances -> scores (higher better). Uses cvpr_compare helper if available."""
    if _score_from_distance is not None:
        try:
            return _score_from_distance(distances, invert=True)
        except Exception:
            pass
    d = np.asarray(distances)
    if method == 'hist_intersection':
        return -d
    return -d

def load_descriptors(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Descriptor folder not found: {path}")
    mats = sorted([f for f in os.listdir(path) if f.endswith('.mat')])
    if len(mats) == 0:
        raise RuntimeError(f"No .mat files in: {path}")
    feats = []
    files = []
    for m in mats:
        data = sio.loadmat(os.path.join(path, m))
        if 'F' not in data:
            warnings.warn(f"No 'F' in {m}; skipping.")
            continue
        feats.append(np.asarray(data['F']).reshape(-1))
        files.append(m.replace('.mat', '.bmp'))
    X = np.array(feats, dtype=np.float64)
    return X, files

def sanitize_scores(scores):
    """Ensure scores finite and return sanitized array."""
    s = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(s).all():
        s = np.where(np.isfinite(s), s, SCORE_REPLACE)
    return s

def compute_baseline_map(X, classes, method):
    N = X.shape[0]
    aps = []
    valid_q = 0
    for q in range(N):
        if classes.count(classes[q]) <= 1:
            aps.append(0.0)
            continue
        dists = cvpr_compare_batch(X[q], X, method=method)
        dists[q] = np.inf
        scores = distance_to_scores(dists, method=method)
        scores = sanitize_scores(scores)
        y_true = np.array([1 if classes[i] == classes[q] else 0 for i in range(N)], dtype=int)
        try:
            ap = average_precision_score(y_true, scores)
        except Exception:
            ap = 0.0
        aps.append(ap)
        valid_q += 1
    if valid_q == 0:
        raise RuntimeError("No valid queries (no class with >1 example).")
    map_val = np.mean([a for i,a in enumerate(aps) if classes.count(classes[i])>1])
    return map_val

def compute_map_euclidean(Xp, classes):
    N = Xp.shape[0]
    aps = []
    for q in range(N):
        if classes.count(classes[q]) <= 1:
            aps.append(0.0); continue
        diffs = Xp - Xp[q]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists[q] = np.inf
        scores = -dists
        scores = sanitize_scores(scores)
        y_true = np.array([1 if classes[i] == classes[q] else 0 for i in range(N)], dtype=int)
        try:
            ap = average_precision_score(y_true, scores)
        except Exception:
            ap = 0.0
        aps.append(ap)
    return np.mean([a for i,a in enumerate(aps) if classes.count(classes[i])>1])

def compute_map_mahalanobis(Xp, classes, explained_variance, eig_reg_eps=EIG_REG_EPS, max_inv=MAX_INV):
    # diagonal Mahalanobis: inv_diag = 1/(eig + eps), clipped
    eig = np.asarray(explained_variance, dtype=np.float64)
    eig_reg = eig + eig_reg_eps
    inv_diag = 1.0 / eig_reg
    inv_diag = np.clip(inv_diag, 0.0, max_inv).astype(np.float64)

    N = Xp.shape[0]
    aps = []
    skipped = 0
    for q in range(N):
        if classes.count(classes[q]) <= 1:
            aps.append(0.0); continue
        diffs = (Xp - Xp[q]).astype(np.float64)
        d2 = np.sum((diffs ** 2) * inv_diag.reshape(1, -1), axis=1)
        if not np.isfinite(d2).all():
            d2 = np.where(np.isfinite(d2), d2, 1e12)
        d2 = np.maximum(d2, 0.0)
        dists = np.sqrt(d2)
        dists[q] = np.inf
        scores = -dists
        scores = sanitize_scores(scores)
        y_true = np.array([1 if classes[i] == classes[q] else 0 for i in range(N)], dtype=int)
        try:
            ap = average_precision_score(y_true, scores)
        except Exception:
            ap = 0.0
            skipped += 1
        aps.append(ap)
    if skipped:
        print(f"Warning: skipped {skipped} queries due to numeric issues in Mahalanobis.")
    return np.mean([a for i,a in enumerate(aps) if classes.count(classes[i])>1])

def main():
    descriptor_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)
    print("Loading descriptors from:", descriptor_path)
    X, files = load_descriptors(descriptor_path)
    N, D = X.shape
    print(f"Loaded {N} descriptors, dimension {D}")

    # quick sanity
    if not np.isfinite(X).all():
        raise RuntimeError("Descriptor matrix contains NaN or Inf - fix descriptors first.")
    rownorms = np.linalg.norm(X, axis=1)
    print("Row norms: min {:.3e}, median {:.3e}, max {:.3e}".format(rownorms.min(), np.median(rownorms), rownorms.max()))

    # normalize rows (L2) to match evaluation pipeline
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    classes = [os.path.basename(f).split('_')[0] for f in files]
    unique, counts = np.unique(classes, return_counts=True)
    print(f"Number of classes: {len(unique)}; sample counts (first 10): {list(zip(unique[:10], counts[:10]))}")

    # Baseline MAP
    print("\nComputing baseline MAP (no PCA) with distance:", DISTANCE_METHOD)
    baseline_map = compute_baseline_map(X, classes, DISTANCE_METHOD)
    print(f"Baseline MAP: {baseline_map:.6f}")

    results = []
    # center before PCA
    Xc = X - np.mean(X, axis=0)

    for n_comp in PCA_COMPONENTS:
        if n_comp <= 0 or n_comp >= D:
            print(f"Skipping n_comp={n_comp} (invalid for D={D})")
            continue
        print(f"\nPCA -> n_components = {n_comp}")
        pca = PCA(n_components=n_comp, whiten=False, random_state=42)
        Xp = pca.fit_transform(Xc).astype(np.float64)
        if not np.isfinite(Xp).all():
            Xp = np.nan_to_num(Xp, nan=0.0, posinf=1e12, neginf=-1e12)

        map_euc = compute_map_euclidean(Xp, classes)
        print(f"  MAP Euclidean: {map_euc:.6f}")

        map_mah = compute_map_mahalanobis(Xp, classes, pca.explained_variance_)
        print(f"  MAP Mahalanobis: {map_mah:.6f}")

        varsum = float(np.sum(pca.explained_variance_ratio_))
        results.append({'n_comp': n_comp, 'map_euclidean': map_euc, 'map_mahalanobis': map_mah, 'variance': varsum})

    # Save text results
    with open(OUT_TXT, 'w') as f:
        f.write("PCA performance results\n")
        f.write("="*60 + "\n")
        f.write(f"Descriptor folder: {DESCRIPTOR_SUBFOLDER}\n")
        f.write(f"Distance baseline: {DISTANCE_METHOD}\n")
        f.write(f"Baseline MAP: {baseline_map:.6f}\n\n")
        f.write("n_comp\tMAP_Euc\tMAP_Mah\tVarianceSum\n")
        for r in results:
            f.write(f"{r['n_comp']}\t{r['map_euclidean']:.6f}\t{r['map_mahalanobis']:.6f}\t{r['variance']:.6f}\n")
    print("Saved results to", OUT_TXT)

    # Plot
    if len(results) > 0:
        dims = [r['n_comp'] for r in results]
        maps_e = [r['map_euclidean'] for r in results]
        maps_m = [r['map_mahalanobis'] for r in results]

        plt.figure(figsize=(10,5))
        plt.plot(dims, maps_e, 'bo-', label='PCA + Euclidean')
        plt.plot(dims, maps_m, 'rs-', label='PCA + Mahalanobis')
        plt.axhline(y=baseline_map, color='g', linestyle='--', label=f'Baseline: {baseline_map:.4f}')
        plt.xlabel('PCA components')
        plt.ylabel('MAP')
        plt.title('PCA Dimensionality vs Performance (robust)')
        plt.legend()
        plt.grid(True)
        plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved plot to", OUT_PNG)
    else:
        print("No PCA results to plot (no valid components).")

if __name__ == "__main__":
    main()
# cvpr_pca_analysis_final.py
"""
Robust PCA analysis for Visual Search coursework.

- Loads descriptors from DESCRIPTOR_FOLDER/DESCRIPTOR_SUBFOLDER (expects .mat files with 'F')
- Computes baseline MAP (no PCA) using DISTANCE_METHOD
- Runs PCA for PCA_COMPONENTS and computes MAP using Euclidean and Mahalanobis (diagonal approx)
- Regularizes tiny eigenvalues, guards against Inf/NaN, and saves plot + results
Outputs:
 - pca_performance_comparison.png
 - pca_results.txt
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
import warnings

# Try import from your compare module
try:
    from cvpr_compare import cvpr_compare_batch
    try:
        from cvpr_compare import score_from_distance as _score_from_distance
    except Exception:
        _score_from_distance = None
except Exception as e:
    raise ImportError("Could not import cvpr_compare.cvpr_compare_batch. Make sure cvpr_compare.py is present.") from e

# ----------------- CONFIG - EDIT THESE -----------------
DESCRIPTOR_FOLDER = r'./descriptors'                   # base descriptors folder (relative or absolute)
DESCRIPTOR_SUBFOLDER = 'spatial_2x2_4bins_rgb'         # e.g. 'spatial_2x2_4bins_rgb' or 'global_8bins_rgb'
DISTANCE_METHOD = 'chi_square'                         # baseline distance used in evaluation
PCA_COMPONENTS = [20, 38, 50, 100, 150, 200]          # dims to test
OUT_PNG = 'pca_performance_comparison.png'
OUT_TXT = 'pca_results.txt'
# Numerical safety
EIG_REG_EPS = 1e-4        # add to eigenvalues before inversion (increase if numerical issues)
MAX_INV = 1e5             # clip inverse eigenvalues to avoid overflow
SCORE_REPLACE = -1e12     # replacement for non-finite scores (similarity scores higher=better)
# ------------------------------------------------------

def distance_to_scores(distances, method=DISTANCE_METHOD):
    """Convert distances -> scores (higher better). Uses cvpr_compare helper if available."""
    if _score_from_distance is not None:
        try:
            return _score_from_distance(distances, invert=True)
        except Exception:
            pass
    d = np.asarray(distances)
    if method == 'hist_intersection':
        return -d
    return -d

def load_descriptors(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Descriptor folder not found: {path}")
    mats = sorted([f for f in os.listdir(path) if f.endswith('.mat')])
    if len(mats) == 0:
        raise RuntimeError(f"No .mat files in: {path}")
    feats = []
    files = []
    for m in mats:
        data = sio.loadmat(os.path.join(path, m))
        if 'F' not in data:
            warnings.warn(f"No 'F' in {m}; skipping.")
            continue
        feats.append(np.asarray(data['F']).reshape(-1))
        files.append(m.replace('.mat', '.bmp'))
    X = np.array(feats, dtype=np.float64)
    return X, files

def sanitize_scores(scores):
    """Ensure scores finite and return sanitized array."""
    s = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(s).all():
        s = np.where(np.isfinite(s), s, SCORE_REPLACE)
    return s

def compute_baseline_map(X, classes, method):
    N = X.shape[0]
    aps = []
    valid_q = 0
    for q in range(N):
        if classes.count(classes[q]) <= 1:
            aps.append(0.0)
            continue
        dists = cvpr_compare_batch(X[q], X, method=method)
        dists[q] = np.inf
        scores = distance_to_scores(dists, method=method)
        scores = sanitize_scores(scores)
        y_true = np.array([1 if classes[i] == classes[q] else 0 for i in range(N)], dtype=int)
        try:
            ap = average_precision_score(y_true, scores)
        except Exception:
            ap = 0.0
        aps.append(ap)
        valid_q += 1
    if valid_q == 0:
        raise RuntimeError("No valid queries (no class with >1 example).")
    map_val = np.mean([a for i,a in enumerate(aps) if classes.count(classes[i])>1])
    return map_val

def compute_map_euclidean(Xp, classes):
    N = Xp.shape[0]
    aps = []
    for q in range(N):
        if classes.count(classes[q]) <= 1:
            aps.append(0.0); continue
        diffs = Xp - Xp[q]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        dists[q] = np.inf
        scores = -dists
        scores = sanitize_scores(scores)
        y_true = np.array([1 if classes[i] == classes[q] else 0 for i in range(N)], dtype=int)
        try:
            ap = average_precision_score(y_true, scores)
        except Exception:
            ap = 0.0
        aps.append(ap)
    return np.mean([a for i,a in enumerate(aps) if classes.count(classes[i])>1])

def compute_map_mahalanobis(Xp, classes, explained_variance, eig_reg_eps=EIG_REG_EPS, max_inv=MAX_INV):
    # diagonal Mahalanobis: inv_diag = 1/(eig + eps), clipped
    eig = np.asarray(explained_variance, dtype=np.float64)
    eig_reg = eig + eig_reg_eps
    inv_diag = 1.0 / eig_reg
    inv_diag = np.clip(inv_diag, 0.0, max_inv).astype(np.float64)

    N = Xp.shape[0]
    aps = []
    skipped = 0
    for q in range(N):
        if classes.count(classes[q]) <= 1:
            aps.append(0.0); continue
        diffs = (Xp - Xp[q]).astype(np.float64)
        d2 = np.sum((diffs ** 2) * inv_diag.reshape(1, -1), axis=1)
        if not np.isfinite(d2).all():
            d2 = np.where(np.isfinite(d2), d2, 1e12)
        d2 = np.maximum(d2, 0.0)
        dists = np.sqrt(d2)
        dists[q] = np.inf
        scores = -dists
        scores = sanitize_scores(scores)
        y_true = np.array([1 if classes[i] == classes[q] else 0 for i in range(N)], dtype=int)
        try:
            ap = average_precision_score(y_true, scores)
        except Exception:
            ap = 0.0
            skipped += 1
        aps.append(ap)
    if skipped:
        print(f"Warning: skipped {skipped} queries due to numeric issues in Mahalanobis.")
    return np.mean([a for i,a in enumerate(aps) if classes.count(classes[i])>1])

def main():
    descriptor_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)
    print("Loading descriptors from:", descriptor_path)
    X, files = load_descriptors(descriptor_path)
    N, D = X.shape
    print(f"Loaded {N} descriptors, dimension {D}")

    # quick sanity
    if not np.isfinite(X).all():
        raise RuntimeError("Descriptor matrix contains NaN or Inf - fix descriptors first.")
    rownorms = np.linalg.norm(X, axis=1)
    print("Row norms: min {:.3e}, median {:.3e}, max {:.3e}".format(rownorms.min(), np.median(rownorms), rownorms.max()))

    # normalize rows (L2) to match evaluation pipeline
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    classes = [os.path.basename(f).split('_')[0] for f in files]
    unique, counts = np.unique(classes, return_counts=True)
    print(f"Number of classes: {len(unique)}; sample counts (first 10): {list(zip(unique[:10], counts[:10]))}")

    # Baseline MAP
    print("\nComputing baseline MAP (no PCA) with distance:", DISTANCE_METHOD)
    baseline_map = compute_baseline_map(X, classes, DISTANCE_METHOD)
    print(f"Baseline MAP: {baseline_map:.6f}")

    results = []
    # center before PCA
    Xc = X - np.mean(X, axis=0)

    for n_comp in PCA_COMPONENTS:
        if n_comp <= 0 or n_comp >= D:
            print(f"Skipping n_comp={n_comp} (invalid for D={D})")
            continue
        print(f"\nPCA -> n_components = {n_comp}")
        pca = PCA(n_components=n_comp, whiten=False, random_state=42)
        Xp = pca.fit_transform(Xc).astype(np.float64)
        if not np.isfinite(Xp).all():
            Xp = np.nan_to_num(Xp, nan=0.0, posinf=1e12, neginf=-1e12)

        map_euc = compute_map_euclidean(Xp, classes)
        print(f"  MAP Euclidean: {map_euc:.6f}")

        map_mah = compute_map_mahalanobis(Xp, classes, pca.explained_variance_)
        print(f"  MAP Mahalanobis: {map_mah:.6f}")

        varsum = float(np.sum(pca.explained_variance_ratio_))
        results.append({'n_comp': n_comp, 'map_euclidean': map_euc, 'map_mahalanobis': map_mah, 'variance': varsum})

    # Save text results
    with open(OUT_TXT, 'w') as f:
        f.write("PCA performance results\n")
        f.write("="*60 + "\n")
        f.write(f"Descriptor folder: {DESCRIPTOR_SUBFOLDER}\n")
        f.write(f"Distance baseline: {DISTANCE_METHOD}\n")
        f.write(f"Baseline MAP: {baseline_map:.6f}\n\n")
        f.write("n_comp\tMAP_Euc\tMAP_Mah\tVarianceSum\n")
        for r in results:
            f.write(f"{r['n_comp']}\t{r['map_euclidean']:.6f}\t{r['map_mahalanobis']:.6f}\t{r['variance']:.6f}\n")
    print("Saved results to", OUT_TXT)

    # Plot
    if len(results) > 0:
        dims = [r['n_comp'] for r in results]
        maps_e = [r['map_euclidean'] for r in results]
        maps_m = [r['map_mahalanobis'] for r in results]

        plt.figure(figsize=(10,5))
        plt.plot(dims, maps_e, 'bo-', label='PCA + Euclidean')
        plt.plot(dims, maps_m, 'rs-', label='PCA + Mahalanobis')
        plt.axhline(y=baseline_map, color='g', linestyle='--', label=f'Baseline: {baseline_map:.4f}')
        plt.xlabel('PCA components')
        plt.ylabel('MAP')
        plt.title('PCA Dimensionality vs Performance ')
        plt.legend()
        plt.grid(True)
        plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved plot to", OUT_PNG)
    else:
        print("No PCA results to plot (no valid components).")

if __name__ == "__main__":
    main()
