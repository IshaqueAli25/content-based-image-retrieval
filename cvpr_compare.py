# cvpr_compare.py
import numpy as np

"""
cvpr_compare.py
Vectorized distance and scoring functions for visual search.
Supports: euclidean, l1, cosine, chi_square (vectorized), histogram_intersection.
Functions:
 - cvpr_compare_single(f1, f2, method)
 - cvpr_compare_batch(query, X, method)  # faster for many comparisons
 - score_from_distance(distances, invert=True)  # convert distances -> scores for AP
"""

EPS = 1e-10

def cvpr_compare_single(F1, F2, method='euclidean'):
    f1 = F1.flatten()
    f2 = F2.flatten()

    if method == 'euclidean':
        return np.sqrt(np.sum((f1 - f2) ** 2))
    elif method == 'l1':
        return np.sum(np.abs(f1 - f2))
    elif method == 'cosine':
        n1 = np.linalg.norm(f1)
        n2 = np.linalg.norm(f2)
        if n1 == 0 or n2 == 0:
            return 1.0
        cos_sim = np.dot(f1, f2) / (n1 * n2)
        return 1.0 - cos_sim
    elif method == 'chi_square':
        num = (f1 - f2) ** 2
        den = f1 + f2 + EPS
        return 0.5 * np.sum(num / den)
    elif method == 'hist_intersection':
        # distance = -sum(min(a,b)) -> smaller is better so we return negative intersection as distance
        return -np.sum(np.minimum(f1, f2))
    else:
        raise ValueError(f"Unknown distance method: {method}")

def cvpr_compare_batch(query, X, method='euclidean'):
    """
    Fast batch comparison between a single query (D,) and X (N,D).
    Returns a numpy array of distances shape (N,).
    """
    q = query.flatten()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if method == 'euclidean':
        dif = X - q  # (N,D)
        return np.sqrt(np.sum(dif * dif, axis=1))
    elif method == 'l1':
        return np.sum(np.abs(X - q), axis=1)
    elif method == 'cosine':
        qn = np.linalg.norm(q) + EPS
        Xn = np.linalg.norm(X, axis=1) + EPS
        dots = X.dot(q)
        cos_sim = dots / (Xn * qn)
        return 1.0 - cos_sim
    elif method == 'chi_square':
        # 0.5 * sum((x - q)^2 / (x + q + eps))
        num = (X - q) ** 2
        den = X + q + EPS
        return 0.5 * np.sum(num / den, axis=1)
    elif method == 'hist_intersection':
        # Return negative intersection (so smaller = worse); we can later negate for scores
        inter = np.sum(np.minimum(X, q), axis=1)
        return -inter
    else:
        raise ValueError(f"Unknown distance method: {method}")

def score_from_distance(distances, invert=True):
    """
    Turn distances into scores suitable for average_precision_score (higher better).
    If invert=True: returns -distances (so smaller distances => higher scores).
    For histogram_intersection where distances are negative intersections, we return -distances as positive intersection scores.
    """
    distances = np.array(distances)
    if invert:
        return -distances
    return distances

def distance_to_scores(distances, method='chi_square'):
    """
    Convert distances to similarity scores (higher = better).
    For histogram-based distances (chi-square, l1, l2, cosine),
    lower distance means more similar, so we invert sign.
    For histogram intersection, we keep positive since it's already similarity.
    """
    d = np.asarray(distances)
    if method == 'hist_intersection':
        return -d  # since intersection already gives similarity
    return -d  # for others: smaller distance → higher score
