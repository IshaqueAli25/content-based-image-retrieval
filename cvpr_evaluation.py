# cvpr_evaluation.py
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cvpr_compare import cvpr_compare_batch, score_from_distance
from sklearn.metrics import average_precision_score

# CONFIG
DESCRIPTOR_FOLDER = r'C:\Users\Ishaque\Desktop\0.CV CW\descriptors'
DESCRIPTOR_SUBFOLDER = 'spatial_2x2_4bins_rgb'  # adjust to your out folder
DISTANCE_METHOD = 'chi_square'  # 'euclidean', 'l1', 'cosine', 'chi_square', 'hist_intersection'
TOP_K = 20

descriptor_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)
if not os.path.exists(descriptor_path):
    raise FileNotFoundError(f"Descriptor folder not found: {descriptor_path}")

ALLFEAT = []
ALLFILES = []
for fname in sorted(os.listdir(descriptor_path)):
    if not fname.endswith('.mat'):
        continue
    data = sio.loadmat(os.path.join(descriptor_path, fname))
    imgname = fname.replace('.mat', '.bmp')
    ALLFILES.append(imgname)
    ALLFEAT.append(data['F'].flatten())

ALLFEAT = np.array(ALLFEAT, dtype=np.float32)
NIMG, DIM = ALLFEAT.shape

print(f"Loaded {NIMG} descriptors, dimension {DIM}")

# ensure normalized rows (L2)
norms = np.linalg.norm(ALLFEAT, axis=1, keepdims=True) + 1e-10
ALLFEAT = ALLFEAT / norms

# helper to get class from filename
def cls_from_name(fn):
    return fn.split('_')[0]

all_classes = [cls_from_name(f) for f in ALLFILES]

average_precisions = []
confusion = {}
# we'll build confusion counts for top-1 (excluding query)
unique_classes = sorted(list(set(all_classes)))
class_to_idx = {c:i for i,c in enumerate(unique_classes)}
confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

for q_idx in range(NIMG):
    query = ALLFEAT[q_idx]
    q_class = all_classes[q_idx]

    dists = cvpr_compare_batch(query, ALLFEAT, method=DISTANCE_METHOD)
    # exclude the query itself for ranking by set to +inf (worst)
    dists[q_idx] = np.inf

    # scores for AP: higher = better
    scores = score_from_distance(dists, invert=True)

    # ground-truth vector
    y_true = np.array([1 if c == q_class else 0 for c in all_classes], dtype=int)
    # average precision (sklearn handles all ordering)
    try:
        ap = average_precision_score(y_true, scores)
    except Exception:
        ap = 0.0
    average_precisions.append(ap)

    # top-1 predicted
    top1 = np.argmin(dists)  # because smaller distance is better
    pred_cls = all_classes[top1]
    confusion_matrix[class_to_idx[q_class], class_to_idx[pred_cls]] += 1

    if (q_idx+1) % 100 == 0:
        print(f"Processed {q_idx+1}/{NIMG}")

MAP = np.mean(average_precisions)
print(f"\nMAP: {MAP:.4f}")

# Precision @ k
# compute avg precision at ranks across queries: compute for top k
prec_at_k = []
for k in [5,10,20]:
    precisions = []
    for q_idx in range(NIMG):
        query = ALLFEAT[q_idx]
        q_class = all_classes[q_idx]
        dists = cvpr_compare_batch(query, ALLFEAT, method=DISTANCE_METHOD)
        dists[q_idx] = np.inf
        order = np.argsort(dists)[:k]
        retrieved = [all_classes[i] for i in order]
        prec = sum(1 for r in retrieved if r == q_class) / k
        precisions.append(prec)
    prec_at_k.append(np.mean(precisions))

print(f"P@5: {prec_at_k[0]:.4f}, P@10: {prec_at_k[1]:.4f}, P@20: {prec_at_k[2]:.4f}")

# Save results file
out_txt = f'results_{DESCRIPTOR_SUBFOLDER}_{DISTANCE_METHOD}.txt'
with open(out_txt, 'w') as f:
    f.write("Visual Search Results\n")
    f.write("="*60 + "\n")
    f.write(f"Descriptor: {DESCRIPTOR_SUBFOLDER}\n")
    f.write(f"Distance: {DISTANCE_METHOD}\n")
    f.write(f"Dimension: {DIM}\n\n")
    f.write(f"MAP: {MAP:.4f}\n")
    f.write(f"P@5: {prec_at_k[0]:.4f}\n")
    f.write(f"P@10: {prec_at_k[1]:.4f}\n")
    f.write(f"P@20: {prec_at_k[2]:.4f}\n")

print(f"Saved: {out_txt}")

# PR curve averaged (we'll compute recall axis from thresholds)
# For plotting a single averaged PR curve, a simple approach is average precision at discrete recall steps.
# Here we will compute precision averaged across queries at each recall point computed from the ranked lists.

max_k = 50
avg_precisions_curve = np.zeros(max_k)
avg_recalls_curve = np.zeros(max_k)

for q_idx in range(NIMG):
    query = ALLFEAT[q_idx]
    q_class = all_classes[q_idx]
    dists = cvpr_compare_batch(query, ALLFEAT, method=DISTANCE_METHOD)
    dists[q_idx] = np.inf
    order = np.argsort(dists)
    relevant = [1 if all_classes[i] == q_class else 0 for i in order]
    total_relevant = sum(relevant)
    if total_relevant == 0:
        continue
    cum_rel = np.cumsum(relevant[:max_k])
    ranks = np.arange(1, max_k+1)
    precisions = cum_rel / ranks
    recalls = cum_rel / total_relevant
    avg_precisions_curve += precisions
    avg_recalls_curve += recalls

# normalize by number of queries
avg_precisions_curve = avg_precisions_curve / NIMG
avg_recalls_curve = avg_recalls_curve / NIMG

plt.figure(figsize=(8,6))
plt.plot(avg_recalls_curve, avg_precisions_curve, '-o', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve\n{DESCRIPTOR_SUBFOLDER} + {DISTANCE_METHOD}')
plt.grid(True, alpha=0.3)
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig(f'PR_curve_{DESCRIPTOR_SUBFOLDER}_{DISTANCE_METHOD}.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved PR curve")

# Confusion matrix (normalized)
conf_norm = confusion_matrix.astype(float)
row_sums = conf_norm.sum(axis=1, keepdims=True) + 1e-10
conf_norm = conf_norm / row_sums

plt.figure(figsize=(12,10))
plt.imshow(conf_norm, interpolation='nearest', cmap='YlOrRd')
plt.title(f'Confusion Matrix (Normalized) - {DISTANCE_METHOD}')
plt.colorbar(label='Classification Rate')
ticks = np.arange(len(unique_classes))
plt.xticks(ticks, unique_classes, rotation=90)
plt.yticks(ticks, unique_classes)
for i in range(len(unique_classes)):
    for j in range(len(unique_classes)):
        val = confusion_matrix[i,j]
        rate = conf_norm[i,j]
        if val > 0:
            color = 'white' if rate > 0.5 else 'black'
            plt.text(j, i, f"{val}\n({rate:.2f})", ha='center', va='center', color=color, fontsize=7)
plt.tight_layout()
plt.savefig(f'confusion_matrix_{DESCRIPTOR_SUBFOLDER}_{DISTANCE_METHOD}.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved confusion matrix")

# per-class accuracy bar plot
class_acc = []
for i,c in enumerate(unique_classes):
    total = confusion_matrix[i,:].sum()
    corr = confusion_matrix[i,i]
    acc = corr / (total + 1e-10)
    class_acc.append((c, acc, corr, total))

class_acc.sort(key=lambda x: x[1], reverse=True)
classes_sorted = [x[0] for x in class_acc]
accs_sorted = [x[1] for x in class_acc]
colors = ['green' if a>0.6 else 'orange' if a>0.4 else 'red' for a in accs_sorted]
plt.figure(figsize=(12,6))
plt.barh(classes_sorted, accs_sorted, color=colors, edgecolor='black')
plt.xlabel('Classification Accuracy')
plt.xlim([0,1])
plt.title(f'Per-Class Accuracy\n{DESCRIPTOR_SUBFOLDER} + {DISTANCE_METHOD}')
plt.tight_layout()
plt.savefig(f'class_accuracy_{DESCRIPTOR_SUBFOLDER}_{DISTANCE_METHOD}.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved per-class accuracy")
print("EVALUATION COMPLETE")
