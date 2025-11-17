# content-based-image-retrieval
Computer Vision project implementing image descriptors, visual search, and PCA-based analysis.

**A visual search (CBIR) system using color histogram descriptors, Chi-square distance and PCA**  
Student project (MSc AI) — Mir Ishaque Ali (6931803)

---

## Project summary
This repository implements a content-based image retrieval (CBIR) pipeline for the **MSRC Object Category (v2)** dataset.  
Main features:
- Global color histograms (8 bins per channel)  
- Spatial 2×2 grid color histograms (4 bins per channel)  
- Similarity metrics: Chi-square, Euclidean (L2), Manhattan (L1), histogram intersection  
- PCA dimensionality reduction (Euclidean / Mahalanobis comparison)  
- Evaluation: Precision–Recall, MAP, per-class accuracy, confusion matrix  
- Simple file-browser demo to run visual search and view top-10 results

---

## ⚠️ Dataset license (IMPORTANT)
The **MSRC dataset is NOT included** in this repository due to licensing restrictions.  
Download the dataset from Microsoft Research and place it in the project folder as:

