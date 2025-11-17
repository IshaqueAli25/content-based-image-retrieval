# extractSpatialGrid.py
import numpy as np
import cv2

def extractSpatialGrid(img, grid_rows=2, grid_cols=2, bins_per_channel=4,
                       use_texture=False, orientation_bins=8, colorspace='rgb',
                       normalize='l1'):
    """
    img: float image in [0,1]
    returns: 1xD descriptor
    """
    h, w, _ = img.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    descriptors = []

    # Convert color space once if needed
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if colorspace.lower() == 'hsv':
        img_conv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    elif colorspace.lower() == 'lab':
        img_conv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    else:
        img_conv = img.astype(np.float32)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < grid_rows - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < grid_cols - 1 else w

            cell = img_conv[y0:y1, x0:x1, :]

            color_hist = _color_hist_cell(cell, bins_per_channel)
            descriptors.append(color_hist)

            if use_texture:
                tex = _texture_hist_cell(cell, orientation_bins)
                descriptors.append(tex)

    F = np.concatenate(descriptors).astype(np.float32).reshape(1, -1)

    # normalize vector
    if normalize == 'l1':
        F = F / (np.sum(np.abs(F), axis=1, keepdims=True) + 1e-10)
    elif normalize == 'l2':
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-10)

    return F

def _color_hist_cell(cell, bins_per_channel):
    h, w, _ = cell.shape
    total_bins = bins_per_channel ** 3
    histogram = np.zeros((total_bins,), dtype=np.float32)

    quantized = (cell * (bins_per_channel - 1e-6)).astype(np.int32)
    quantized = np.clip(quantized, 0, bins_per_channel - 1)

    R = quantized[:, :, 0].flatten()
    G = quantized[:, :, 1].flatten()
    B = quantized[:, :, 2].flatten()

    indices = R * (bins_per_channel**2) + G * bins_per_channel + B
    for idx in indices:
        histogram[idx] += 1.0

    if h * w > 0:
        histogram = histogram / (h * w)

    return histogram

def _texture_hist_cell(cell, orientation_bins=8):
    # compute gradient orientation histogram (magnitude-weighted)
    # cell is in [0,1] float
    gray = 0.299 * cell[:, :, 0] + 0.587 * cell[:, :, 1] + 0.114 * cell[:, :, 2]
    gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    grad_x = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    orient = np.arctan2(grad_y, grad_x)  # -pi..pi
    orient = (orient + np.pi) % (2 * np.pi)
    bin_size = 2 * np.pi / orientation_bins
    q = (orient / bin_size).astype(np.int32)
    q = np.clip(q, 0, orientation_bins - 1)

    hist = np.zeros((orientation_bins,), dtype=np.float32)
    flat_q = q.flatten()
    flat_mag = mag.flatten()
    for b, m in zip(flat_q, flat_mag):
        hist[b] += m

    if np.sum(hist) > 0:
        hist = hist / (np.sum(hist) + 1e-10)

    return hist
