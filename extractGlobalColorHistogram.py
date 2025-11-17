# extractGlobalColorHistogram.py
import numpy as np
import cv2

def extractGlobalColorHistogram(img, bins_per_channel=8, colorspace='rgb', normalize='l1'):
    """
    img: float image in [0,1], shape (H,W,3)
    colorspace: 'rgb', 'hsv', 'lab'
    normalize: 'l1', 'l2', or None
    returns: 1xD array
    """
    # convert to uint8 for OpenCV conversions
    img_uint8 = (np.clip(img, 0, 1) * 255).astype('uint8')

    if colorspace.lower() == 'hsv':
        img_conv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        # map to 0..1
        img_conv = img_conv.astype(np.float32) / 255.0
    elif colorspace.lower() == 'lab':
        img_conv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        img_conv = img_conv.astype(np.float32) / 255.0
    else:
        img_conv = img.astype(np.float32)  # already in [0,1]

    # quantize each channel
    quantized = (img_conv * (bins_per_channel - 1e-6)).astype(np.int32)
    quantized = np.clip(quantized, 0, bins_per_channel - 1)

    h, w, c = img.shape
    total_bins = bins_per_channel ** 3
    hist = np.zeros((total_bins,), dtype=np.float32)

    R = quantized[:, :, 0].flatten()
    G = quantized[:, :, 1].flatten()
    B = quantized[:, :, 2].flatten()
    indices = R * (bins_per_channel**2) + G * bins_per_channel + B

    for idx in indices:
        hist[idx] += 1.0

    # normalize by total pixels
    hist = hist / (h * w + 1e-10)

    # normalize vector
    if normalize == 'l1':
        hist = hist / (np.sum(np.abs(hist)) + 1e-10)
    elif normalize == 'l2':
        hist = hist / (np.linalg.norm(hist) + 1e-10)

    return hist.reshape(1, -1)
