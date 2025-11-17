# cvpr_computedescriptors.py
import os
import numpy as np
import cv2
import scipy.io as sio

# CONFIGURATION - edit as needed

###############

DESCRIPTOR_TYPE = 'global'  #choose  'global' or 'spatial'
DATASET_FOLDER = r'C:\Users\Ishaque\Desktop\0.CV CW\MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = r'C:\Users\Ishaque\Desktop\0.CV CW\descriptors'
BINS_PER_CHANNEL_GLOBAL = 8
GRID_ROWS = 2
GRID_COLS = 2
BINS_PER_CHANNEL_SPATIAL = 4
USE_TEXTURE = False
COLORSPACE = 'rgb'  # 'rgb', 'hsv'
NORMALIZE = 'l2'  # 'l1', 'l2', or None

#############

# Output subfolder
if DESCRIPTOR_TYPE == 'global':
    OUT_SUBFOLDER = f'global_{BINS_PER_CHANNEL_GLOBAL}bins_{COLORSPACE}'
else:
    tex_suf = '_texture' if USE_TEXTURE else ''
    OUT_SUBFOLDER = f'spatial_{GRID_ROWS}x{GRID_COLS}_{BINS_PER_CHANNEL_SPATIAL}bins_{COLORSPACE}{tex_suf}'

# Import extractors
if DESCRIPTOR_TYPE == 'global':
    from extractGlobalColorHistogram import extractGlobalColorHistogram as extractor
else:
    from extractSpatialGrid import extractSpatialGrid as extractor

print("="*80)
print("DESCRIPTOR EXTRACTION")
print("="*80)
print(f"Descriptor type: {DESCRIPTOR_TYPE}")
print(f"Output folder: {os.path.join(OUT_FOLDER, OUT_SUBFOLDER)}")
print("="*80)

os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

images_folder = os.path.join(DATASET_FOLDER, 'Images')
if not os.path.exists(images_folder):
    raise FileNotFoundError(f"Images folder not found: {images_folder}")

count = 0
for filename in sorted(os.listdir(images_folder)):
    if not filename.lower().endswith('.bmp'):
        continue
    count += 1
    img_path = os.path.join(images_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: cannot read {filename}")
        continue
    # OpenCV reads BGR -> convert to RGB and scale to [0,1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if DESCRIPTOR_TYPE == 'global':
        F = extractor(img_rgb, bins_per_channel=BINS_PER_CHANNEL_GLOBAL,
                      colorspace=COLORSPACE, normalize=NORMALIZE)
    else:
        F = extractor(img_rgb, grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
                      bins_per_channel=BINS_PER_CHANNEL_SPATIAL,
                      use_texture=USE_TEXTURE, colorspace=COLORSPACE, normalize=NORMALIZE)
    # Save
    fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
    sio.savemat(fout, {'F': F})
    if count % 100 == 0:
        print(f"Processed {count} images...")

print("="*80)
print(f"Done. Processed {count} images.")
print(f"Saved descriptors to: {os.path.join(OUT_FOLDER, OUT_SUBFOLDER)}")
print("="*80)
