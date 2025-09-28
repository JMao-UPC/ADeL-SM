import os
from pathlib import Path

# -------------------------- Path Configuration --------------------------
# Replace with your raw TIFF video path (e.g., "D:/data/11.tif")
RAW_VIDEO_PATH = str(Path(r"D:/data/11.tif").resolve())
# Temporary folder to save training data (input/target .npy files)
TEMP_TRAIN_FOLDER = str(Path(os.path.expanduser("~")) / "Desktop" / "tempImages")

# -------------------------- Blob Detection Configuration --------------------------
DETECT_THRESHOLD = 6          # Threshold for blob detection (higher = stricter, recommended: 3-10)
DETECT_WINDOW_SIZE = 11       # Window size for local max detection (must be odd, recommended: 7-15)
PIXEL_SIZE = 106              # Physical size of each pixel (unit: nm, adjust to your experiment)
BLOB_BLOCK_RADIUS = 2         # Radius of blob region in labels (2 = 5x5 pixel block, recommended: 1-3)

# -------------------------- Normalization Configuration --------------------------
NORMALIZE_MODE = 1            # Normalization mode: 1=per-frame, 2=global, 3=clipped
NORMALIZE_SCALE = 10          # Upper bound of normalized pixel values (default: 10, recommended: 5-20)
CLIP_PERCENT = 0.001          # Only for MODE=3: clip extreme values (recommended: 0.001-0.01)

# -------------------------- Training Configuration --------------------------
TRAIN_FRAME_NUM = 101         # Number of frames used for training (first N frames, recommended: 50-200)
AVG_FRAME_NUM = 2             # Number of frames to average for blob detection (recommended: 2-5)
PATCH_SIZE = 64               # Size of training patches (must be power of 2, recommended: 32/64/128)
BATCH_SIZE = 8                # Batch size (reduce if GPU memory is insufficient, recommended: 4-16)
TRAIN_EPOCHS = 5              # Number of training epochs (recommended: 5-20, stop when loss plateaus)
LEARNING_RATE = 1e-3          # Learning rate (default: 1e-3, reduce to 1e-4 for late training)