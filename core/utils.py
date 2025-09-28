import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from config.config import (
    DETECT_THRESHOLD, DETECT_WINDOW_SIZE, PIXEL_SIZE,
    NORMALIZE_MODE, NORMALIZE_SCALE, CLIP_PERCENT
)


def detect_blobs(im, threshold=DETECT_THRESHOLD, window_size=DETECT_WINDOW_SIZE, pixel_size=PIXEL_SIZE):
    """
    Blob detection function (aligned with MATLAB's iQ_pkfnd logic).
    Detects local maxima (blobs) in the input image after denoising and background subtraction.

    Args:
        im (np.ndarray): Input 2D image (H x W).
        threshold (float): Detection threshold (mean + threshold * std of 80% data).
        window_size (int): Window size for local maximum detection (must be odd).
        pixel_size (float): Physical size of each pixel (for Gaussian sigma calculation).

    Returns:
        np.ndarray: Coordinates of detected blobs (N x 2, [row, column]).
        tuple: Statistics (mean, std, threshold) of background-subtracted image.
        np.ndarray: Background-subtracted image (H x W).
    """
    im = im.astype(np.float32)

    # Gaussian denoising
    sigma = 100 / pixel_size
    im_smoothed = gaussian_filter(im, sigma=sigma)

    # Background fitting and subtraction
    background = gaussian_filter(im_smoothed, sigma=1)
    im_bg_subtracted = im_smoothed - background

    # Local maximum detection (exclude center pixel from window)
    mask = np.ones((window_size, window_size), dtype=bool)
    mask[window_size // 2, window_size // 2] = False  # Exclude center to avoid self-comparison
    local_max = maximum_filter(im_bg_subtracted, footprint=mask, mode='nearest')
    is_local_max = im_bg_subtracted > local_max

    # Calculate adaptive threshold (using 80% of sorted data to avoid outliers)
    sorted_vals = np.sort(im_bg_subtracted.flatten())
    n_80 = int(len(sorted_vals) * 0.8)
    vals_80 = sorted_vals[:n_80]
    bg_mean = np.mean(vals_80)
    bg_std = np.std(vals_80)
    detection_th = bg_mean + threshold * bg_std

    # Filter blobs by threshold
    blob_rows, blob_cols = np.where(is_local_max)
    blobs = np.array([[r, c] for r, c in zip(blob_rows, blob_cols)
                      if im_bg_subtracted[r, c] > detection_th])

    return blobs, (bg_mean, bg_std, detection_th), im_bg_subtracted


def normalize_video(video, mode=NORMALIZE_MODE, scale=NORMALIZE_SCALE, clip_percent=CLIP_PERCENT):
    """
    Normalize a 3D video (H x W x Frames) using specified mode.

    Args:
        video (np.ndarray): Input 3D video (H x W x Frames).
        mode (int): Normalization mode:
            1: Per-frame normalization (each frame scaled to [0, scale]).
            2: Global normalization (all frames scaled by global min/max).
            3: Clipped normalization (remove extreme values first, then scale).
        scale (float): Upper bound of normalized values (lower bound = 0).
        clip_percent (float): For mode=3: percentage of extreme values to clip (0-1).

    Returns:
        np.ndarray: Normalized video (same shape as input).
        list/tuple: Normalization parameters (for denormalization later).
    """
    video = video.astype(np.float32)
    norm_video = np.zeros_like(video, dtype=np.float32)
    params = []

    if mode == 1:
        # Per-frame normalization (each frame has its own min/max)
        for i in range(video.shape[2]):
            frame = video[:, :, i]
            min_val = frame.min()
            max_val = frame.max()
            # Avoid division by zero (if frame is constant)
            if max_val - min_val < 1e-8:
                norm_frame = np.zeros_like(frame)
            else:
                norm_frame = scale * (frame - min_val) / (max_val - min_val)
            norm_video[:, :, i] = norm_frame
            params.append((min_val, max_val))  # Save (min, max) for each frame

    elif mode == 2:
        # Global normalization (all frames use the same min/max)
        global_min = video.min()
        global_max = video.max()
        if global_max - global_min < 1e-8:
            norm_video = np.zeros_like(video)
        else:
            norm_video = scale * (video - global_min) / (global_max - global_min)
        params = (global_min, global_max)  # Save global (min, max)

    elif mode == 3:
        # Clipped normalization (remove extreme values first)
        flat_video = video.flatten()
        clip_upper = np.percentile(flat_video, 100 * (1 - clip_percent))
        clipped_video = np.clip(video, flat_video.min(), clip_upper)
        # Normalize clipped video globally
        clip_min = clipped_video.min()
        clip_max = clipped_video.max()
        if clip_max - clip_min < 1e-8:
            norm_video = np.zeros_like(video)
        else:
            norm_video = scale * (clipped_video - clip_min) / (clip_max - clip_min)
        params = (clip_min, clip_max, clip_upper)  # Save (min, max, clip threshold)

    else:
        raise ValueError(f"Invalid normalization mode: {mode}. Must be 1, 2, or 3.")

    return norm_video, params


def denormalize_frame(norm_frame, params, mode=NORMALIZE_MODE, scale=NORMALIZE_SCALE):
    """
    Denormalize a single frame back to original pixel range.

    Args:
        norm_frame (np.ndarray): Normalized 2D frame (H x W).
        params (list/tuple): Normalization parameters from `normalize_video`.
        mode (int): Same normalization mode used for the video.
        scale (float): Same scale used for normalization.

    Returns:
        np.ndarray: Denormalized frame (original pixel range).
    """
    norm_frame = norm_frame.astype(np.float32)

    if mode == 1:
        # Per-frame denormalization (params = (min_val, max_val) for this frame)
        min_val, max_val = params[:2]
        denorm_frame = (norm_frame / scale) * (max_val - min_val) + min_val

    elif mode == 2:
        # Global denormalization (params = (global_min, global_max))
        global_min, global_max = params
        denorm_frame = (norm_frame / scale) * (global_max - global_min) + global_min

    elif mode == 3:
        # Clipped denormalization (params = (clip_min, clip_max, clip_upper))
        clip_min, clip_max, _ = params
        denorm_frame = (norm_frame / scale) * (clip_max - clip_min) + clip_min

    else:
        raise ValueError(f"Invalid normalization mode: {mode}. Must be 1, 2, or 3.")

    return denorm_frame.astype(np.float32)