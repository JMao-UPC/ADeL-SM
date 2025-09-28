import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config.config import PATCH_SIZE
from pathlib import Path


class PatchDataset(Dataset):
    """
    Dataset class for loading training patches from .npy files.
    Each sample is a random patch from the input (noisy frame) and target (clean label) pairs.

    Args:
        input_dir (str): Path to directory containing input .npy files (noisy frames).
        target_dir (str): Path to directory containing target .npy files (clean labels).
        patch_size (int): Size of square patches to extract (default from config).
    """

    def __init__(self, input_dir, target_dir, patch_size=PATCH_SIZE):
        # Validate input/target directories
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        # Get sorted list of .npy files (ensure input/target order matches)
        self.input_files = sorted([f for f in self.input_dir.glob("*.npy") if f.is_file()])
        self.target_files = sorted([f for f in self.target_dir.glob("*.npy") if f.is_file()])

        # Check if input and target file counts match
        if len(self.input_files) != len(self.target_files):
            raise ValueError(
                f"Mismatched file counts: {len(self.input_files)} input files vs {len(self.target_files)} target files"
            )

        self.patch_size = patch_size
        # Check if patch size is valid (positive integer)
        if self.patch_size <= 0 or not isinstance(self.patch_size, int):
            raise ValueError(f"Invalid patch size: {self.patch_size}. Must be a positive integer.")

    def __len__(self):
        """Return total number of patches (10 patches per frame to augment data)."""
        return len(self.input_files) * 10

    def __getitem__(self, idx):
        """
        Extract a random patch from the corresponding input-target frame pair.

        Args:
            idx (int): Index of the patch to extract.

        Returns:
            tuple: (input_patch, target_patch) -> both are torch tensors (1 x H x W).
        """
        # Map patch index to frame index (10 patches per frame)
        frame_idx = idx // 10
        input_path = self.input_files[frame_idx]
        target_path = self.target_files[frame_idx]

        # Load .npy files (2D frames: H x W)
        input_frame = np.load(input_path).astype(np.float32)
        target_frame = np.load(target_path).astype(np.float32)

        # Check if frame size is larger than patch size
        frame_h, frame_w = input_frame.shape
        if frame_h < self.patch_size or frame_w < self.patch_size:
            raise ValueError(
                f"Frame size ({frame_h}x{frame_w}) is smaller than patch size ({self.patch_size}x{self.patch_size})"
            )

        # Randomly select top-left corner of the patch
        rand_h = np.random.randint(0, frame_h - self.patch_size)
        rand_w = np.random.randint(0, frame_w - self.patch_size)

        # Extract patches
        input_patch = input_frame[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size]
        target_patch = target_frame[rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size]

        # Add channel dimension (1 x H x W) and convert to torch tensor
        input_patch = torch.from_numpy(input_patch).unsqueeze(0)
        target_patch = torch.from_numpy(target_patch).unsqueeze(0)

        return input_patch, target_patch