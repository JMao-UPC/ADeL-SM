import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile

# Import custom modules
from config.config import *
from core.model import UNet
from core.dataset import PatchDataset
from core.utils import detect_blobs, normalize_video, denormalize_frame


def main():
    """Main workflow: Data Preparation → Model Training → Full Video Inference."""
    # -------------------------- 1. Initialize Paths --------------------------
    input_dir = Path(TEMP_TRAIN_FOLDER) / "input"
    target_dir = Path(TEMP_TRAIN_FOLDER) / "target"
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------- 2. Device Configuration --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device initialized: {device} (CUDA available: {torch.cuda.is_available()})")

    # -------------------------- 3. Load Raw Video --------------------------
    print("\n📥 Loading raw TIFF video...")
    if not Path(RAW_VIDEO_PATH).exists():
        raise FileNotFoundError(f"Raw video not found at: {RAW_VIDEO_PATH}")

    # Load multi-frame TIFF (shape: Frames x H x W → convert to H x W x Frames)
    raw_video = tifffile.imread(RAW_VIDEO_PATH)
    if raw_video.ndim == 3:
        # Convert from (Frames, H, W) to (H, W, Frames)
        raw_video = np.transpose(raw_video, (1, 2, 0))
    elif raw_video.ndim == 2:
        # Single frame → expand to 3D (H, W, 1)
        raw_video = raw_video[:, :, np.newaxis]
    else:
        raise ValueError(
            f"Unsupported video dimension: {raw_video.ndim}. Expected 2 (single frame) or 3 (multi-frame).")

    total_frames = raw_video.shape[2]
    frame_h, frame_w = raw_video.shape[0], raw_video.shape[1]
    print(f"✅ Raw video loaded: {frame_h}x{frame_w} pixels, {total_frames} frames")

    # -------------------------- 4. Prepare Training Data --------------------------
    print("\n📊 Preparing training data (first {} frames)...".format(TRAIN_FRAME_NUM))
    # Use first N frames for training (avoid data leakage)
    train_video = raw_video[:, :, :TRAIN_FRAME_NUM]
    train_frame_count = train_video.shape[2]

    # Generate averaged frames for blob detection (reduce noise)
    if AVG_FRAME_NUM == 1:
        avg_frames = train_video  # No averaging (use original frames)
    else:
        avg_frame_count = train_frame_count - AVG_FRAME_NUM + 1
        avg_frames = np.zeros((frame_h, frame_w, avg_frame_count), dtype=np.float32)
        for i in tqdm(range(avg_frame_count), desc="Generating averaged frames"):
            # Average consecutive N frames
            avg_frames[:, :, i] = np.mean(train_video[:, :, i:i + AVG_FRAME_NUM], axis=2)
        # Update training video to match avg_frames count (for input-label alignment)
        train_video = train_video[:, :, :avg_frame_count]

    print(f"✅ Training data prepared: {train_video.shape[2]} input frames, {avg_frames.shape[2]} averaged frames")

    # -------------------------- 5. Visualize Blob Detection --------------------------
    print("\n👁️  Visualizing blob detection results (first frame)...")
    # Use first averaged frame for visualization
    sample_avg_frame = avg_frames[:, :, 0]
    sample_raw_frame = train_video[:, :, 0]
    blobs, _, _ = detect_blobs(sample_avg_frame)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Raw frame
    axes[0].imshow(sample_raw_frame, cmap="gray")
    axes[0].set_title("Raw Noisy Frame")
    axes[0].axis("off")
    # Averaged frame (denoised)
    axes[1].imshow(sample_avg_frame, cmap="gray")
    axes[1].set_title(f"Averaged Frame (N={AVG_FRAME_NUM})")
    axes[1].axis("off")
    # Blob detection result
    axes[2].imshow(sample_avg_frame, cmap="gray")
    if blobs.shape[0] > 0:
        axes[2].scatter(blobs[:, 1], blobs[:, 0], c="red", s=20, alpha=0.8, label=f"Blobs ({blobs.shape[0]})")
        axes[2].legend()
    axes[2].set_title(f"Blob Detection (Threshold={DETECT_THRESHOLD})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # Confirm with user before proceeding
    user_input = input("\nAre blob detection results acceptable? Continue training? (y/n): ").strip().lower()
    if user_input != "y":
        print("❌ Training aborted by user.")
        return

    # -------------------------- 6. Generate Training Labels --------------------------
    print("\n🏷️  Generating training labels (blob regions as clean signals)...")
    label_video = np.zeros_like(train_video, dtype=np.float32)
    for i in tqdm(range(avg_frames.shape[2]), desc="Processing each frame"):
        avg_frame = avg_frames[:, :, i]
        raw_frame = train_video[:, :, i]
        blobs, _, _ = detect_blobs(avg_frame)

        # Initialize label: background = min value of raw frame (simulate noise suppression)
        label_frame = np.ones_like(raw_frame, dtype=np.float32) * raw_frame.min()

        # Fill blob regions with original raw frame values (preserve true signals)
        for (r, c) in blobs:
            # Define square region around blob (size: 2*RADIUS + 1)
            r_start = max(0, r - BLOB_BLOCK_RADIUS)
            r_end = min(frame_h, r + BLOB_BLOCK_RADIUS + 1)
            c_start = max(0, c - BLOB_BLOCK_RADIUS)
            c_end = min(frame_w, c + BLOB_BLOCK_RADIUS + 1)
            # Copy raw frame values to label
            label_frame[r_start:r_end, c_start:c_end] = raw_frame[r_start:r_end, c_start:c_end]

        label_video[:, :, i] = label_frame

    print(f"✅ Training labels generated: {label_video.shape[2]} frames")

    # -------------------------- 7. Normalize Training Data --------------------------
    print("\n📏 Normalizing training data (mode={})...".format(NORMALIZE_MODE))
    # Normalize input (noisy) and target (clean) videos
    norm_train_video, _ = normalize_video(train_video)
    norm_label_video, _ = normalize_video(label_video)  # Use same normalization as input

    # -------------------------- 8. Save Training Data as .npy --------------------------
    print("\n💾 Saving training data to .npy files...")
    for i in tqdm(range(norm_train_video.shape[2]), desc="Saving files"):
        # Save input (noisy) frame
        input_save_path = input_dir / f"{i:04d}.npy"
        np.save(input_save_path, norm_train_video[:, :, i])
        # Save target (clean) frame
        target_save_path = target_dir / f"{i:04d}.npy"
        np.save(target_save_path, norm_label_video[:, :, i])

    print(
        f"✅ Training data saved: {len(list(input_dir.glob('*.npy')))} input files, {len(list(target_dir.glob('*.npy')))} target files")

    # -------------------------- 9. Initialize Model & Training Components --------------------------
    print("\n🚀 Initializing model and training components...")
    # Load dataset
    train_dataset = PatchDataset(input_dir=str(input_dir), target_dir=str(target_dir))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows (avoids multi-threading errors)
        pin_memory=True if device.type == "cuda" else False  # Speed up GPU data transfer
    )

    # Initialize model, optimizer, loss function
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()  # MSE loss for regression (image restoration)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized: UNet (Total params: {total_params}, Trainable params: {trainable_params})")
    print(f"✅ Training config: Batch size={BATCH_SIZE}, Epochs={TRAIN_EPOCHS}, LR={LEARNING_RATE}")

    # -------------------------- 10. Model Training --------------------------
    print("\n🔥 Starting model training...")
    model.train()  # Set model to training mode (enables gradient computation)
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        # Iterate over training batches
        for batch_idx, (input_batch, target_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")):
            # Move data to device (GPU/CPU)
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            # Forward pass: model prediction
            optimizer.zero_grad()  # Clear gradients from previous iteration
            pred_batch = model(input_batch)

            # Calculate loss
            loss = criterion(pred_batch, target_batch)

            # Backward pass: compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item() * input_batch.size(0)  # Multiply by batch size to account for partial batches

        # Calculate average loss per sample
        avg_epoch_loss = epoch_loss / len(train_dataset)
        print(f"📊 Epoch {epoch + 1}/{TRAIN_EPOCHS} | Average Loss: {avg_epoch_loss:.6f}")

    print("✅ Model training completed!")

    # -------------------------- 11. Full Video Inference --------------------------
    print("\n🔍 Starting full video inference (all {} frames)...".format(total_frames))
    # Normalize full raw video (use same mode as training)
    norm_full_video, norm_params = normalize_video(raw_video)

    # Prepare inference results storage
    pred_frames = []
    model.eval()  # Set model to evaluation mode (disables training-specific layers)

    # Disable gradient computation for inference (saves memory and speed)
    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Inferring each frame"):
            # Get normalized frame and reshape to model input format (1 x 1 x H x W)
            norm_frame = norm_full_video[:, :, frame_idx]
            input_tensor = torch.from_numpy(norm_frame).float().unsqueeze(0).unsqueeze(0).to(device)

            # Model prediction
            pred_tensor = model(input_tensor)

            # Convert tensor to numpy array (remove batch/channel dimensions)
            norm_pred_frame = pred_tensor.cpu().squeeze(0).squeeze(0).numpy()

            # Denormalize to original pixel range
            if NORMALIZE_MODE == 1:
                # Per-frame normalization: use params for current frame
                frame_norm_params = norm_params[frame_idx]
            else:
                # Global/clipped normalization: use single set of params
                frame_norm_params = norm_params
            pred_frame = denormalize_frame(norm_pred_frame, frame_norm_params)

            # Save to results list
            pred_frames.append(pred_frame)

    # Convert list to 3D array (Frames x H x W) for TIFF saving
    pred_video = np.stack(pred_frames, axis=0)
    print(f"✅ Full video inference completed: {pred_video.shape[0]} frames")

    # -------------------------- 12. Save Inference Results --------------------------
    print("\n💾 Saving inference results...")
    # Create output filename with key parameters (for easy identification)
    output_filename = f"{Path(RAW_VIDEO_PATH).stem}-th{DETECT_THRESHOLD}-avg{AVG_FRAME_NUM}-C{NORMALIZE_MODE}.tif"
    output_path = Path(RAW_VIDEO_PATH).parent / output_filename

    # Save as multi-frame TIFF (shape: Frames x H x W)
    tifffile.imwrite(
        output_path,
        pred_video.astype(np.float32),  # Use float32 to preserve precision
        photometric='minisblack'  # Grayscale format
    )

    # -------------------------- 13. Final Summary --------------------------
    print("\n" + "=" * 80)
    print("🎉 Full workflow completed successfully!")
    print(f"📌 Key Results:")
    print(f"  - Input Video: {Path(RAW_VIDEO_PATH).name} ({frame_h}x{frame_w}, {total_frames} frames)")
    print(f"  - Output Video: {output_filename}")
    print(f"  - Saved to: {output_path.parent}")
    print("\n📌 Training Configuration:")
    print(f"  - Frames Used for Training: {TRAIN_FRAME_NUM}")
    print(f"  - Averaging Frames (Blob Detection): {AVG_FRAME_NUM}")
    print(f"  - Normalization Mode: {NORMALIZE_MODE} (Scale: {NORMALIZE_SCALE})")
    print(f"  - Epochs: {TRAIN_EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    print(f"  - Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print("\n📌 Detection Parameters:")
    print(f"  - Blob Threshold: {DETECT_THRESHOLD}, Window Size: {DETECT_WINDOW_SIZE}x{DETECT_WINDOW_SIZE}")
    print(f"  - Blob Region Radius: {BLOB_BLOCK_RADIUS}")
    print("=" * 80)


if __name__ == "__main__":
    main()