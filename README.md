# ADeL-SM
Adaptive deep learning framework for high-fidelity single-molecule imaging

## 📋 Table of content
--
- [Overview](#overview)

### 🚀 Quick start ADeL-SM
  - 💻 [Environment](#environment)
  - 📦 [Install dependencies](#install-dependencies)
  - 📥 [Download the demo code and data](#download-the-demo-code-and-data)
  - ▶️ [Run the trained model](#run-the-trained-model)
  - 🛠️ [Work for your own data](#work-for-your-own-data)

### 📚 Other information
  - 📊 [Results](#results)
  - 📄 [Citation](#citation)
  - ✉️ [Email](#email)

## 📚 Overview
--
ADeL-SM is an adaptive deep learning framework designed to enhance single-molecule fluorescence imaging. By combining advanced denoising strategies with precise localization algorithms, it significantly improves signal-to-noise ratio and localization accuracy, enabling reliable quantitative analysis of molecular dynamics. The method has been validated on simulated and experimental data, demonstrating its ability to preserve single-molecule signals even under high-density and low-SNR conditions.


## ⏳ Quick start DeepSeMi
--
This tutorial will show how ADeL-SM enables high-fidelity single-molecule imaging.

### 💡 Environment
The ADeL-SM framework runs on Python and supports both CPU and GPU acceleration (recommended for faster training).

#### Minimum Requirements
- **Python**: 3.9 ~ 3.11 (64-bit)  
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+  
- **Disk Space**: ≥ 5GB (for code, dependencies, and demo data)  

#### GPU Acceleration (Highly Recommended)
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.0  
- **CUDA**: 11.7 ~ 12.1  
- **Driver**: NVIDIA GPU Driver ≥ 450.80.02 (for CUDA 11.x)  

*✨ Pro Tip: GPU training is 10-20x faster than CPU—perfect for large microscopy datasets!*

### 💡 Install dependencies
Create a virtual environment and install dependencies. Ensure PyTorch, TorchVision, and CUDA versions match.

```bash
$ conda create -n adel-sm-env python=3.9
$ conda activate adel-sm-env
$ pip install --upgrade pip
$ pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118  # Use appropriate CUDA version
$ pip install scipy==1.11.4 tifffile scikit-image==0.22.0 h5py==3.10.0 opencv-python==4.9.0.80

### 💡 Run the trained model 
After preparing demo data (in `demo_data` folder) and pre-trained model (in `pretrained_weights` folder), run the demo script directly:
```bash
# For ADeL-SM denoising/inference
python demo_inference.py --data_path ./demo_data --weight_path ./pretrained_weights/adel-sm_model.pth







