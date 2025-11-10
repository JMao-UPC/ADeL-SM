# ADeL-SM
![ADeL-SM](assets/ADeL-SM.png)
Adaptive deep learning framework for high-fidelity single-molecule imaging

## 📋 Table of content
---
- [Overview](#overview)

### 🚀 Quick start ADeL-SM
  - 💻 [Environment](#environment)
  - 📦 [Install dependencies](#install-dependencies)
  - 🛠️ [Work for your own data](#work-for-your-own-data)

### 📚 Other information
  - ✉️ [Email](#email)

## 📚 Overview
---
ADeL-SM is an adaptive deep learning framework designed to enhance single-molecule fluorescence imaging. By combining advanced denoising strategies with precise localization algorithms, it significantly improves signal-to-noise ratio and localization accuracy, enabling reliable quantitative analysis of molecular dynamics. The method has been validated on simulated and experimental data, demonstrating its ability to preserve single-molecule signals even under high-density and low-SNR conditions.
![TOC]

## ⏳ Quick start DeepSeMi
---
This tutorial will show how ADeL-SM enables high-fidelity single-molecule imaging.

### 💡 Environment
The ADeL-SM framework runs on Python and supports both CPU and GPU acceleration (recommended for faster training).

#### Minimum Requirements
- **Python**: 3.9 ~ 3.11 (64-bit)  
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+  
- **Disk Space**: ≥ 5GB (for code, dependencies, and data)  

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
```

### 💡 Work for your own data
   **Configure parameters**  
   All input paths and runtime parameters are set in `config/config.py`.  
   ```python
   # Key configurations in config/config.py
   INPUT_DIR = "./demo_data"          # Input data folder
   BATCH_SIZE = 8                     # Batch size (adjust based on memory)
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-select device (GPU/CPU)
   ```
   Run the main.py to use the network based on your own data. 
   ```bash
   $ python main.py
```
### ✉️ Contact
For technical issues, questions about the framework, or collaboration inquiries, please reach out via email:  **BZ23030032@s.upc.edu.cn**  






