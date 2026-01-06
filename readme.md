# Jetson PyTorch & OpenCV Installer

<div align="center">

**Automated Installation Scripts for PyTorch and OpenCV on NVIDIA Jetson Devices**

[![JetPack](https://img.shields.io/badge/JetPack-5.1.1%20%7C%205.1.2%20%7C%206.0%20%7C%206.1%20%7C%206.2-blue)](https://developer.nvidia.com/embedded/jetpack)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Package Name:** `jetson-pytorch-opencv-installer`

</div>

---

## 📖 Overview

**Jetson PyTorch & OpenCV Installer** is a comprehensive package that provides automated installation scripts for essential AI/ML libraries on NVIDIA Jetson devices. This package simplifies the complex process of installing PyTorch (with CUDA support) and OpenCV (with CUDA/cuDNN/GStreamer) on Jetson platforms, eliminating the need for manual compilation and dependency management.

### Key Benefits

- ✅ **Automated Detection**: Automatically detects JetPack version and suggests correct configuration
- ✅ **Pre-built Wheels**: Uses NVIDIA's optimized pre-built PyTorch wheels when available
- ✅ **Memory Management**: Intelligent memory checking and optional swap space creation
- ✅ **Error Prevention**: Built-in safeguards to prevent common installation issues
- ✅ **CUDA Integration**: Properly configures libraries to leverage Jetson's GPU capabilities

---

## 👤 Author

**Vahid Behtaji**  
📧 Email: [vahidbehtaji2013@gmail.com](mailto:vahidbehtaji2013@gmail.com)

## 🛠 Available Scripts

| Script                 | Purpose                                 | JetPack Support             | Installation Method             | Estimated Time | Memory Required |
| ---------------------- | --------------------------------------- | --------------------------- | ------------------------------- | -------------- | --------------- |
| `install_pytorch.sh`   | PyTorch with CUDA support + Torchvision | 5.1.1, 5.1.2, 6.0, 6.1, 6.2 | Pre-built wheels + Source build | 25-50 min      | 2-4 GB RAM      |
| `build_opencv.sh`      | OpenCV with CUDA/cuDNN/GStreamer        | All versions                | Source compilation              | 2-4 hours      | 4-8 GB RAM      |
| `test_pytorch.sh`      | Verify PyTorch & Torchvision installation | All versions                | Test script (no installation)   | < 1 min        | Minimal         |

### Script Comparison

| Feature | `install_pytorch.sh` | `build_opencv.sh` |
|---------|---------------------|-------------------|
| **Complexity** | ⭐ Low | ⭐⭐⭐ High |
| **User Interaction** | ✅ Interactive | ⚠️ Minimal |
| **Automatic Detection** | ✅ Yes | ❌ No |
| **Memory Management** | ✅ Yes | ⚠️ Manual |
| **CUDA Support** | ✅ Automatic | ✅ Automatic |
| **Recommended For** | All users | Advanced users |

## 🚀 Quick Start

### Step-by-Step Installation Guide

| Step | Action | Command | Expected Time | Notes |
|------|--------|---------|---------------|-------|
| **1** | **Navigate to Package Directory** | `cd /path/to/jetson-pytorch-opencv-installer` | - | Ensure you're in the correct directory |
| **2** | **Make Scripts Executable** | `chmod +x *.sh` | < 1 min | Required for script execution (includes test_pytorch.sh) |
| **3** | **Verify System Requirements** | `apt list --installed \| grep nvidia-jetpack` | < 1 min | Check JetPack version |
| **4** | **Run PyTorch Installer** | `./install_pytorch.sh` | 5-10 min | Interactive installation |
| **5** | **Follow Interactive Prompts** | See prompts below | - | Enter JetPack code, confirm wheel URL |
| **6** | **Install Torchvision (Optional)** | Answer `y` when prompted | 20-40 min | Required for ultralytics |
| **7** | **Verify Installation** | `./test_pytorch.sh` | < 1 min | Comprehensive test script |
| **8** | **Build OpenCV (Optional)** | `./build_opencv.sh` | 2-4 hours | Only if needed for your project |

### Detailed Installation Steps

#### Step 1: Preparation

```bash
# Navigate to the package directory
cd /path/to/jetson-pytorch-opencv-installer

# Make scripts executable
chmod +x install_pytorch.sh build_opencv.sh

# Verify your JetPack version (optional, script will detect automatically)
apt list --installed | grep nvidia-jetpack
```

#### Step 2: Install PyTorch

```bash
# Run the PyTorch installer
./install_pytorch.sh
```

**During Installation, You Will Be Prompted:**

| Prompt | Your Action | Example |
|--------|-------------|---------|
| **JetPack Detection** | Script automatically detects | `Detected JetPack version: 6.2.1` |
| **Suggested Code** | Review the suggestion | `Suggested JetPack code: 62` |
| **Enter JetPack Code** | Enter code or press Enter | `62` (or press Enter to use suggested) |
| **Wheel URL** | Press Enter for default | Press Enter |
| **Install Torchvision?** | Type `y` for yes, `n` for no | `y` |
| **Create Swap?** | Type `y` if memory is low | `y` (if prompted) |

#### Step 3: Verification

```bash
# Test PyTorch installation
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Test Torchvision (if installed)
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

# Or use the test script (recommended)
./test_pytorch.sh
```

---

## 📋 Script Documentation

### PyTorch Installer (`install_pytorch.sh`)

Interactive installer for PyTorch with CUDA support on Jetson devices.

#### Features
- **Automatic JetPack Detection**: Automatically detects your JetPack version and suggests the appropriate code
- **Smart Code Suggestion**: Displays detected JetPack version and suggests the correct code (e.g., 6.2.1 → 62)
- **Smart Wheel Selection**: Provides tested default wheel URLs with override options
- **cuSPARSELt Integration**: Automatically installs cuSPARSELt for JP ≥ 6.0 compatibility
- **CUDA Validation**: Verifies PyTorch installation and CUDA availability
- **Torchvision Support**: Optional torchvision installation with memory-optimized build (MAX_JOBS=1)
- **Memory Management**: Checks available memory and offers to create swap space if needed
- **OOM Prevention**: Uses single-threaded compilation (MAX_JOBS=1) to prevent out-of-memory errors

#### Usage
```bash
./install_pytorch.sh
```

#### Interactive Prompts Flow

The script follows this interactive flow:

| Step | Prompt | User Input | Default/Action |
|------|--------|------------|----------------|
| **1** | `Detecting JetPack version...` | Automatic | Script detects version |
| **2** | `Detected JetPack version: X.X.X` | Display only | Shows detected version |
| **3** | `Suggested JetPack code: XX` | Display only | Shows suggested code |
| **4** | `Enter JetPack code [suggested: XX]:` | Enter code or press Enter | Uses suggested if Enter |
| **5** | `Wheel URL [default: ...]:` | Press Enter or custom URL | Uses default if Enter |
| **6** | `Install torchvision? [y/N]:` | `y` or `n` | `n` (skip) |
| **7** | `Create 4GB swap file? [y/N]:` | `y` or `n` | Only if memory < 2GB |

**Example Session:**

```bash
=== PyTorch Installer for Jetson ===
Detecting JetPack version...
Detected JetPack version: 6.2.1
Suggested JetPack code: 62

Enter JetPack code (examples: 511, 512, 60, 61, 62) [suggested: 62]: 
# User presses Enter

Wheel URL [press Enter for default: https://...]: 
# User presses Enter

Install torchvision? (required for ultralytics) [y/N]: y
# User types 'y'

Available memory: 1500 MB
WARNING: Low available memory (1500 MB). Consider adding swap space.
Create 4GB swap file to help with compilation? [y/N]: y
# User types 'y'

Building torchvision from source (this may take 20-40 minutes)...
# Build proceeds automatically
```

#### Supported Configurations
- **JetPack 5.1.1/5.1.2**: PyTorch wheels without cuSPARSELt requirement
- **JetPack 6.0+**: Latest PyTorch wheels with automatic cuSPARSELt installation

#### Testing PyTorch Installation
After installation, verify PyTorch is working correctly:

```python
# Test basic PyTorch functionality
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

    # Test tensor operations on GPU
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = torch.mm(x, y)
    print('GPU tensor operations: SUCCESS')
"
```

#### Troubleshooting

**JetPack Detection Issues:**
- If automatic detection fails, manually check your JetPack version:
  ```bash
  apt list --installed | grep nvidia-jetpack
  # Or
  cat /etc/nv_tegra_release
  ```

**cuSPARSELt Issues:**
- Ensure you're using the script on JP 6.0+ for automatic cuSPARSELt installation
- Script automatically detects CUDA version (defaults to 12.6 for JetPack 6.x)

**Memory Issues:**
- Script checks available memory before building torchvision
- If memory is low (< 2GB), script offers to create a 4GB swap file
- Torchvision build uses MAX_JOBS=1 (single-threaded) to prevent OOM errors
- Close other applications during installation to free up RAM
- If build still fails, manually create swap space:
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

**Wheel Compatibility:**
- Verify your JetPack version matches the selected wheel
- Script provides tested default wheels for each JetPack version

**Torchvision Build Failures:**
- If torchvision build fails with "Killed" error, it's likely an OOM issue
- Script automatically uses MAX_JOBS=1, but you can manually set it:
  ```bash
  export MAX_JOBS=1
  python3 setup.py install
  ```
- Ensure swap space is available if system RAM is limited

---

### PyTorch Test Script (`test_pytorch.sh`)

Comprehensive test script to verify PyTorch and Torchvision installation.

#### Features
- **PyTorch Verification**: Checks PyTorch version, installation path, and CUDA availability
- **CUDA Testing**: Verifies CUDA version, GPU detection, and GPU tensor operations
- **Torchvision Check**: Tests Torchvision installation and basic functionality
- **Detailed Output**: Provides comprehensive information about GPU properties
- **Error Reporting**: Clear error messages with troubleshooting suggestions

#### Usage
```bash
# Make script executable (if not already)
chmod +x test_pytorch.sh

# Run the test
./test_pytorch.sh
```

#### Test Output
The script provides detailed information including:
- PyTorch version and installation path
- CUDA availability and version
- GPU name, memory, and compute capability
- GPU tensor operation test
- Torchvision version and functionality

#### Example Output
```
==========================================
PyTorch & Torchvision Installation Test
==========================================

Testing PyTorch installation...
-----------------------------------
✅ PyTorch version: 2.5.0a0+872d972e41.nv24.8
✅ PyTorch path: /home/jetson/web_gui/venv/lib/python3.10/site-packages/torch/__init__.py
✅ CUDA available: True
✅ CUDA version: 12.6
✅ CUDA cuDNN version: 8902
✅ GPU count: 1
✅ GPU name: Orin (nvgpu)
✅ GPU memory: 15.75 GB
✅ GPU compute capability: 8.7
✅ GPU tensor operations: SUCCESS

Testing Torchvision installation...
-----------------------------------
✅ TorchVision version: 0.20.0a0
✅ TorchVision path: /home/jetson/web_gui/venv/lib/python3.10/site-packages/torchvision/__init__.py
✅ TorchVision transforms: SUCCESS

==========================================
Test Summary
==========================================
✅ PyTorch: INSTALLED AND WORKING
✅ TorchVision: INSTALLED AND WORKING

🎉 PyTorch installation test completed successfully!
```

---

### OpenCV Builder (`build_opencv.sh`)

Comprehensive OpenCV compilation script with CUDA, cuDNN, and GStreamer support.

#### Features
- **Full CUDA Integration**: Builds OpenCV with CUDA and cuDNN acceleration
- **GStreamer Support**: Enables hardware-accelerated video processing
- **Python Bindings**: Includes Python 3 bindings for cv2
- **System-wide Installation**: Installs OpenCV globally for all users
- **Optimized Build**: Configured for Jetson hardware optimization

#### Usage
```bash
# Start the build process (this will take 2-4 hours)
./build_opencv.sh

# Monitor progress
tail -f /tmp/opencv_build.log  # if logging is implemented
```

#### Build Configuration
The script configures OpenCV with:
- CUDA acceleration for image processing
- cuDNN support for deep learning inference
- GStreamer integration for video I/O
- Python 3 bindings
- Optimized compiler flags for ARM64

#### Testing OpenCV Installation
Verify your OpenCV installation after the build completes:

```python
# Test basic OpenCV functionality
python3 -c "
import cv2
import numpy as np

print(f'OpenCV version: {cv2.__version__}')

# Test CUDA support
print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')

# Test basic image operations
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (15, 15), 0)
print('Basic image operations: SUCCESS')

# Test CUDA image operations (if CUDA is available)
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
    result = gpu_gray.download()
    print('GPU image operations: SUCCESS')
"
```

#### Performance Testing
Test OpenCV performance with CUDA acceleration:

```python
# Performance comparison script
python3 -c "
import cv2
import numpy as np
import time

# Create test image
img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

# CPU processing
start_time = time.time()
for _ in range(100):
    gray_cpu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_cpu = cv2.GaussianBlur(gray_cpu, (15, 15), 0)
cpu_time = time.time() - start_time
print(f'CPU processing time: {cpu_time:.3f} seconds')

# GPU processing (if available)
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    start_time = time.time()
    for _ in range(100):
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        gpu_blur = cv2.cuda.bilateralFilter(gpu_gray, -1, 50, 50)
    cv2.cuda.deviceSynchronize()
    gpu_time = time.time() - start_time
    print(f'GPU processing time: {gpu_time:.3f} seconds')
    print(f'Speedup: {cpu_time/gpu_time:.2f}x')
"
```

---

## 🔧 System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Jetson Model** | Nano, Xavier NX | AGX Xavier, Orin | All models supported |
| **RAM** | 4 GB | 8 GB+ | 8GB+ for OpenCV compilation |
| **Storage** | 16 GB free | 32 GB+ free | Includes temporary build files |
| **Swap Space** | Optional | 4 GB+ | Recommended for torchvision build |

### Software Requirements

| Software | Version | Installation Method |
|----------|---------|-------------------|
| **Ubuntu** | 18.04 / 20.04 / 22.04 | Pre-installed with JetPack |
| **JetPack** | 5.1.1, 5.1.2, 6.0, 6.1, 6.2 | Pre-installed |
| **CUDA** | 11.4+ / 12.0+ | Included with JetPack |
| **Python** | 3.6+ | Pre-installed |
| **pip** | Latest | `python3 -m pip install --upgrade pip` |

### Pre-installation Setup Checklist

| Task | Command | Purpose | Required |
|------|---------|---------|----------|
| **Update System** | `sudo apt update && sudo apt upgrade -y` | Get latest packages | ✅ Yes |
| **Install Build Tools** | `sudo apt install -y build-essential cmake git wget curl` | Compilation dependencies | ✅ Yes |
| **Verify CUDA** | `nvcc --version` | Confirm CUDA installation | ✅ Yes |
| **Check JetPack** | `apt list --installed \| grep nvidia-jetpack` | Verify JetPack version | ⚠️ Recommended |
| **Check Python** | `python3 --version` | Verify Python 3.6+ | ⚠️ Recommended |
| **Check Disk Space** | `df -h` | Ensure sufficient space | ⚠️ Recommended |

**Quick Setup Script:**

```bash
#!/bin/bash
# Pre-installation setup script

echo "=== Pre-installation Setup ==="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential tools
echo "Installing build tools..."
sudo apt install -y build-essential cmake git wget curl python3-pip

# Verify CUDA
echo "Verifying CUDA installation..."
nvcc --version || echo "WARNING: CUDA not found"

# Check JetPack
echo "Checking JetPack version..."
apt list --installed | grep nvidia-jetpack || echo "WARNING: JetPack not detected"

# Check disk space
echo "Checking disk space..."
df -h / | tail -1

echo "=== Setup Complete ==="
```

## 📊 Performance Benchmarks

### Installation Times

| Component | Jetson Nano | Jetson Xavier NX | Jetson Orin | Notes |
|-----------|-------------|-------------------|-------------|-------|
| **PyTorch** | 8-12 min | 5-8 min | 5-10 min | Pre-built wheel |
| **Torchvision** | 40-60 min | 25-35 min | 20-30 min | Source build (MAX_JOBS=1) |
| **OpenCV** | 4-6 hours | 2-3 hours | 2-4 hours | Source compilation |

### Resource Usage

| Operation | CPU Usage | RAM Usage | Disk Usage | Network |
|-----------|-----------|-----------|------------|---------|
| **PyTorch Install** | Low | < 500 MB | ~800 MB | High (download) |
| **Torchvision Build** | 100% (1 core) | 2-4 GB | ~2 GB | Low |
| **OpenCV Build** | 100% (all cores) | 4-8 GB | ~5 GB | Medium |

### Optimization Tips

| Tip | Impact | Implementation |
|-----|--------|----------------|
| **Use Swap Space** | ⭐⭐⭐ High | Script offers automatic creation |
| **Close Applications** | ⭐⭐ Medium | Free up RAM before building |
| **MAX_JOBS=1** | ⭐⭐⭐ High | Already implemented for torchvision |
| **SSD Storage** | ⭐⭐ Medium | Faster compilation times |
| **Stable Internet** | ⭐⭐⭐ High | Required for downloads |

## 🔍 Verification & Testing

### Post-Installation Verification Checklist

| Test | Command | Expected Result |
|------|---------|-----------------|
| **PyTorch Import** | `python3 -c "import torch"` | No errors |
| **CUDA Available** | `python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| **CUDA Version** | `python3 -c "import torch; print(torch.version.cuda)"` | Version number (e.g., `12.6`) |
| **GPU Name** | `python3 -c "import torch; print(torch.cuda.get_device_name(0))"` | GPU model name |
| **Torchvision** | `python3 -c "import torchvision; print(torchvision.__version__)"` | Version number |
| **GPU Tensor Ops** | See test script below | `SUCCESS` |

**Complete Test Script:**

```python
#!/usr/bin/env python3
"""Comprehensive PyTorch installation test"""

import sys

def test_pytorch():
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: True")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
            
            # Test GPU operations
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations: SUCCESS")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_torchvision():
    try:
        import torchvision
        print(f"✅ TorchVision version: {torchvision.__version__}")
        return True
    except ImportError:
        print("⚠️  TorchVision not installed")
        return False
    except Exception as e:
        print(f"❌ TorchVision error: {e}")
        return False

if __name__ == "__main__":
    print("=== PyTorch Installation Test ===\n")
    pytorch_ok = test_pytorch()
    print()
    torchvision_ok = test_torchvision()
    print()
    
    if pytorch_ok:
        print("✅ PyTorch installation: SUCCESS")
    else:
        print("❌ PyTorch installation: FAILED")
        sys.exit(1)
```

## 🤝 Contributing

We welcome contributions! Please consider:

| Contribution Type | Description | Priority |
|-------------------|-------------|----------|
| **Bug Reports** | Report issues and errors | High |
| **Feature Requests** | Suggest new features | Medium |
| **Code Contributions** | Submit pull requests | High |
| **Documentation** | Improve documentation | Medium |
| **Testing** | Test on different JetPack versions | High |

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA for providing Jetson platform and documentation
- PyTorch team for excellent ML framework
- OpenCV contributors for computer vision library

## ⚠️ Important Notes

| Note | Details |
|------|---------|
| **System Backup** | Always create a system backup before running installation scripts |
| **JetPack Compatibility** | Script automatically detects version; ensure it's supported (5.1.1, 5.1.2, 6.0, 6.1, 6.2) |
| **Internet Connection** | Stable connection required for downloading packages and wheels |
| **Memory Requirements** | See Performance Benchmarks section for detailed requirements |
| **Installation Time** | See Performance Benchmarks section for estimated times |

## 📞 Support & Contact

### Author Information

| Information | Details |
|-------------|---------|
| **Name** | Vahid Behtaji |
| **Email** | [vahidbehtaji2013@gmail.com](mailto:vahidbehtaji2013@gmail.com) |
| **Contact For** | Questions, issues, feature requests, contributions |

### Getting Help

| Issue Type | Action | Response Time |
|------------|--------|---------------|
| **Installation Problems** | Check Troubleshooting section first | - |
| **Bug Reports** | Email with error logs | 1-3 days |
| **Feature Requests** | Email with detailed description | 3-7 days |
| **General Questions** | Email with specific question | 1-2 days |

---

<div align="center">

**Made with ❤️ for the Jetson Community**

⭐ Star this repository if you find it helpful!

</div>
