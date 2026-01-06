#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO}"; exit 1' ERR

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: This script targets NVIDIA Jetson (aarch64)."
  exit 1
fi

echo "=== PyTorch Installer for Jetson ==="

# Detect and display JetPack version
SUGGESTED_CODE=""
echo "Detecting JetPack version..."
if command -v dpkg >/dev/null 2>&1; then
  JETPACK_VERSION=$(dpkg -l | grep -i "nvidia-jetpack" | grep -E "^ii" | head -1 | awk '{print $3}' | cut -d'+' -f1 | cut -d'.' -f1,2)
  if [[ -n "${JETPACK_VERSION}" ]]; then
    echo "Detected JetPack version: ${JETPACK_VERSION}"
    # Map JetPack version to code
    case "${JETPACK_VERSION}" in
      5.1*)
        SUGGESTED_CODE="511"
        ;;
      5.1.1)
        SUGGESTED_CODE="511"
        ;;
      5.1.2)
        SUGGESTED_CODE="512"
        ;;
      6.0*)
        SUGGESTED_CODE="60"
        ;;
      6.1*)
        SUGGESTED_CODE="61"
        ;;
      6.2*)
        SUGGESTED_CODE="62"
        ;;
      *)
        SUGGESTED_CODE=""
        ;;
    esac
    if [[ -n "${SUGGESTED_CODE}" ]]; then
      echo "Suggested JetPack code: ${SUGGESTED_CODE}"
    fi
  else
    # Try alternative method: check /etc/nv_tegra_release
    if [[ -f /etc/nv_tegra_release ]]; then
      L4T_VERSION=$(cat /etc/nv_tegra_release | head -1 | cut -d',' -f2 | sed 's/^[[:space:]]*//' | cut -d' ' -f1)
      echo "Detected L4T version: ${L4T_VERSION}"
      # L4T R36.x corresponds to JetPack 6.x
      if [[ "${L4T_VERSION}" =~ ^36\. ]]; then
        echo "This appears to be JetPack 6.x"
        echo "Suggested JetPack codes: 60, 61, or 62"
      elif [[ "${L4T_VERSION}" =~ ^35\. ]]; then
        echo "This appears to be JetPack 5.x"
        echo "Suggested JetPack codes: 511 or 512"
      fi
    else
      echo "Could not detect JetPack version automatically."
    fi
  fi
else
  echo "Could not detect JetPack version (dpkg not available)."
fi

echo ""
read -rp "Enter JetPack code (examples: 511, 512, 60, 61, 62)${SUGGESTED_CODE:+ [suggested: ${SUGGESTED_CODE}]}: " JP
JP="${JP//[[:space:]]/}"

# Detect python tag (cpXX) automatically
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
CPTAG="cp${PYVER}-cp${PYVER}"
echo "Detected Python tag: ${CPTAG}"

# Map JP -> default wheel URL (user can override later)
WHEEL_URL=""
case "${JP}" in
  511)
    WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"  # cp38
    ;;
  512)
    WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl" # cp38
    ;;
  60)
    # Several 2.4.0 wheels exist; default to the newer one
    WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl"
    ;;
  61)
    # Known working (your link)
    WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
    ;;
  62)
    echo "NOTICE: NVIDIA hasn't published a v62 wheel directory yet."
    echo "Provide a direct wheel URL for JP62 (or use a Jetson container)."
    ;;
  *)
    echo "ERROR: Unknown JP code '${JP}'. Valid examples: 511, 512, 60, 61, 62"
    exit 1
    ;;
esac

read -rp "Wheel URL [press Enter for default: ${WHEEL_URL:-<none>}]: " USER_URL || true
USER_URL="${USER_URL//[[:space:]]/}"
# If user entered just "y" or "yes", treat as default
if [[ "${USER_URL}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
  USER_URL=""
fi
TORCH_URL="${USER_URL:-$WHEEL_URL}"

if [[ -z "${TORCH_URL}" ]]; then
  echo "ERROR: No wheel URL provided."
  echo "Tip: Browse https://developer.download.nvidia.com/compute/redist/jp/ then pick your v<JP>/pytorch/ wheel."
  exit 1
fi

echo "Selected: ${TORCH_URL}"
echo "Updating apt deps…"

# Handle package lock issue
if sudo lsof /var/lib/dpkg/lock-frontend 2>/dev/null | grep -q packagekitd; then
  echo "WARNING: packagekitd is holding the lock. Stopping it temporarily..."
  sudo systemctl stop packagekit || true
  sleep 2
fi

sudo apt-get update -y
sudo apt-get install -y python3-pip libopenblas-dev curl

# Install cuSPARSELt first for JP >= 60 (PyTorch >= 24.06 needs it)
if [[ "${JP}" =~ ^6 ]]; then
  echo "Installing cuSPARSELt (required for JP ${JP})…"
  
  # Detect CUDA version first
  DEST_CUDA="/usr/local/cuda"
  if [[ -z "${CUDA_VERSION:-}" ]]; then
    if [[ -f "${DEST_CUDA}/version.txt" ]]; then
      CUDA_VERSION="$(grep -oE '[0-9]+\.[0-9]+' "${DEST_CUDA}/version.txt" | head -n1)"
    elif command -v nvcc >/dev/null 2>&1; then
      CUDA_VERSION="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
    elif ls /usr/local/cuda-*/version.txt 2>/dev/null | head -1 | xargs grep -oE '[0-9]+\.[0-9]+' 2>/dev/null | head -n1 >/dev/null; then
      CUDA_VERSION="$(ls /usr/local/cuda-*/version.txt 2>/dev/null | head -1 | xargs grep -oE '[0-9]+\.[0-9]+' 2>/dev/null | head -n1)"
    else
      # Default to 12.6 for JetPack 6.x (most common)
      echo "WARNING: Could not detect CUDA version, defaulting to 12.6 for JetPack 6.x"
      CUDA_VERSION="12.6"
    fi
  fi
  
  # Ensure CUDA_VERSION is set (final fallback)
  if [[ -z "${CUDA_VERSION:-}" ]]; then
    CUDA_VERSION="12.6"
    echo "Using default CUDA version: ${CUDA_VERSION}"
  else
    echo "Detected CUDA version: ${CUDA_VERSION}"
  fi
  
  # Use bundled installer next to this script if present, else fetch inline
  if [[ -f "./install_cusparselt.sh" ]]; then
    CUDA_VERSION="${CUDA_VERSION}" bash ./install_cusparselt.sh
  else
    TMP="$(mktemp -d)"
    cat > "${TMP}/install_cusparselt.sh" <<CUS
#!/usr/bin/env bash
set -Eeo pipefail
trap 'echo "cuSPARSELt installer error on line \${LINENO}"; exit 1' ERR
DEST_CUDA="\${DEST_CUDA:-/usr/local/cuda}"
CUSPARSELT_VER="\${CUSPARSELT_VER:-0.7.1.0}"
CUDA_VERSION="${CUDA_VERSION}"
ARCH_DIR="linux-aarch64"
BASE_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/\${ARCH_DIR}"
PKG="libcusparse_lt-\${ARCH_DIR}-\${CUSPARSELT_VER}-archive"
TAR="\${PKG}.tar.xz"
TMPD="\$(mktemp -d)"
cd "\$TMPD"
if ! curl -fS --retry 3 -O "\${BASE_URL}/\${TAR}"; then
  FALLBACK_BASE="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa"
  PKG="libcusparse_lt-linux-sbsa-0.5.2.1-archive"
  TAR="\${PKG}.tar.xz"
  curl -fS --retry 3 -O "\${FALLBACK_BASE}/\${TAR}"
fi
tar xf "\${TAR}"
sudo mkdir -p "\${DEST_CUDA}/include" "\${DEST_CUDA}/lib64"
sudo cp -a "\${PKG}/include/." "\${DEST_CUDA}/include/"
sudo cp -a "\${PKG}/lib/."     "\${DEST_CUDA}/lib64/"
sudo ldconfig
ls "\${DEST_CUDA}/lib64"/libcusparseLt.so* >/dev/null 2>&1 || { echo "ERROR: libcusparseLt missing"; exit 1; }
CUS
    bash "${TMP}/install_cusparselt.sh"
  fi
fi

echo "Upgrading pip and (if needed) numpy pin…"
python3 -m pip install --upgrade pip
# NVIDIA docs often require numpy==1.26.1 for these wheels (esp. JP6.0/6.1)
if [[ "${JP}" == "60" || "${JP}" == "61" ]]; then
  python3 -m pip install "numpy"
fi

echo "Installing torch wheel…"
python3 -m pip install --no-cache-dir "${TORCH_URL}"

echo "Verifying PyTorch + CUDA…"
python3 - <<'PY'
import sys
try:
    import torch
    print("torch:", torch.__version__)
    print("built with CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("ERROR importing torch:", e)
    sys.exit(1)
PY

echo "SUCCESS: PyTorch installed."

# Ask if user wants to install torchvision
read -rp "Install torchvision? (required for ultralytics) [y/N]: " INSTALL_TV || true
INSTALL_TV="${INSTALL_TV//[[:space:]]/}"
if [[ "${INSTALL_TV}" =~ ^[Yy]$ ]]; then
  echo "Installing torchvision build dependencies…"
  sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev || {
    echo "WARNING: Some dependencies may have failed. Continuing anyway..."
  }
  
  # Check available memory and suggest swap if needed
  AVAIL_MEM=$(free -m | awk '/^Mem:/{print $7}')
  echo "Available memory: ${AVAIL_MEM} MB"
  if [[ ${AVAIL_MEM} -lt 2000 ]]; then
    echo "WARNING: Low available memory (${AVAIL_MEM} MB). Consider adding swap space."
    read -rp "Create 4GB swap file to help with compilation? [y/N]: " CREATE_SWAP || true
    if [[ "${CREATE_SWAP}" =~ ^[Yy]$ ]]; then
      echo "Creating swap file (this may take a few minutes)…"
      if [[ ! -f /swapfile ]]; then
        sudo fallocate -l 4G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo "Swap file created and activated."
      else
        echo "Swap file already exists. Activating..."
        sudo swapon /swapfile 2>/dev/null || echo "Swap already active or failed to activate."
      fi
    fi
  fi
  
  echo "Building torchvision from source (this may take 20-40 minutes with MAX_JOBS=1)…"
  echo "NOTE: Using MAX_JOBS=1 to avoid out-of-memory errors on Jetson."
  TMPD="$(mktemp -d)"
  cd "${TMPD}"
  git clone --depth 1 --branch v0.20.0 https://github.com/pytorch/vision.git torchvision_build
  cd torchvision_build
  
  # Limit parallel jobs to 1 to avoid OOM (Out of Memory) errors
  # This significantly reduces memory usage during compilation
  export MAX_JOBS=1
  echo "Building with MAX_JOBS=${MAX_JOBS} (single-threaded to save memory)..."
  MAX_JOBS=1 python3 setup.py install
  
  cd ~
  rm -rf "${TMPD}"
  
  echo "Verifying torchvision…"
  python3 - <<'PY'
import sys
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("ERROR importing torchvision:", e)
    sys.exit(1)
PY
  
  echo "SUCCESS: torchvision installed."
else
  echo "Skipping torchvision installation."
  echo "NOTE: You can build it later with:"
  echo "  git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.20.0 && python3 setup.py install"
fi
