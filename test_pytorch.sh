#!/usr/bin/env bash
# Test script for PyTorch and Torchvision installation on Jetson
# Author: Vahid Behtaji
# Email: vahidbehtaji2013@gmail.com

set -e

echo "=========================================="
echo "PyTorch & Torchvision Installation Test"
echo "=========================================="
echo ""

# Test PyTorch
echo "Testing PyTorch installation..."
echo "-----------------------------------"
python3 <<'PYTHON_EOF'
import sys

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ PyTorch path: {torch.__file__}")
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: True")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ CUDA cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"✅ GPU memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"✅ GPU compute capability: {props.major}.{props.minor}")
        
        # Test GPU tensor operations
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations: SUCCESS")
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            sys.exit(1)
    else:
        print("⚠️  CUDA available: False")
        print("⚠️  PyTorch is installed but CUDA is not available")
        print("⚠️  This may indicate:")
        print("   - CUDA drivers not properly installed")
        print("   - PyTorch was installed without CUDA support")
        print("   - libcusparseLt.so.0 missing (for JetPack 6.x)")
        
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
    print("   Run ./install_pytorch.sh to install PyTorch")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error testing PyTorch: {e}")
    sys.exit(1)
PYTHON_EOF

PYTORCH_EXIT=$?

echo ""
echo "Testing Torchvision installation..."
echo "-----------------------------------"
python3 <<'PYTHON_EOF'
import sys

try:
    import torchvision
    print(f"✅ TorchVision version: {torchvision.__version__}")
    print(f"✅ TorchVision path: {torchvision.__file__}")
    
    # Test basic torchvision functionality
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        print("✅ TorchVision transforms: SUCCESS")
    except Exception as e:
        print(f"⚠️  TorchVision transforms test failed: {e}")
        
except ImportError:
    print("⚠️  TorchVision not installed")
    print("   This is optional but required for ultralytics")
    print("   Run ./install_pytorch.sh and choose 'y' when asked to install torchvision")
except Exception as e:
    print(f"❌ Error testing TorchVision: {e}")
PYTHON_EOF

TORCHVISION_EXIT=$?

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="

if [ $PYTORCH_EXIT -eq 0 ]; then
    echo "✅ PyTorch: INSTALLED AND WORKING"
else
    echo "❌ PyTorch: FAILED"
fi

if [ $TORCHVISION_EXIT -eq 0 ]; then
    echo "✅ TorchVision: INSTALLED AND WORKING"
else
    echo "⚠️  TorchVision: NOT INSTALLED (optional)"
fi

echo ""
if [ $PYTORCH_EXIT -eq 0 ]; then
    echo "🎉 PyTorch installation test completed successfully!"
    exit 0
else
    echo "❌ PyTorch installation test failed. Please check the errors above."
    exit 1
fi
