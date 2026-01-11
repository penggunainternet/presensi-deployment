#!/bin/bash
set -e

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Detect Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
echo "Python version detected: 3.$((PYTHON_VERSION % 100))"

# Try to install tflite runtime based on Python version
echo "Attempting to install TensorFlow Lite Runtime..."
pip install tflite-runtime==2.13.0 --no-deps 2>/dev/null || \
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.13.0-cp311-cp311-linux_x86_64.whl 2>/dev/null || \
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.13.0-cp310-cp310-linux_x86_64.whl 2>/dev/null || \
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.13.0-cp39-cp39-linux_x86_64.whl 2>/dev/null || \
echo "⚠️  TFLite Runtime not available - will use OpenCV fallback"

echo "Installing requirements..."
pip install -r requirements.txt

echo "Build complete!"
