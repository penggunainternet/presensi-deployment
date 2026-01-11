#!/bin/bash
set -e

echo "Installing TensorFlow Lite Runtime..."
pip install --upgrade pip setuptools wheel

# Install tflite runtime untuk Linux x86_64
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.13.0-cp311-cp311-linux_x86_64.whl || \
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.13.0-cp310-cp310-linux_x86_64.whl || \
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.13.0-cp39-cp39-linux_x86_64.whl

echo "Installing requirements..."
pip install -r requirements.txt

echo "Build complete!"
