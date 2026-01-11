import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_LOGGING_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warnings
warnings.filterwarnings('ignore')

# Model cache directory
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Buat folder jika belum ada
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
