import numpy as np

# Basic Configuration
BATCH_SIZE = 128  # 保持 64，配合 Resize 可以稳定运行
AMP_ENABLED = True
DEVICE_NOISE_STD = 2e-11

# Data Processing
CROP_HEIGHT = 64
RESIZE_SIZE = (224, 224) # 新增：标准 ResNet 输入尺寸
NORM_MEAN = [0.5, 0.5, 0.5] # 修改：适配 3 通道 (ImageFolder 默认 RGB)
NORM_STD = [0.5, 0.5, 0.5]
NUM_CLASSES = 5

# Training Hyperparameters
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
WARMUP_EPOCHS = 10

# Logging & Checkpointing
LOG_DIR = "logs"
LOG_FILE = "train.log"
MODEL_SAVE_PATH = "organic_fingerprint_model.pth"

# Physical Measurement Data (Hardcoded)
LTP_RAW = [
    -3.35E-11,
    -6.26E-11,
-9.54E-11,
-1.25E-10,
-1.49E-10,
-1.70E-10,
-1.91E-10,
-2.09E-10,
-2.22E-10,
-2.35E-10,
-2.46E-10,
-2.58E-10,
-2.68E-10,
-2.83E-10,
-2.96E-10,
-3.07E-10,
-3.17E-10,
-3.31E-10,
-3.40E-10,
-3.50E-10,
-3.64E-10,
-3.78E-10,
-3.89E-10,
-4.03E-10,
-4.17E-10,
-4.31E-10,
-4.39E-10,
-4.48E-10,
-4.59E-10
]

LTD_RAW = [
    -4.41E-10,
-4.18E-10,
-3.65E-10,
-3.16E-10,
-2.71E-10,
-2.46E-10,
-2.11E-10,
-1.84E-10,
-1.67E-10,
-1.49E-10,
-1.33E-10,
-1.13E-10,
-9.29E-11,
-8.74E-11,
-7.33E-11,
-7.28E-11,
-6.78E-11,
-5.99E-11,
-5.31E-11,
-5.40E-11,
-4.85E-11,
-4.40E-11,
-4.11E-11,
-3.92E-11,
-3.59E-11,
-3.18E-11,
-2.91E-11,
-2.90E-11,
-2.71E-11,
]

ltp_arr = np.array(LTP_RAW)
ltd_arr = np.array(LTD_RAW)

global_min = min(ltp_arr.min(), ltd_arr.min())
global_max = max(ltp_arr.max(), ltd_arr.max())

def normalize(data, g_min, g_max):
    return (data - g_min) / (g_max - g_min)

ltp_norm = normalize(ltp_arr, global_min, global_max)
ltd_norm = normalize(ltd_arr, global_min, global_max)

def calculate_delta_and_fit(norm_data):
    g_old = norm_data[:-1]
    g_new = norm_data[1:]
    delta = g_new - g_old
    coeffs = np.polyfit(g_old, delta, 3)
    return coeffs

LTP_POLY = calculate_delta_and_fit(ltp_norm)
LTD_POLY = calculate_delta_and_fit(ltd_norm)

PHYSICAL_MIN = global_min
PHYSICAL_MAX = global_max