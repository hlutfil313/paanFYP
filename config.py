"""
Configuration file for Thermal Image Enhancement Application
"""

# Application settings
APP_NAME = "Thermal Image Enhancement for Defect Detection"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Thermal Image Enhancement Team"

# GUI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600

# Image display settings
IMAGE_DISPLAY_WIDTH = 400
IMAGE_DISPLAY_HEIGHT = 300
COMPARISON_WINDOW_WIDTH = 1000
COMPARISON_WINDOW_HEIGHT = 600

# Enhancement settings
DEFAULT_ENHANCEMENT_METHOD = "histogram_equalization"
ENHANCEMENT_METHODS = [
    "histogram_equalization",
    "adaptive_histogram_equalization",
    "gaussian_filtering",
    "edge_enhancement",
    "temperature_mapping",
    "multi_scale_enhancement"
]

# Defect detection settings
DEFAULT_DETECTION_METHOD = "threshold_based"
DETECTION_METHODS = [
    "threshold_based",
    "contour_based",
    "blob_detection",
    "edge_based",
    "region_growing"
]

# File settings
SUPPORTED_IMAGE_FORMATS = [
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"
]
DEFAULT_SAVE_FORMAT = "PNG"
DEFAULT_SAVE_QUALITY = 95

# Processing settings
MAX_IMAGE_SIZE = 2048  # Maximum image dimension for processing
DEFAULT_BLUR_KERNEL = 5
DEFAULT_THRESHOLD_PERCENTILE = 95
DEFAULT_MIN_DEFECT_AREA = 50

# Color maps for thermal visualization
THERMAL_COLORMAPS = [
    "hot",
    "plasma",
    "viridis",
    "inferno",
    "magma",
    "jet",
    "rainbow"
]

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "thermal_enhancer.log"

# Output directories
DATA_DIR = "data"
SAMPLES_DIR = "data/samples"
RESULTS_DIR = "data/results"
COMPARISONS_DIR = "data/comparisons"

# Create directories if they don't exist
import os
for directory in [DATA_DIR, SAMPLES_DIR, RESULTS_DIR, COMPARISONS_DIR]:
    os.makedirs(directory, exist_ok=True) 