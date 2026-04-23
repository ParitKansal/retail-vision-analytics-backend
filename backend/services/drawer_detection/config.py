
import os
from typing import List

# Manually load .env since python-dotenv might not be available or should be minimal
def load_env_file(filepath: str):
    if not os.path.exists(filepath):
        print(f"Warning: .env file not found at {filepath}")
        return

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Load env variables from local .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_env_file(env_path)

class Config:
    # Brightness Thresholds
    BRIGHT_MIN_THRESHOLD = int(os.getenv('BRIGHT_MIN_THRESHOLD', 175))
    DARK_MAX_THRESHOLD = int(os.getenv('DARK_MAX_THRESHOLD', 80))
    BRIGHT_AVG_MIN_THRESHOLD = int(os.getenv('BRIGHT_AVG_MIN_THRESHOLD', 102))
    DARK_AVG_MAX_THRESHOLD = int(os.getenv('DARK_AVG_MAX_THRESHOLD', 80))

    # Geometric Fitting
    _scales_str = os.getenv('FIT_SCALES', '1.0,0.5,0.25,0.2')
    FIT_SCALES: List[float] = [float(s) for s in _scales_str.split(',')]
    FIT_ANGLE_MAX = int(os.getenv('FIT_ANGLE_MAX', 15))
    FIT_ANGLE_STEP = int(os.getenv('FIT_ANGLE_STEP', 1))
    ERODE_PIXELS = int(os.getenv('ERODE_PIXELS', 1))

    # Output / Debug
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    SAVE_RAW_IMAGES = os.getenv('SAVE_RAW_IMAGES', 'False').lower() == 'true'
    SAVE_ANNOTATED_IMAGES = os.getenv('SAVE_ANNOTATED_IMAGES', 'True').lower() == 'true'
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')

    @classmethod
    def print_config(cls):
        """Prints the current configuration for verification."""
        print("Driver Status Detector Configuration:")
        print(f"  Bright Thresholds: Avg >= {cls.BRIGHT_AVG_MIN_THRESHOLD}, Max >= {cls.BRIGHT_MIN_THRESHOLD}")
        print(f"  Dark Thresholds:   Avg <  {cls.DARK_AVG_MAX_THRESHOLD}, Max <= {cls.DARK_MAX_THRESHOLD}")
        print(f"  Fit Scales:        {cls.FIT_SCALES}")
        print(f"  Angle Range:       +/- {cls.FIT_ANGLE_MAX} deg (step {cls.FIT_ANGLE_STEP})")
        print(f"  Output Dir:        {cls.OUTPUT_DIR}")
        print(f"  Debug Mode:        {cls.DEBUG_MODE}")
