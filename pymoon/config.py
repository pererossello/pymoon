"""
Configuration file for pymoon package.
Contains physical parameters and hardcoded paths to data files.
"""

import os

# Get the package directory
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

# Data directory (assuming files are in pymoon/files/ relative to project root)
DATA_DIR = os.path.join(PROJECT_ROOT, "files")

# Hardcoded paths to TIF files
PATH_LROC_COLOR_8K = os.path.join(DATA_DIR, "lroc_color_poles_8k.tif")
PATH_LROC_COLOR_4K = os.path.join(DATA_DIR, "lroc_color_poles_4k.tif")
PATH_LROC_COLOR_2K = os.path.join(DATA_DIR, "lroc_color_poles_2k.tif")
PATH_LDEM_16 = os.path.join(DATA_DIR, "ldem_16.tif")
PATH_LDEM_4 = os.path.join(DATA_DIR, "ldem_4.tif")

# Moon constants
R_MOON = 1737.4  # Moon radius in km
R_SUN = 6.9634e5  # Sun radius in km
D_MOON_SUN = 1.496e8  # Moon-Sun distance in km

# Default parameters
DEFAULT_MAX_STEPS = 50
DEFAULT_PENUMBRA = True