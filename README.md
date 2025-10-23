# PyMoon

PyMoon is a lightweight toolkit for experimenting with lunar phase rendering. It combines geometric shadowing, configurable photometric models, and optional Digital Elevation Maps (DEM) and texture maps. 

## Features
- Generate RGBA Moon discs from arbitrary Sun/observer vectors with `get_moon_mask`.
- Use different photometry models.
- Optional DEM-based self-shadowing and texture maps (provided in `files/`).
- Utilities to tint the rendered disc or reuse the raw illumination mask in other pipelines.

## Quick Start
```python
from PIL import Image
from pymoon import config
from pymoon.main import get_moon_mask

moon_rgba = get_moon_mask(
    N=1024,
    s_vec=(-1.0, 0.1, 0.4),      # Sun direction
    obs_vec=(0, 0, 1),          # Observer direction
    dem=config.PATH_LDEM_4,     # Optional DEM for relief
    moonmap=config.PATH_LROC_COLOR_4K,  # Optional texture
    model="ls_opposition",      # Photometry model
    shadows=True,               # Cast terrain shadows
    gamma_corr=1.6,
)

Image.fromarray(moon_rgba).save("moon.png")
```

Pass `return_mask=True` to retrieve the illumination mask directly. The helper `tint_moon_image` applies a colour wash while preserving luminance.

## Data & Notebooks
- Raster assets (LROC colour maps and LOLA DEMs) are stored in `files/`; paths are exposed through `pymoon.config`.
- Example workflows live in `notebooks/` and `notebooks_for_doc/`. Open `notebooks/example_1.ipynb` for a guided tour of the rendering pipeline.