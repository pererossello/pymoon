import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


def _normalize_extension(extension: str) -> str:
    if not extension:
        raise ValueError("An image extension must be provided.")
    return extension if extension.startswith(".") else f".{extension}"


def _frame_sort_key(path: Path) -> Tuple[str, int]:
    stem = path.stem
    prefix, suffix = stem.rsplit("_", 1) if "_" in stem else (stem, "")
    return (prefix, int(suffix)) if suffix.isdigit() else (prefix, 0)


def _sequence_metadata(
    first_file: Path, digit_format: Optional[str]
) -> Tuple[str, str, str]:
    stem = first_file.stem
    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            inferred_format = digit_format or f"0{len(suffix)}d"
            return prefix, inferred_format, "_"
    inferred_format = digit_format or "04d"
    return stem, inferred_format, ""


def _resolve_output_dir(folder: Path, output_dir: Optional[os.PathLike]) -> Path:
    if output_dir is None:
        return folder.parent
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def image_to_mp4(
    folder,
    title="video",
    fps=36,
    digit_format=None,
    res=None,
    resize_factor=1,
    custom_bitrate=None,
    extension=".jpg",
    reverse=False,
    output_dir=None,
):
    folder_path = Path(folder)
    ext = _normalize_extension(extension)
    files = sorted(folder_path.glob(f"*{ext}"), key=_frame_sort_key)
    if not files:
        raise ValueError("No image files found in the specified folder.")

    first_file = files[0]
    prefix, digit_format, separator = _sequence_metadata(first_file, digit_format)

    with Image.open(first_file) as im:
        resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resx * resize_factor)
        resy = int(resy * resize_factor)
        resx = max(resx + (resx % 2), 2)
        resy = max(resy + (resy % 2), 2)

    output_folder = _resolve_output_dir(folder_path, output_dir)
    output_file = output_folder / f"{title}.mp4"

    crf = 5
    bitrate = custom_bitrate or "5000k"
    preset = "slow"
    tune = "film"

    pattern = folder_path / f"{prefix}{separator}%{digit_format}{ext}"
    filters = [f"scale={resx}:{resy}"]
    if reverse:
        filters.append("reverse")

    command = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-tune",
        tune,
        "-b:v",
        str(bitrate),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        ",".join(filters),
        str(output_file),
    ]

    subprocess.run(command, check=True)
    return output_file


def image_to_gif(
    folder,
    title="video",
    outfold=None,
    fps=24,
    digit_format=None,
    quality=500,
    max_colors=256,
    extension=".png",
    reverse=False,
):
    folder_path = Path(folder)
    ext = _normalize_extension(extension)
    files = sorted(folder_path.glob(f"*{ext}"), key=_frame_sort_key)
    if not files:
        raise ValueError("No image files found in the specified folder.")

    first_file = files[0]
    prefix, digit_format, separator = _sequence_metadata(first_file, digit_format)

    output_folder = _resolve_output_dir(folder_path, outfold)
    output_file = output_folder / f"{title}.gif"
    palette_file = output_folder / f"{title}_palette.png"

    pattern = folder_path / f"{prefix}{separator}%{digit_format}{ext}"
    base_filters = [f"fps={fps}", f"scale={quality}:-1:flags=lanczos"]
    palette_filters = base_filters + [f"palettegen=max_colors={max_colors}"]
    gif_filters = base_filters.copy()
    if reverse:
        gif_filters.append("reverse")
    paletteuse = "paletteuse=dither=bayer:bayer_scale=5"

    palette_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(pattern),
        "-vf",
        ",".join(palette_filters),
        str(palette_file),
    ]

    gif_command = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-i",
        str(pattern),
        "-i",
        str(palette_file),
        "-lavfi",
        f"{','.join(gif_filters)} [x]; [x][1:v] {paletteuse}",
        str(output_file),
    ]

    try:
        subprocess.run(palette_command, check=True)
        subprocess.run(gif_command, check=True)
    finally:
        if palette_file.exists():
            palette_file.unlink()

    return output_file


def make_video(
    folder,
    title="video",
    output_format="gif",
    extension=".png",
    **kwargs,
):
    """
    Create a GIF or MP4 from a folder containing sequentially numbered frames
    such as ``image_0001.png``.
    """
    fmt = output_format.lower()
    if fmt in {"gif", "animated_gif"}:
        return image_to_gif(
            folder=folder,
            title=title,
            extension=extension,
            **kwargs,
        )
    if fmt in {"mp4", "video"}:
        return image_to_mp4(
            folder=folder,
            title=title,
            extension=extension,
            **kwargs,
        )
    raise ValueError(f"Unsupported output format '{output_format}'.")
