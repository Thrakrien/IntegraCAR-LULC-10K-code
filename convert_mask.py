"""Convert RGB GeoTIFF masks to indexed masks for segmentation."""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from tqdm import tqdm


# Original color IDs used by the 25-class mask conversion.
# Class names below are kept in Portuguese to match the source taxonomy.
RGB_TO_CLASS_25: dict[tuple[int, int, int], int] = {
    (150, 150, 150): 0,   # Afloramento Rochoso
    (251, 154, 153): 1,   # Área Edificada
    (69, 175, 213): 2,    # Brejo
    (150, 109, 207): 3,   # Campo Rupestre/Altitude
    (128, 214, 16): 4,    # Cultivo Agrícola - Abacaxi
    (247, 223, 8): 5,     # Cultivo Agrícola - Banana
    (119, 9, 29): 6,      # Cultivo Agrícola - Café
    (209, 163, 117): 7,   # Cultivo Agrícola - Cana-De-Açúcar
    (231, 67, 97): 8,     # Cultivo Agrícola - Coco-Da-Baía
    (245, 141, 23): 9,    # Cultivo Agrícola - Mamão
    (55, 196, 201): 10,   # Cultivo Agrícola - Outros Cultivos Permanentes
    (225, 175, 38): 11,   # Cultivo Agrícola - Outros Cultivos Temporários
    (81, 77, 77): 12,     # Extração Mineração
    (211, 127, 122): 13,  # Macega
    (156, 68, 203): 14,   # Mangue
    (133, 196, 221): 15,  # Massa D'Água
    (13, 103, 19): 16,    # Mata Nativa
    (51, 160, 44): 17,    # Mata Nativa em Estágio Inicial de Regeneração
    (31, 205, 170): 18,   # Outros
    (178, 214, 32): 19,   # Pastagem
    (207, 103, 65): 20,   # Reflorestamento - Eucalipto
    (243, 184, 129): 21,  # Reflorestamento - Pinus
    (151, 132, 233): 22,  # Reflorestamento - Seringueira
    (63, 231, 161): 23,   # Restinga
    (245, 222, 193): 24,  # Solo Exposto
}

# Grouping from the project's five-class article reproduction.
# Class IDs: 0 Infraestrutura, 1 Vegetação, 2 Agropastoril,
# 3 Macega, 4 Água.
CLASS_5_BY_CLASS_25: tuple[int, ...] = (
    0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0,
    3, 1, 4, 1, 1, 0, 2, 1, 1, 1, 1, 0,
)


def color_mapping(num_classes: int) -> dict[tuple[int, int, int], int]:
    """Return the RGB to class ID mapping for the selected scheme."""
    if num_classes == 25:
        return RGB_TO_CLASS_25
    if num_classes == 5:
        return {
            rgb: CLASS_5_BY_CLASS_25[class_id]
            for rgb, class_id in RGB_TO_CLASS_25.items()
        }
    raise ValueError("num_classes must be 5 or 25")


def convert_rgb_mask(
    rgb_mask: np.ndarray,
    mapping: dict[tuple[int, int, int], int],
) -> np.ndarray:
    """Convert an H x W x 3 uint8 RGB array to class IDs.

    Raises:
        ValueError: If the RGB array has an invalid shape or unknown colors.
    """
    if rgb_mask.ndim != 3 or rgb_mask.shape[2] != 3:
        raise ValueError("RGB mask must have shape (height, width, 3)")
    if rgb_mask.dtype != np.uint8:
        raise ValueError("RGB mask bands must be uint8")

    class_map = np.full(rgb_mask.shape[:2], 255, dtype=np.uint8)
    for rgb, class_id in mapping.items():
        class_map[np.all(rgb_mask == rgb, axis=-1)] = class_id

    unknown = class_map == 255
    if np.any(unknown):
        colors, counts = np.unique(
            rgb_mask[unknown], axis=0, return_counts=True
        )
        first_color = tuple(int(value) for value in colors[0])
        raise ValueError(
            f"unknown RGB color {first_color} occurs {int(counts[0])} "
            f"time(s); {len(colors)} unknown color(s) in total"
        )
    return class_map


def convert_file(
    input_path: Path,
    output_path: Path,
    mapping: dict[tuple[int, int, int], int],
) -> None:
    """Convert one RGB GeoTIFF, retaining its spatial metadata."""
    with rasterio.open(input_path) as src:
        if src.count < 3:
            raise ValueError("input must have at least three RGB bands")
        rgb_mask = np.moveaxis(src.read((1, 2, 3)), 0, -1)
        profile = src.profile.copy()

    class_map = convert_rgb_mask(rgb_mask, mapping)
    profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(class_map, 1)


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    num_classes: int = 5,
    overwrite: bool = False,
) -> int:
    """Convert TIFF masks directly inside a directory in sorted order.

    Returns:
        Number of masks converted.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")

    input_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".tif"
    )
    if not input_paths:
        raise ValueError(f"no .tif files found in {input_dir}")

    output_paths = [output_dir / f"{path.stem}.tif" for path in input_paths]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("input files would produce duplicate output names")
    if not overwrite:
        existing = next((path for path in output_paths if path.exists()), None)
        if existing is not None:
            raise ValueError(
                f"output already exists: {existing}; use --overwrite"
            )

    mapping = color_mapping(num_classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path, output_path in tqdm(
        zip(input_paths, output_paths),
        total=len(input_paths),
        desc="Converting masks",
    ):
        try:
            convert_file(input_path, output_path, mapping)
        except (ValueError, rasterio.errors.RasterioError) as exc:
            raise ValueError(f"{input_path}: {exc}") from exc
    return len(input_paths)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI options and convert the selected masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing the original RGB GeoTIFF masks."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the converted single-band GeoTIFF masks."
    )
    parser.add_argument(
        "--classes", type=int, choices=(5, 25), default=5,
        help="Number of output classes (default: 5)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace converted masks that already exist."
    )
    args = parser.parse_args(argv)

    try:
        count = convert_directory(
            args.input_dir, args.output_dir, args.classes, args.overwrite
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Converted {count} mask(s) to {args.classes} "
        f"classes in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
