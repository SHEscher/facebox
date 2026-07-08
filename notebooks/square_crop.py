"""
Crop face images to squares.

Run me with:

```shell
uv run square_crop.py --input FACE_IMAGE_DIR
```

"""
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "facebox @ file:////Users/zimon/Bibliothek/Software/Python/facebox",
#     "opencv-python",
# ]
# ///

import argparse
from pathlib import Path

import cv2
from facebox import crop_square

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Crop face images to squares.")
    parser.add_argument("--input", "-i", type=Path, default=Path("data/faces"), help="Path to directory with face images (default: data/faces)")
    parser.add_argument("--margin", "-m", type=float, default=1.0, help="Percentage margin for cropping (default: 1.0)")
    args = parser.parse_args()

    # Validate input directory
    face_dir: Path = args.input
    if not face_dir.is_dir():
        print(f"Error: '{face_dir}' is not a valid directory.")
        raise SystemExit(1)

    # Image suffixes to process
    IMG_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png"}  # , ".bmp", ".tiff"}

    # Process face images, crop only if not squared
    n_cropped = 0
    n_skipped = 0
    for suffix in IMG_SUFFIXES:
        for img_path in [*face_dir.glob(f"*{suffix}"), *face_dir.glob(f"*{suffix.upper()}")]:
            if "_cropped" in img_path.stem:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read '{img_path}', skipping.")
                n_skipped += 1
                continue

            if img.shape[0] == img.shape[1]:
                n_skipped += 1
                continue

            print(f"Cropping {img_path.name} (shape: {img.shape[:2]})")
            crop_square(
                image_path=img_path,
                perc_margin=args.margin,
                plot_cropped_image=False,
                save=True,
            )
            n_cropped += 1

    print(f"\nDone: {n_cropped} image(s) cropped, {n_skipped} skipped.")
