# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo",
#     "facebox @ file:////Users/zimon/Bibliothek/Software/Python/facebox",
#     "matplotlib",
#     "opencv-python",
# ]
# ///
"""
Crop face images to squares.

Interactive (upload images, tune the margin, preview crops):

```shell
uv run marimo edit square_crop.py      # edit / author
uv run marimo run square_crop.py       # run as an app
```

Command line (batch-crop every non-square image in a directory, in place):

```shell
uv run square_crop.py --input FACE_IMAGE_DIR --margin 1.0
```

marimo detects how it is invoked (`mo.app_meta().mode`), so the same file works
both interactively and as a CLI script.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Square-crop faces

    Upload one or more face images, tune the crop **margin**, and preview the
    square crop centred on the detected face.

    Run this file from the command line to batch-crop a whole directory instead.

    ```shell
     uv run notebooks/square_crop.py --input data/faces --margin 1.0
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".jpg", ".jpeg", ".png"],
        multiple=True,
        kind="area",
        label="Upload face images to crop",
    )
    file_upload
    return (file_upload,)


@app.cell(hide_code=True)
def _(mo):
    margin_slider = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="Crop margin (× face size)", show_value=True)
    margin_slider
    return (margin_slider,)


@app.cell(hide_code=True)
def _(Path, crop_square, file_upload, margin_slider, mo, plt, tempfile):
    def preview_uploads(uploaded_files, perc_margin):
        panels = []
        for uploaded in uploaded_files:
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.contents)
                tmp_path = Path(tmp.name)

            cropped = crop_square(
                image_path=tmp_path,
                perc_margin=perc_margin,
                plot_cropped_image=False,
                save=False,
            )
            tmp_path.unlink(missing_ok=True)

            fig, ax = plt.subplots()
            ax.imshow(cropped)
            ax.set_title(f"{uploaded.name} → {cropped.shape[1]}×{cropped.shape[0]}")
            ax.axis("off")
            panels.append(fig)
        return panels

    _panels = preview_uploads(file_upload.value, margin_slider.value) if file_upload.value else []
    (mo.vstack(_panels) if _panels else mo.md("*Upload one or more images to preview the square crop.*"))
    return


@app.cell(hide_code=True)
def _(crop_square, cv2):
    def crop_directory(face_dir, perc_margin):
        if not face_dir.is_dir():
            print(f"Error: '{face_dir}' is not a valid directory.")
            raise SystemExit(1)

        img_suffixes = {".jpg", ".jpeg", ".png"}
        n_cropped = 0
        n_skipped = 0
        for suffix in img_suffixes:
            for img_path in [
                *face_dir.glob(f"*{suffix}"),
                *face_dir.glob(f"*{suffix.upper()}"),
            ]:
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
                    perc_margin=perc_margin,
                    plot_cropped_image=False,
                    save=True,
                )
                n_cropped += 1

        print(f"\nDone: {n_cropped} image(s) cropped, {n_skipped} skipped.")

    return (crop_directory,)


@app.cell(hide_code=True)
def _(Path, crop_directory, mo):
    if mo.app_meta().mode == "script":
        cli_args = mo.cli_args()
        cli_input_dir = Path(cli_args.get("input", "data/faces"))
        cli_margin = float(cli_args.get("margin", 1.0))
        crop_directory(cli_input_dir, cli_margin)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import tempfile
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    from facebox import crop_square

    return Path, crop_square, cv2, mo, plt, tempfile


if __name__ == "__main__":
    app.run()
