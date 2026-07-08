# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo",
#     "facebox @ git+https://github.com/SHEscher/facebox",
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
def _(Path, crop_square, file_upload, margin_slider, tempfile):
    def compute_crops(uploaded_files, perc_margin):
        crops = []
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
            crops.append((uploaded.name, cropped))
        return crops

    crops = compute_crops(file_upload.value, margin_slider.value) if file_upload.value else []
    return (crops,)


@app.cell(hide_code=True)
def _(crops, mo, plt):
    def preview_crops(cropped_items):
        panels = []
        for name, cropped in cropped_items:
            fig, ax = plt.subplots()
            ax.imshow(cropped)
            ax.set_title(f"{name} → {cropped.shape[1]}×{cropped.shape[0]}")
            ax.axis("off")
            panels.append(fig)
        return panels

    _panels = preview_crops(crops)
    (mo.hstack(_panels) if _panels else mo.md("*Upload one or more images to preview the square crop.*"))
    return


@app.cell(hide_code=True)
def _(Path, crops, mo):
    def default_name(name):
        return f"{Path(name).stem}_cropped{Path(name).suffix or '.png'}"

    name_inputs = mo.ui.array([mo.ui.text(value=default_name(name), label=f"Save “{name}” as") for name, _ in crops])
    _start = Path("data/faces") if Path("data/faces").is_dir() else Path.cwd()
    output_dir = mo.ui.file_browser(
        initial_path=_start,
        selection_mode="directory",
        multiple=False,
        label="Output folder — tick ☑ the destination folder (click the name to navigate in)",
    )
    save_button = mo.ui.run_button(label="💾 Save cropped image(s)")

    mo.vstack([output_dir, name_inputs, save_button]) if crops else mo.md("")
    return name_inputs, output_dir, save_button


@app.cell(hide_code=True)
def _(Path, crops, cv2, mo, name_inputs, output_dir, save_button):
    def save_crops(cropped_items, names, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for (orig_name, cropped), chosen in zip(cropped_items, names):
            out_name = chosen.strip() or f"{Path(orig_name).stem}_cropped"
            if not Path(out_name).suffix:
                out_name += Path(orig_name).suffix or ".png"
            dest = out_dir / out_name
            # facebox returns RGB(A) arrays; OpenCV writes BGR(A). Convert by
            # channel count so alpha isn't mistaken for a colour channel.
            if cropped.ndim == 3 and cropped.shape[2] == 4:
                bgr = cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)
            else:
                bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dest), bgr)
            saved.append(dest)
        return saved

    if not save_button.value or not crops:
        _result = mo.md("*Set names, tick an output folder, then click **Save**.*")
    elif output_dir.path(0) is None:
        _result = mo.md("⚠️ *Tick ☑ the destination folder in the browser above first.*")
    else:
        _saved = save_crops(crops, name_inputs.value, output_dir.path(0))
        _result = mo.md("**Saved:**\n\n" + "\n".join(f"- `{d}`" for d in _saved))
    _result
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
