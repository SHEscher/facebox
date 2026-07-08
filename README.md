# Facebox

*Face detection, landmarking, bounding boxes, cropping and more.*

Much of the functionality of this package is built around `mediapipe`
(see [Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)).

The required mediapipe model files are downloaded automatically on first use,
so no manual model setup is needed.


## Installation

Install straight from GitHub with [uv](https://docs.astral.sh/uv/) (or `pip`):

```shell
uv add "git+https://github.com/SHEscher/facebox"
# or
pip install "git+https://github.com/SHEscher/facebox"
```

Requires Python `>=3.12`.


## Usage

```python
from facebox import (
    find_landmarks,     # find face landmarks (and optional blendshapes)
    find_bounding_box,  # detect the face and draw a bounding box
    crop_square,        # crop a square image centred on the face
    get_landmark_model,       # fetch/locate the landmark model file
    get_face_detector_model,  # fetch/locate the detector model file
)
```

### API overview

| Function | What it does |
| --- | --- |
| `find_landmarks(image_path, show_landmarks=True, plot_blendshapes=False)` | Returns the mediapipe landmark result and an annotated image. |
| `find_bounding_box(image_path, show_bounding_box=True)` | Returns the detection result and an image with the face bounding box drawn. |
| `crop_square(image_path, perc_margin=1.0, plot_cropped_image=True, save=False)` | Crops a square around the detected face; `perc_margin` adds padding (× face size). With `save=True` it writes `<name>_cropped<ext>` next to the original. |
| `get_landmark_model()` / `get_face_detector_model()` | Return the local model path, downloading the model file if missing. |

Example — crop an image to a square around the face:

```python
from facebox import crop_square

cropped = crop_square("data/faces/portrait.jpg", perc_margin=1.0, save=True)
```

## Scripts and notebooks

[marimo](https://marimo.io) notebooks that wrap toolbox functions, such as `crop_square` can be found in `notebooks/`.
These can be run both as **interactive apps** and as **command-line scripts**.

*Note, these notebooks are self-contained [PEP 723](https://peps.python.org/pep-0723/) scripts.*

### Interactive

> Here is an example for `square_crop.py` script/notebook (more to come) 

Upload one or more face images, tune the crop margin with a slider, preview the
square crops, name them, and save them to a chosen folder:

```shell
uv run marimo run notebooks/square_crop.py    # run as an interactive app
# or
uv run marimo edit notebooks/square_crop.py   # author / edit
```

### Command line

Batch-crop every non-square image in a directory, in place (each saved as
`<name>_cropped<ext>`):

```shell
uv run notebooks/square_crop.py --input FACE_IMAGE_DIR --margin 1.0
```

| Argument | Default | Description |
| --- | --- | --- |
| `--input` | `data/faces` | Directory of face images to crop. |
| `--margin` | `1.0` | Padding around the face, as a multiple of the face size. |


## Related work

See the [MTCNN](https://github.com/ipazc/mtcnn) model.
