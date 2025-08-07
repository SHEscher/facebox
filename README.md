# Facebox

*Face detection, landmarking, bounding boxes, cropping and more.*

Much of the functionality of this package is built around `mediapipe`
(see [Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)).


## Usage

```python
from facebox import (
    find_landmarks,  # find face landmarks
    find_bounding_box,  # draw bounding box around face
    crop_square,  # crop image around face, where width==height
)
```

## Related work

See the [MTCNN](https://github.com/ipazc/mtcnn) model.
