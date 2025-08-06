"""
Collection of functions to process face images.

Most functions are wrapped around the tools in `mediapipe` (see their Apache 2.0 license):

    * https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

Author: Simon M. Hofmann
Years: 2025
"""

# %% Import
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
LANDMARK_MODEL_NAME = (
    "face_landmarker_v2_with_blendshapes.task"  # "face_landmarker.task"
)
LANDMARK_MODEL_PATH = Path("./model") / LANDMARK_MODEL_NAME
DETECTION_MODEL_NAME = "blaze_face_short_range.tflite"  # "detector.tflite"
DETECTION_MODEL_PATH = Path("./model") / DETECTION_MODEL_NAME

# For bounding box
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _draw_landmarks_on_image(bgr_image, detection_result):
    """Draw landmarks on image."""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(bgr_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def _plot_face_blendshapes_bar_graph(face_blendshapes):
    """Extract the face blendshapes category names and scores."""
    face_blendshapes_names = [
        face_blendshapes_category.category_name
        for face_blendshapes_category in face_blendshapes
    ]
    face_blendshapes_scores = [
        face_blendshapes_category.score
        for face_blendshapes_category in face_blendshapes
    ]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(
        face_blendshapes_ranks,
        face_blendshapes_scores,
        label=[str(x) for x in face_blendshapes_ranks],
    )
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(
            patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top"
        )

    ax.set_xlabel("Score")
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def get_landmark_model(model_path: str | Path = LANDMARK_MODEL_PATH) -> Path:
    """Check if landmark model exists, otherwise download it."""
    model_path = Path(model_path)
    if not model_path.is_file():
        # Download the default model
        import urllib.request

        url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task"
        )

        LANDMARK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, LANDMARK_MODEL_PATH)

        if not LANDMARK_MODEL_PATH.is_file():
            terminal_cmd = f"mkdir model; wget -O '{LANDMARK_MODEL_PATH}' -q '{url}'"
            msg = f"Could not find model '{model_path}'.\nTry to download this via the terminal:\n\n\t{terminal_cmd}\n"
            raise FileNotFoundError(msg)

    return model_path


def get_face_detector_model(model_path: str | Path = DETECTION_MODEL_PATH) -> Path:
    """Check if face detector model exists, otherwise download it."""
    model_path = Path(model_path)
    if not model_path.is_file():
        # Load default model
        import urllib.request

        url = (
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/"
            f"{DETECTION_MODEL_NAME}"
        )
        DETECTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, DETECTION_MODEL_PATH)

        if not DETECTION_MODEL_PATH.is_file():
            terminal_cmd = (
                f"mkdir model; wget -q -O '{DETECTION_MODEL_PATH}' -q '{url}'"
            )
            msg = f"Could not find model '{model_path}'.\nTry to download this via the terminal:\n\n\t{terminal_cmd}\n"
            raise FileNotFoundError(msg)

    return model_path


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> None | tuple[int, int]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        """Check if normalized value is valid."""
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def _visualize_bounding_box(image: np.ndarray, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.
    Returns:
        Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = "" if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            annotated_image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
    return annotated_image


def find_landmarks(
    image_path: str | Path, show_landmarks: bool = True, plot_blendshapes: bool = False
) -> tuple[mp.tasks.vision.FaceLandmarkerResult, np.ndarray]:
    """Find face landmarks in the input image."""
    # Set options for landmark model
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=get_landmark_model(LANDMARK_MODEL_PATH)
        ),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,  # ...RunningMode.VIDEO, ...RunningMode.LIVE_STREAM
    )

    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file(
        str(image_path)
    )  # image loaded as SRGBA (4 channels)
    # Load the input image from a numpy array.
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    # The landmarker is initialized and applied
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Perform face landmarking on the provided single image.
        # The face landmarker must be created with the image mode (see above `running_mode`)
        face_landmarker_result = landmarker.detect(mp_image)

    # Draw the annotated image
    annotated_image = _draw_landmarks_on_image(
        bgr_image=cv2.cvtColor(
            mp_image.numpy_view(), cv2.COLOR_RGB2BGR
        ),  # requires BGR images
        detection_result=face_landmarker_result,
    )
    if show_landmarks:
        cv2.imshow(winname="Image", mat=annotated_image)
        print("\nPress any key to exit image view\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Visualize the face blendshapes categories using a bar graph
    if plot_blendshapes:
        _plot_face_blendshapes_bar_graph(
            face_blendshapes=face_landmarker_result.face_blendshapes[0]
        )

    # Print the transformation matrix.
    # print("\nTransformation matrix\n")
    # print(face_landmarker_result.facial_transformation_matrixes[0])

    return face_landmarker_result, annotated_image


def find_bounding_box(
    image_path: str | Path,
    show_bounding_box: bool = True,
) -> tuple[mp.tasks.vision.FaceDetectorResult, np.ndarray]:
    """Find bounding box in the input image."""
    base_options = mp.tasks.BaseOptions(
        model_asset_path=get_face_detector_model(DETECTION_MODEL_PATH)
    )
    options = mp.tasks.vision.FaceDetectorOptions(base_options=base_options)
    # detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    # STEP 3: Load the input image.
    mp_image = mp.Image.create_from_file(
        str(image_path)
    )  # image loaded as SRGBA (4 channels)

    # STEP 4: Detect faces in the input image.
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = _visualize_bounding_box(image_copy, detection_result)
    if show_bounding_box:
        rgb_annotated_image = cv2.cvtColor(
            annotated_image, cv2.COLOR_BGR2RGB
        )  # cv2.COLOR_BGR2RGB)
        cv2.imshow(winname="Face detection", mat=rgb_annotated_image)
        print("\nPress any key to exit image view\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detection_result, annotated_image


def crop_square(
    image_path: str | Path,
    perc_margin: float = 1.0,
    plot_cropped_image: bool = True,
    save: bool = False,
):
    """
    Crop face image to square image centred around the face.

    :param image_path: Path to the image to be cropped.
    :param perc_margin: Crop will be set perc_margin * 100 (%) around the face
    :param plot_cropped_image: If true, show  cropped image
    :param save: If true, save cropped image
    """

    if perc_margin < 0:
        msg = "perc_margin should be between 0.0 and ~ 2.0"
        raise ValueError(msg)

    image = mp.Image.create_from_file(str(image_path)).numpy_view()
    detection_result, _ = find_bounding_box(
        image_path=image_path, show_bounding_box=False
    )

    dect_0 = detection_result.detections[0]
    bbox_0 = dect_0.bounding_box

    # Adapt the bounding box, i.e., make it larger
    x_st = max(bbox_0.origin_x - round(bbox_0.height * perc_margin), 0)
    x_end = min(
        bbox_0.origin_x + bbox_0.height + round(bbox_0.height * perc_margin),
        image.shape[1],
    )
    y_st = max(bbox_0.origin_y - round(bbox_0.width * perc_margin), 0)
    y_end = min(
        bbox_0.origin_y + bbox_0.width + round(bbox_0.width * perc_margin),
        image.shape[0],
    )

    # Make sure that both axis lengths are equal
    xy_max_len = min(x_end - x_st, y_end - y_st)

    # Center axis if too long
    if x_end - x_st > xy_max_len:
        diff_len = (x_end - x_st) - xy_max_len
        x_st += diff_len // 2
        x_end -= diff_len - (diff_len // 2)

    if y_end - y_st > xy_max_len:
        diff_len = (y_end - y_st) - xy_max_len
        y_st += diff_len // 2
        y_end -= diff_len - (diff_len // 2)

    # Crop image
    cropped_image = image[y_st:y_end, x_st:x_end]

    if plot_cropped_image:
        plt.imshow(cropped_image)  # numpy has height on axis 0 and width on axis 1
        plt.title(f"{cropped_image.shape = }")
        plt.show()

    # Save cropped image
    if save:
        image_name = Path(image_path).stem + "_cropped" + Path(image_path).suffix
        cropped_image_path = Path(image_path).parent / image_name
        cv2.imwrite(
            filename=str(cropped_image_path),
            img=cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR),
        )  # save via BGR

    return cropped_image


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
