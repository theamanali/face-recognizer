from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw

from config import (
    BOUNDING_BOX_COLOR,
    DEFAULT_ENCODINGS_PATH,
    DEFAULT_TOLERANCE,
    TEXT_COLOR,
    VALIDATION_DIR,
)
from face_matching import recognize_face
from face_store import load_encodings


def recognize_faces(
    image_location: str,
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    loaded_encodings, model = load_encodings(encodings_location)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(
        input_image,
        input_face_locations,
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = recognize_face(
            unknown_encoding,
            loaded_encodings,
            tolerance=tolerance,
        )
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate(
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    for filepath in VALIDATION_DIR.rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()),
                encodings_location=encodings_location,
                tolerance=tolerance,
            )
