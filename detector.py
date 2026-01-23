# detector.py

from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import cv2
import pickle
import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "red"
TEXT_COLOR = "black"

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# use to train model
def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []

    for path in Path("training").glob("*/*"):

        name = _convert_name(path.parent.name)
        image = face_recognition.load_image_file(path)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}

        with encodings_location.open("wb") as f:
            pickle.dump(name_encodings, f)

def _convert_name(name: str) -> str:
    return name.title().replace("_", " ")

# encode_known_faces()

# recognize image
def recognize_faces(image_location: str,
                    model: str = "hog",
                    encodings_location: Path = DEFAULT_ENCODINGS_PATH
                    ) -> None:
    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)

        input_image = face_recognition.load_image_file(image_location)
        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encoding in zip(
                input_face_locations, input_face_encodings
        ):
            name = _recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            _display_face(draw, bounding_box, name)

        del draw
        pillow_image.show()

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
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

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"],
        unknown_encoding,
        tolerance=0.5
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )

    if votes:
        return votes.most_common(1)[0][0]

def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

def recognize_video(
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    source=0,  # 0 = webcam, or pass video file path
    process_every_n_frames: int = 3,
    recognition_scale: float = 0.5,
):
    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)

    video_capture = cv2.VideoCapture(source)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    detected_face_locations = []
    detected_face_names = []
    scale_back = int(1 / recognition_scale) if recognition_scale > 0 else 4

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            small_frame = cv2.resize(
                frame,
                (0, 0),
                fx=recognition_scale,
                fy=recognition_scale,
            )
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            detected_face_locations = []
            detected_face_names = []

            for (top, right, bottom, left), unknown_encoding in zip(
                face_locations, face_encodings
            ):
                name = _recognize_face(unknown_encoding, loaded_encodings)
                if not name:
                    name = "Unknown"

                detected_face_locations.append(
                    (
                        top * scale_back,
                        right * scale_back,
                        bottom * scale_back,
                        left * scale_back,
                    )
                )
                detected_face_names.append(name)

        for (top, right, bottom, left), name in zip(
            detected_face_locations, detected_face_names
        ):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Video Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

recognize_video()
