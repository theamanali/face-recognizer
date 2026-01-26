# detector.py
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import cv2
import pickle
import face_recognition
import argparse
import time
from typing import Optional
from tqdm import tqdm

MODEL_NAME = "hog"
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "red"
TEXT_COLOR = "black"

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# use to train model
def encode_known_faces(model: str = MODEL_NAME, encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []

    training_paths = sorted(Path("training").glob("*/*"))
    for path in tqdm(training_paths, desc="Encoding faces", unit="image"):
        name = _convert_name(path.parent.name)
        image = face_recognition.load_image_file(path)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    if not encodings:
        raise ValueError("No faces found in training images.")

    name_encodings = {"names": names, "encodings": encodings, "model": model}

    with encodings_location.open("wb") as f:
        pickle.dump(name_encodings, f)

def _convert_name(name: str) -> str:
    return name.title().replace("_", " ")

# encode_known_faces()

# recognize image
def recognize_faces(image_location: str,
                    model: Optional[str] = None,
                    encodings_location: Path = DEFAULT_ENCODINGS_PATH
                    ) -> None:
    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)
        model = _resolve_model(loaded_encodings, model)

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
    return None

def _resolve_model(
    loaded_encodings,
    requested_model: Optional[str],
) -> str:
    trained_model = loaded_encodings.get("model")
    if trained_model is None:
        raise ValueError(
            "Encodings are missing model metadata. Retrain with '--train' "
            "and optionally '--model {hog|cnn}' before recognition."
        )
    if requested_model and requested_model != trained_model:
        raise ValueError(
            f"Encodings were trained with '{trained_model}' model. "
            f"Use '--model {trained_model}' (or omit --model) for recognition, "
            f"or retrain with '--train --model {requested_model}'."
        )
    return trained_model

def validate(model: Optional[str] = None):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

def recognize_video(
    model: Optional[str] = None,
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    source=0,  # 0 = webcam, or pass video file path
    process_every_n_frames: int = 4,
    recognition_scale: float = 0.25,
):
    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)
    model = _resolve_model(loaded_encodings, model)

    video_capture = cv2.VideoCapture(source)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    detected_face_locations = []
    detected_face_names = []
    scale_back = 1.0 / recognition_scale

    start_time = time.time()
    frame_counter = 0
    processing_times = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        frame_counter += 1
        if frame_count % process_every_n_frames == 0:
            frame_start = time.time()

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
                        int(top * scale_back),
                        int(right * scale_back),
                        int(bottom * scale_back),
                        int(left * scale_back),
                    )
                )
                detected_face_names.append(name)

            frame_end = time.time()
            processing_times.append(frame_end - frame_start)

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

    total_time = time.time() - start_time
    fps = frame_counter / total_time
    avg_processing_time = sum(processing_times) / len(processing_times)

    print("\n=== Performance Metrics ===")
    print(f"Total Runtime: {total_time:.2f} seconds")
    print(f"Total Frames: {frame_counter}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average Frame Processing Time: {avg_processing_time:.4f} seconds")

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Recognize faces in an image")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train on input data")
    group.add_argument(
        "--validate", action="store_true", help="Validate trained model"
    )
    group.add_argument(
        "--image", metavar="PATH", help="Path to an image with an unknown face"
    )
    group.add_argument(
        "--webcam", action="store_true", help="Test the model with a webcam"
    )
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default=None,
        help=f"Face detection model for training only. Default is '{MODEL_NAME}'.",
    )
    args = parser.parse_args()

    if args.model and not args.train:
        parser.error("--model can only be used with --train.")

    if args.train:
        encode_known_faces(model=args.model or MODEL_NAME)
    if args.validate:
        validate()
    if args.image:
        recognize_faces(image_location=args.image)
    if args.webcam:
        recognize_video()

if __name__ == "__main__":
    main()
