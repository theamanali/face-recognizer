import pickle
from pathlib import Path

import face_recognition
from tqdm import tqdm

from config import (
    DEFAULT_ENCODINGS_PATH,
    MODEL_NAME,
    OUTPUT_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
)


def ensure_directories() -> None:
    for directory in (TRAINING_DIR, OUTPUT_DIR, VALIDATION_DIR):
        directory.mkdir(exist_ok=True)


def encode_known_faces(
    model: str = MODEL_NAME,
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    names = []
    encodings = []

    training_paths = sorted(TRAINING_DIR.glob("*/*"))
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


def resolve_model(loaded_encodings) -> str:
    trained_model = loaded_encodings.get("model")
    if trained_model is None:
        raise ValueError(
            "Encodings are missing model metadata. Retrain with '--train' "
            "and optionally '--model {hog|cnn}' before recognition."
        )
    return trained_model


def load_encodings(encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    try:
        with encodings_location.open("rb") as f:
            loaded_encodings = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Encodings file not found at '{encodings_location}'. Run '--train' first."
        ) from exc
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(
            f"Encodings file at '{encodings_location}' is invalid or corrupted. "
            "Retrain with '--train' to regenerate it."
        ) from exc

    model = resolve_model(loaded_encodings)
    return loaded_encodings, model
