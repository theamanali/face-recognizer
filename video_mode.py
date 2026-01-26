import time
from pathlib import Path

import cv2
import face_recognition

from config import (
    DEFAULT_DETECTION_INTERVAL,
    DEFAULT_ENCODINGS_PATH,
    DEFAULT_RECOGNITION_SCALE,
    DEFAULT_TOLERANCE,
)
from face_matching import recognize_face
from face_store import load_encodings


def _open_video_capture(source=0):
    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        video_capture.release()
        raise RuntimeError(
            f"Could not open video source '{source}'. "
            "Check camera permissions, camera availability, or source path."
        )
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return video_capture


def _detect_faces_in_frame(
    frame,
    recognition_scale,
    model,
    loaded_encodings,
    tolerance: float,
):
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
    scale_back = 1.0 / recognition_scale
    detected_face_locations = []
    detected_face_names = []

    for (top, right, bottom, left), unknown_encoding in zip(
        face_locations, face_encodings
    ):
        name = recognize_face(
            unknown_encoding,
            loaded_encodings,
            tolerance=tolerance,
        )
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

    return detected_face_locations, detected_face_names, time.time() - frame_start


def _draw_video_faces(frame, detected_face_locations, detected_face_names):
    for (top, right, bottom, left), name in zip(
        detected_face_locations, detected_face_names
    ):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(
            frame,
            (left, bottom - 25),
            (right, bottom),
            (0, 0, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )


def _print_video_metrics(total_time, frame_counter, processing_times) -> None:
    fps = frame_counter / total_time if total_time > 0 else 0.0
    avg_processing_time = (
        sum(processing_times) / len(processing_times) if processing_times else 0.0
    )

    print("\n=== Performance Metrics ===")
    print(f"Total Runtime: {total_time:.2f} seconds")
    print(f"Total Frames: {frame_counter}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average Frame Processing Time: {avg_processing_time:.4f} seconds")


def recognize_video(
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    source=0,  # 0 = webcam, or pass video file path
    detection_interval: int = DEFAULT_DETECTION_INTERVAL,
    recognition_scale: float = DEFAULT_RECOGNITION_SCALE,
    tolerance: float = DEFAULT_TOLERANCE,
):
    if detection_interval < 1:
        raise ValueError("'detection_interval' must be >= 1.")
    if recognition_scale <= 0:
        raise ValueError("'recognition_scale' must be > 0.")

    loaded_encodings, model = load_encodings(encodings_location)
    video_capture = _open_video_capture(source)

    frame_count = 0
    detected_face_locations = []
    detected_face_names = []

    start_time = time.time()
    frame_counter = 0
    processing_times = []

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            frame_counter += 1
            if frame_count % detection_interval == 0:
                detected_face_locations, detected_face_names, processing_time = (
                    _detect_faces_in_frame(
                        frame=frame,
                        recognition_scale=recognition_scale,
                        model=model,
                        loaded_encodings=loaded_encodings,
                        tolerance=tolerance,
                    )
                )
                processing_times.append(processing_time)

            _draw_video_faces(frame, detected_face_locations, detected_face_names)

            cv2.imshow("Video Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    _print_video_metrics(total_time, frame_counter, processing_times)
