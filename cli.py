import argparse

from config import (
    DEFAULT_DETECTION_INTERVAL,
    DEFAULT_RECOGNITION_SCALE,
    DEFAULT_TOLERANCE,
    MODEL_NAME,
)
from face_store import encode_known_faces, ensure_directories
from image_mode import recognize_faces, validate
from video_mode import recognize_video


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
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=f"Face match tolerance for recognition (default: {DEFAULT_TOLERANCE}).",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=None,
        help=(
            "Run recognition every N frames in webcam mode "
            f"(default: {DEFAULT_DETECTION_INTERVAL})."
        ),
    )
    parser.add_argument(
        "--recognition-scale",
        type=float,
        default=None,
        help=(
            "Scale factor for webcam recognition resolution "
            f"(default: {DEFAULT_RECOGNITION_SCALE})."
        ),
    )
    args = parser.parse_args()

    if args.model and not args.train:
        parser.error("--model can only be used with --train.")

    if args.train:
        if args.tolerance is not None:
            parser.error("--tolerance can only be used with --image, --validate, or --webcam.")
        if args.detection_interval is not None:
            parser.error("--detection-interval can only be used with --webcam.")
        if args.recognition_scale is not None:
            parser.error("--recognition-scale can only be used with --webcam.")

    if args.image or args.validate:
        if args.detection_interval is not None:
            parser.error("--detection-interval can only be used with --webcam.")
        if args.recognition_scale is not None:
            parser.error("--recognition-scale can only be used with --webcam.")

    tolerance = args.tolerance if args.tolerance is not None else DEFAULT_TOLERANCE
    detection_interval = (
        args.detection_interval
        if args.detection_interval is not None
        else DEFAULT_DETECTION_INTERVAL
    )
    recognition_scale = (
        args.recognition_scale
        if args.recognition_scale is not None
        else DEFAULT_RECOGNITION_SCALE
    )

    if args.tolerance is not None and not (0 < tolerance <= 1):
        parser.error("--tolerance must be > 0 and <= 1.")
    if args.detection_interval is not None and detection_interval < 1:
        parser.error("--detection-interval must be >= 1.")
    if args.recognition_scale is not None and not (0 < recognition_scale <= 1):
        parser.error("--recognition-scale must be > 0 and <= 1.")

    ensure_directories()

    if args.train:
        encode_known_faces(model=args.model or MODEL_NAME)
    if args.validate:
        validate(tolerance=tolerance)
    if args.image:
        recognize_faces(image_location=args.image, tolerance=tolerance)
    if args.webcam:
        recognize_video(
            detection_interval=detection_interval,
            recognition_scale=recognition_scale,
            tolerance=tolerance,
        )
