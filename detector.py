from cli import main
from face_store import encode_known_faces
from image_mode import recognize_faces, validate
from video_mode import recognize_video

__all__ = [
    "encode_known_faces",
    "recognize_faces",
    "validate",
    "recognize_video",
    "main",
]

if __name__ == "__main__":
    main()
