from collections import Counter

import face_recognition

from config import DEFAULT_TOLERANCE


def recognize_face(
    unknown_encoding,
    loaded_encodings,
    tolerance: float = DEFAULT_TOLERANCE,
):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"],
        unknown_encoding,
        tolerance=tolerance,
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )

    if votes:
        return votes.most_common(1)[0][0]
    return None
