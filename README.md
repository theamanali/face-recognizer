# Face Recognizer

Face Recognizer is a Python command-line application for training and running
face recognition using `face_recognition`, `dlib`, and OpenCV.

## Capabilities

- Build encodings from labeled training images.
- Recognize faces in a single image.
- Batch-validate images from a validation directory.
- Run real-time webcam recognition with configurable performance controls.

## Repository Structure

- `detector.py`: executable entrypoint.
- `cli.py`: argument parsing and command dispatch.
- `face_store.py`: encoding generation and persistence.
- `face_matching.py`: face matching and vote-based label selection.
- `image_mode.py`: single-image and validation workflows.
- `video_mode.py`: webcam recognition workflow.
- `config.py`: shared constants and default values.

Data directories:

- `training/<person_name>/<image files>`
- `validation/`
- `output/encodings.pkl` (generated)

## Requirements

- Python 3.10 or newer.
- Native dependencies required by `dlib` (compiler/toolchain may be needed).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Add labeled images under `training/<person_name>/`.
2. Train encodings:

```bash
python detector.py --train --model hog
```

3. Run recognition:

```bash
python detector.py --image path/to/image.jpg --tolerance 0.5
```

## Command Reference

Show help:

```bash
python detector.py --help
```

Train:

```bash
python detector.py --train --model hog
```

Validate:

```bash
python detector.py --validate --tolerance 0.5
```

Webcam:

```bash
python detector.py --webcam --tolerance 0.5 --detection-interval 4 --recognition-scale 0.25
```

## Flag Constraints

- `--model` is valid only with `--train`.
- `--tolerance` is valid with `--image`, `--validate`, or `--webcam`.
- `--detection-interval` and `--recognition-scale` are valid only with `--webcam`.

Validation ranges when provided:

- `--tolerance`: `0 < value <= 1`
- `--detection-interval`: `value >= 1`
- `--recognition-scale`: `0 < value <= 1`

## Operational Notes

- Person labels are derived from folder names (`john_doe` becomes `John Doe`).
- If `output/encodings.pkl` is missing or corrupted, rerun training.
- Webcam mode requires camera permissions and an available video source.
