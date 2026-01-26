# Face Recognizer

A Python CLI for training and running face recognition with `face_recognition`, `dlib`, and `OpenCV`. 

![Webcam Demo](/media/demo.gif)
Demo run with:
- Recognition scale: 0.25
- Detection interval: 4
- Tolerance: 0.5

## Features

- Train encodings from labeled images (`training/<person_name>/...`)
- Validate model accuracy against a set of images in `validation/`
- Recognize faces in a single image
- Run video recognition with tunable performance controls

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
python detector.py --train
```

3. Run recognition:

```bash
python detector.py --image path/to/image.jpg
```

## CLI Interface

Show all flags:

```bash
python detector.py --help
```

### Main Modes
1. `--train` : Build encodings from labeled images in `training/<person_name>/` and write them to `output/encodings.pkl`.
   ```bash
   python detector.py --train
     ```
   - Optional flags:

     - `--model <hog|cnn>`: 
Face detection backend for encoding. Default is `hog`. `hog` is faster on CPU, `cnn` is typically more accurate but much slower. Recommended only on systems with dedicated GPU. 


2. `--image <path>`: Recognize faces in a single image.
    ```bash
    python detector.py --image path/to/image.jpg
    ```
   - Optional flags:
     - `--tolerance <float>`: 
     Face match threshold. Lower is stricter, higher is looser. Default is 0.5. Acceptable values: `0 < value <= 1`
     

3. `--validate`: Run recognition on every image in `validation/` (useful for quick checks).

    ```bash
    python detector.py --validate
    ```

   - Optional flags:
     - `--tolerance <float>` (same as `--image`)


4. `--webcam`: Run real-time recognition using your default camera.

    ```bash
    python detector.py --webcam
    ```

   - Optional flags:

     - `--tolerance <float>`
     - `--detection-interval <int>`: Detect faces every N frames and reuse locations in between. Higher values improve FPS but can miss fast movement. Default is 4. Acceptable values: `value >= 1`
     - `--recognition-scale <float>`: Downscale frames before recognition for speed. Lower values are faster but reduce accuracy on small or distant faces. Default is 0.25. Acceptable values: `0 < value <= 1`
     
## Operational Notes

- Person labels are derived from folder names (`john_doe` becomes `John Doe`).
- If `output/encodings.pkl` is missing or corrupted, rerun training.
- Webcam mode requires camera permissions and an available video source.
