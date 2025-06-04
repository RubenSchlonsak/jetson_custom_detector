# Jetson Custom Face Detector

This repository contains example scripts for training a face detection model with PyTorch and deploying it on an NVIDIA Jetson device.  The workflow covers training, converting the model to ONNX/TensorRT and running inference inside a Docker container.

## Project layout

- **`training/`** – utilities and scripts for training
  - `training_face_detector.py` – example training script using `torchvision`
  - `dataset_utils.py` – helper functions for preparing annotation files
  - `requirements.txt` – Python dependencies for the training environment
- **`conversion/`** – convert a trained model
  - `convert_model.py` – export the PyTorch model to ONNX
  - `convert_to_tensorrt.py` – helper for creating a TensorRT engine
- **`deployment/`** – files used on the Jetson
  - `Dockerfile` – minimal container with TensorRT and OpenCV
  - `inference_jetson.py` – sample inference application
  - `requirements_jetson.txt` – additional packages installed in the container

Two directories are created during the workflow:

- `data/` – dataset location (e.g. the WIDER FACE dataset)
- `models/` – checkpoints, ONNX files and TensorRT engines

## Usage

1. **Set up the training environment**
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r training/requirements.txt
   ```

2. **Prepare the dataset**
   Download your face detection dataset (for example WIDER FACE) and create annotation CSV files expected by `dataset_utils.py`. Place the images and CSVs inside the `data/` folder.

3. **Train the model**
   ```bash
   python training/training_face_detector.py
   ```
   A PyTorch checkpoint is saved under `models/source_model/`.

4. **Convert to ONNX**
   ```bash
   python conversion/convert_model.py
   ```
   This creates `models/onnx_model/face_detector_model.onnx`.

5. **Build a TensorRT engine**
   On the Jetson you can run `convert_to_tensorrt.py` or use `trtexec` manually to produce `models/tensorrt_engine/face_detector_model.engine`.

6. **Run on the Jetson**
   Copy the TensorRT engine to the `deployment/` folder and build the Docker image:
   ```bash
   cd deployment
   docker build -t custom-face-detector .
   docker run -it --rm --runtime nvidia \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix/:/tmp/.X11-unix \
       --device /dev/video0 \
       custom-face-detector
   ```

## Notes
- Ensure the input size used during training matches the one specified in `inference_jetson.py`.
- TensorRT conversion requires enough memory on the Jetson; using FP16 can reduce the footprint.
