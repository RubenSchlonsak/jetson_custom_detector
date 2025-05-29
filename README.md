# Jetson Custom Face Detector (Hackathon Project)

This project trains a custom face detection model using PyTorch, converts it for optimized inference with TensorRT, and deploys it on an NVIDIA Jetson Nano using Docker. It uses the "Face Detection in Images" dataset from Kaggle.

## Project Structure

-   **`data/`**: Holds datasets.
    -   `face_detection_dataset/`: Downloaded "Face Detection in Images" dataset.
-   **`training/`**: Scripts for model training.
    -   `train_face_detector.py`: Main training script.
    -   `requirements_train.txt`: Python dependencies for training.
    -   `dataset_utils.py`: (New) Utilities for downloading and preparing the dataset.
-   **`models/`**: Storage for trained models.
    -   `source_model/face_detector.pth`: Trained PyTorch model.
    -   `onnx_model/face_detector.onnx`: Model in ONNX format.
    -   `tensorrt_engine/face_detector.engine`: Compiled TensorRT engine.
-   **`conversion/`**: Scripts for model conversion.
    -   `convert_model.py`: Script to convert PyTorch model to ONNX.
-   **`deployment/`**: Files for Jetson Nano.
    -   `Dockerfile`: For the Jetson inference environment.
    -   `inference_jetson.py`: Inference script using TensorRT.
    -   `requirements_jetson.txt`: Python dependencies for Jetson.
    -   `face_detector.engine`: Copy your converted engine here for Docker build.
-   **`utils/`**: (Optional) General utility scripts.

## Workflow

1.  **Setup Training Environment (PC/Cloud with GPU Recommended)**:
    * Clone this repository.
    * Create a Python virtual environment:
        ```bash
        python3 -m venv env
        source env/bin/activate
        pip install -r training/requirements_train.txt
        ```
    * **Kaggle API Setup (for dataset download)**:
        * Install Kaggle API: `pip install kaggle`
        * Go to your Kaggle account, click on your profile picture -> "Account" -> "API" section -> "Create New API Token". This will download `kaggle.json`.
        * Place `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<Windows-User>\.kaggle\kaggle.json` (Windows). Make sure it's readable only by you (`chmod 600 ~/.kaggle/kaggle.json`).

2.  **Data Preparation**:
    * Run the dataset utility script to download and prepare the data:
        ```bash
        cd training
        python dataset_utils.py
        cd ..
        ```
        This will download the "Face Detection in Images" dataset to `data/face_detection_dataset/`.

3.  **Model Training**:
    * Run the training script (ideally on a machine with a GPU):
        ```bash
        python training/train_face_detector.py
        ```
    * The trained model will be saved to `models/source_model/face_detector.pth`.

4.  **Model Conversion**:
    * Convert the trained PyTorch model to ONNX format:
        ```bash
        python conversion/convert_model.py
        ```
        The ONNX model will be saved to `models/onnx_model/face_detector.onnx`.
    * **Convert ONNX to TensorRT Engine (on Jetson Nano)**:
        * Copy the `models/onnx_model/face_detector.onnx` file to your Jetson Nano.
        * On the Jetson, use `trtexec` (NVIDIA's command-line tool for TensorRT) to create the engine. Example:
            ```bash
            trtexec --onnx=face_detector.onnx \
                    --saveEngine=../models/tensorrt_engine/face_detector.engine \
                    --explicitBatch \
                    --fp16 \
                    --workspace=1024 
                    # Add --minShapes, --optShapes, --maxShapes if your ONNX has dynamic input shapes
                    # e.g., --minShapes=input:1x3x224x224 --optShapes=input:1x3x480x640 --maxShapes=input:1x3x720x1280
                    # Replace 'input' with your actual model input name
            ```
        * This command assumes your ONNX model has a fixed batch size or you're using explicit batch. Adjust `fp16` (for 16-bit floating point precision) or use `int8` (requires a calibration process) as needed for performance. Adjust workspace size.
        * Place the generated `face_detector.engine` into the `jetson_face_detector_custom_en/models/tensorrt_engine/` directory and also copy it to `jetson_face_detector_custom_en/deployment/face_detector.engine` for the Docker build.

5.  **Deployment on Jetson Nano**:
    * Ensure `deployment/face_detector.engine` exists.
    * Navigate to the `deployment/` directory on your Jetson: `cd deployment/`
    * Build the Docker image: `docker build -t custom-face-detector .`
    * Run the Docker container:
        ```bash
        # Allow GUI display from Docker
        xhost +si:localuser:root

        docker run -it --rm --runtime nvidia \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix/:/tmp/.X11-unix \
            --device /dev/video0 \
            custom-face-detector
        ```

## Notes
* The `INPUT_SIZE` in `train_face_detector.py` and `inference_jetson.py` should be consistent with how your model was trained and how the ONNX model expects input.
* TensorRT conversion can be complex. `trtexec` is generally robust. Ensure your Jetson has enough memory for the conversion and inference.
* For the hackathon, pre-training the model and pre-converting to TensorRT will save a lot of time.