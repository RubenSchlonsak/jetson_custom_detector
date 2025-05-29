# Jetson Custom Detector (Hackathon Project)

This project aims to train a custom object detection model and deploy it on an NVIDIA Jetson Nano using Docker and TensorRT.

## Project Structure

- **`data/`**: Contains datasets. Large data files should not be versioned directly in Git. Create e.g., `data/custom_dataset/images` and `data/custom_dataset/annotations` here.
- **`training/`**: Scripts for model training (e.g., `train_model.py`). This training should be performed on a powerful machine with a GPU.
- **`models/`**: Storage for trained models (`source_model/`), ONNX versions (`onnx_model/`), and TensorRT engines (`tensorrt_engine/`). Large files here should also not be versioned directly in Git.
- **`conversion/`**: Scripts for model conversion (e.g., PyTorch/TensorFlow -> ONNX -> TensorRT).
- **`deployment/`**: Contains the `Dockerfile` and the inference script (`inference_jetson.py`) for the Jetson Nano. The `your_model.engine` should be placed here if it's to be copied into the Docker image.
- **`utils/`**: General utility scripts.

## Workflow (Overview)

1.  **Data Preparation**:
    * Collect and annotate your image data in the `data/custom_dataset/` directory.
2.  **Model Training**:
    * Set up a training environment on a PC/Cloud server with a GPU.
    * Install dependencies from `training/requirements_train.txt`.
    * Adapt `training/train_model.py` to your chosen model (e.g., YOLO, SSD with a MobileNet backbone via PyTorch/TensorFlow) and your data, then train it.
    * Save the trained model to `models/source_model/`.
3.  **Model Conversion**:
    * Export your trained model to ONNX format (to `models/onnx_model/`).
    * Use scripts from `conversion/` (or `trtexec` directly on the Jetson) to convert the ONNX model into a TensorRT engine. Save the engine as `your_model.engine` (e.g., in `models/tensorrt_engine/` or directly in `deployment/`).
4.  **Deployment on Jetson Nano**:
    * Copy `your_model.engine` (if not already there) into the `deployment/` directory on your Jetson Nano.
    * Ensure that `deployment/inference_jetson.py` correctly loads and uses the engine.
    * Navigate to the `deployment/` directory: `cd deployment/`
    * Build the Docker image: `docker build -t custom-detector .`
    * Run the Docker container (allow access to camera and X server):
        ```bash
        xhost +si:localuser:root # Adjust if running Docker without root
        docker run -it --rm --runtime nvidia \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix/:/tmp/.X11-unix \
            --device /dev/video0 \
            # Optional: If the engine was not copied into the image, mount it:
            # -v $(pwd)/your_model.engine:/app/your_model.engine \
            custom-detector
        ```

## Setup

### Training Environment (PC/Cloud)

```bash
# Clone the repository (after creating it on GitHub etc.)
# git clone <your-repo-url>
# cd jetson_custom_detector
python3 -m venv env
source env/bin/activate
pip install -r training/requirements_train.txt
