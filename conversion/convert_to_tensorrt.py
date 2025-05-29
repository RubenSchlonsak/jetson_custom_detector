# conversion/convert_to_tensorrt.py
import torch
import os
# Ensure you have 'tensorrt' and 'pycuda' for full Python API conversion,
# or use 'trtexec' command-line tool on the Jetson.

# Path to the PyTorch model saved after training
SOURCE_MODEL_PATH = "../models/source_model/face_detector_model.pth"
ONNX_MODEL_PATH = "../models/onnx_model/face_detector_model.onnx"
TENSORRT_ENGINE_PATH = "../models/tensorrt_engine/face_detector_model.engine" # Or move to deployment/
# For TensorRT conversion, you also need to know the number of classes your model was trained for
NUM_CLASSES_FOR_MODEL = 1 + 1 # Example: 1 face class + background

# Dummy function for loading your specific model architecture
# This needs to match the architecture used in train_model.py
def load_trained_pytorch_model(model_path, num_classes):
    print(f"Loading PyTorch model from: {model_path}")
    # Re-instantiate the model structure
    # Example using the same function from training script (if it's structured that way)
    # from training.train_model import get_object_detection_model # This might cause issues if train_model.py has side effects on import
    
    # Safer: Redefine or import the model architecture directly here
    # For this example, let's assume get_object_detection_model is safe to import or redefine
    try:
        # --- Re-define or import your model architecture ---
        # This should be the same architecture used during training.
        # Example for torchvision's FasterRCNN with MobileNetV3
        import torchvision
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None) # Load structure only
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        # --- End of model architecture re-definition ---

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load weights
        model.eval() # Set to evaluation mode
        print("PyTorch model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

def export_to_onnx(pytorch_model, onnx_output_path, dummy_input_shape=(1, 3, 320, 320)):
    # Ensure the dummy_input_shape matches what your model expects (e.g., 320x320 for mobilenetv3_large_320_fpn)
    # Some models might have different input sizes (e.g. 416x416 for some YOLOs)
    # The size should be consistent with training preprocessing if possible, or a standard size for the backbone.
    
    print(f"Exporting PyTorch model to ONNX: {onnx_output_path}")
    if pytorch_model is None:
        print("PyTorch model is None, cannot export.")
        return False

    try:
        device = torch.device("cpu") # ONNX export is often done on CPU
        pytorch_model.to(device)
        dummy_input = torch.randn(dummy_input_shape, device=device)
        
        # Define input and output names for the ONNX model.
        # These can be important for some conversion tools or inference runtimes.
        # For object detection models from torchvision, output names are often not explicitly needed for ONNX export
        # but can be useful. The actual output structure depends on the model.
        # A common FasterRCNN output might be boxes, labels, scores.
        # Check your model's documentation or behavior.
        input_names = ["input_image"]
        # For a model like torchvision's FasterRCNN, it returns a list of dicts in eval mode.
        # For ONNX export, it might behave differently or you might need to wrap it.
        # Often, ONNX export works best if the model's forward() returns tensors directly.
        # It's simpler to let PyTorch infer output names or not specify them if not strictly needed for trtexec.
        
        torch.onnx.export(pytorch_model,
                          dummy_input,
                          onnx_output_path,
                          verbose=False,
                          input_names=input_names,
                          # output_names=['boxes', 'labels', 'scores'], # Optional, check if needed/correct
                          opset_version=11, # Common opset, check compatibility
                          dynamic_axes={'input_image': {0: 'batch_size'}, # If you want dynamic batch size
                                        # For outputs, if they exist and are named:
                                        # 'boxes': {0: 'batch_size'},
                                        # 'scores': {0: 'batch_size'},
                                        # 'labels': {0: 'batch_size'}
                                       } if dummy_input_shape[0] == 1 else None # Only if batch size is 1 in dummy
                         )
        print(f"ONNX model saved to {onnx_output_path}")
        return True
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return False

def convert_onnx_to_tensorrt_trtexec(onnx_path, engine_path, precision="fp16"):
    print(f"Converting ONNX ({onnx_path}) to TensorRT Engine ({engine_path}) using trtexec...")
    print("This command should be run on the Jetson Nano or a compatible environment.")
    # Workspace size might need adjustment based on model
    # For newer TensorRT versions, --explicitBatch is often default or handled by ONNX exporter flags
    command = f"trtexec --onnx={onnx_path} --saveEngine={engine_path}"
    if precision == "fp16":
        command += " --fp16"
    elif precision == "int8":
        # INT8 requires a calibration dataloader and a calibration cache. This is more complex.
        # command += " --int8 --calib=<path_to_calibration_cache_or_first_few_batches_of_data>"
        print("INT8 precision selected. This requires a calibration process not fully implemented in this placeholder.")
        print("For INT8, you'd typically need to provide a calibration dataset or cache to trtexec.")
    
    print("\nRun the following command on your Jetson (or compatible environment with TensorRT):")
    print("------------------------------------------------------------------------------------")
    print(command)
    print("------------------------------------------------------------------------------------")
    print("\nIf successful, this will create the .engine file.")
    # This script won't execute trtexec itself, just print the command.
    # You'd typically run trtexec in the Jetson's terminal.

if __name__ == '__main__':
    print("--- Model Conversion Pipeline ---")

    # Ensure output directories exist
    if not os.path.exists(os.path.dirname(ONNX_MODEL_PATH)):
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH))
    if not os.path.exists(os.path.dirname(TENSORRT_ENGINE_PATH)):
        os.makedirs(os.path.dirname(TENSORRT_ENGINE_PATH))

    # 1. Load the trained PyTorch model
    if not os.path.exists(SOURCE_MODEL_PATH):
        print(f"Error: Trained PyTorch model not found at {SOURCE_MODEL_PATH}")
        print("Please train the model first using 'train_model.py' or place a trained .pth file there.")
    else:
        pytorch_model_loaded = load_trained_pytorch_model(SOURCE_MODEL_PATH, NUM_CLASSES_FOR_MODEL)

        if pytorch_model_loaded:
            # 2. Export to ONNX
            # Define the expected input shape for ONNX export. This should match what the model expects.
            # For fasterrcnn_mobilenet_v3_large_320_fpn, input images are resized to min_size=320.
            # The actual input tensor shape to the backbone might vary.
            # Let's assume an input that works for many common models: (Batch, Channels, Height, Width)
            # Common input for models like MobileNet based detectors might be around 300x300 or 320x320.
            # Check your specific model's documentation.
            # For the torchvision fasterrcnn_mobilenet_v3_large_320_fpn, it internally handles various input sizes,
            # but for ONNX export, a fixed representative size is good.
            # The '320' in the name suggests an internal processing size.
            # The model can take images of different sizes and resizes them.
            # A common test size:
            dummy_input_h, dummy_input_w = 320, 320 # Or match your training image size if fixed
            onnx_export_success = export_to_onnx(
                pytorch_model_loaded,
                ONNX_MODEL_PATH,
                dummy_input_shape=(1, 3, dummy_input_h, dummy_input_w)
            )

            if onnx_export_success and os.path.exists(ONNX_MODEL_PATH):
                # 3. Convert ONNX to TensorRT (print command for trtexec)
                # Precision can be "fp32", "fp16", "int8"
                convert_onnx_to_tensorrt_trtexec(ONNX_MODEL_PATH, TENSORRT_ENGINE_PATH, precision="fp16")
            else:
                print("ONNX export failed or ONNX file not found. Skipping TensorRT conversion command.")
        else:
            print("Failed to load PyTorch model. Aborting conversion.")