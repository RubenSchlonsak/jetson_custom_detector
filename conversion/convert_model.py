# conversion/convert_model.py
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn # Make sure this matches training
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

# Import class definitions or number of classes if needed
# from training.train_face_detector import NUM_CLASSES # Or define it directly
NUM_CLASSES = 2 # Background + Face

SOURCE_MODEL_PATH = "../models/source_model/face_detector.pth"
ONNX_MODEL_PATH = "../models/onnx_model/face_detector.onnx"
# Define an example input size (Batch, Channel, Height, Width)
# This should match what you expect during inference.
# For ONNX export, a fixed size or dynamic axes are needed.
# For Jetson, a fixed input size is often easier to start with for TensorRT.
EXAMPLE_INPUT_HEIGHT = 480 # Example, adjust if your training used a different size
EXAMPLE_INPUT_WIDTH = 640  # Example

def get_model_for_conversion(num_classes, model_path):
    # Load the same model architecture as used in training
    try: # New weights API
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, progress=False, num_classes=num_classes)
        # If you didn't use num_classes arg during init but replaced head, load pretrained weights and then replace head:
        # model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    except TypeError: # Fallback for older torchvision
        model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, progress=False, num_classes=num_classes)


    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}")
        # Load weights carefully if model was saved as state_dict
        # Ensure the model architecture here matches exactly how it was when state_dict was saved
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load to CPU for export
    else:
        print(f"Warning: Trained model weights not found at {model_path}. Exporting with un-fine-tuned or random weights.")
    
    model.eval() # Set model to evaluation mode
    return model

def main():
    print("Starting PyTorch to ONNX conversion...")
    os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)

    # Load the model
    model = get_model_for_conversion(NUM_CLASSES, SOURCE_MODEL_PATH)
    model.to(torch.device('cpu')) # Export on CPU

    # Create a dummy input tensor with the desired export resolution
    # Batch size is 1 for this example.
    dummy_input = torch.randn(1, 3, EXAMPLE_INPUT_HEIGHT, EXAMPLE_INPUT_WIDTH, device='cpu')
    print(f"Using dummy input of shape: {dummy_input.shape}")

    # Define input and output names for the ONNX graph (important for TensorRT)
    input_names = ["input_image"]
    # FasterRCNN typically outputs: boxes, labels, scores
    output_names = ["boxes", "labels", "scores"]

    print(f"Exporting model to ONNX: {ONNX_MODEL_PATH}")
    try:
        torch.onnx.export(model,
                          dummy_input,
                          ONNX_MODEL_PATH,
                          verbose=False, # Set to True for more details
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=11, # Or 12, 13, ... check compatibility with your TensorRT version
                          # dynamic_axes for variable input size / batch size (optional, can make TRT conversion harder)
                          # dynamic_axes={'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                          #               'boxes': {0: 'batch_size', 1: 'num_detections'},
                          #               'labels': {0: 'batch_size', 1: 'num_detections'},
                          #               'scores': {0: 'batch_size', 1: 'num_detections'}}
                         )
        print("ONNX export successful!")
        print(f"Model saved to {ONNX_MODEL_PATH}")
        print("\nNext steps:")
        print(f"1. Copy '{ONNX_MODEL_PATH}' to your Jetson Nano.")
        print("2. On the Jetson, convert ONNX to TensorRT engine using `trtexec`:")
        print(f"   trtexec --onnx={os.path.basename(ONNX_MODEL_PATH)} \\")
        print(f"           --saveEngine=../models/tensorrt_engine/face_detector.engine \\")
        print(f"           --explicitBatch --fp16 --workspace=1024 \\")
        print(f"           # Add --minShapes/optShapes/maxShapes if you used dynamic_axes and your model needs them.")
        print(f"           # Example for dynamic shape (replace 'input_image' with your actual input name if different):")
        print(f"           # --minShapes=input_image:1x3x{EXAMPLE_INPUT_HEIGHT//2}x{EXAMPLE_INPUT_WIDTH//2} \\")
        print(f"           # --optShapes=input_image:1x3x{EXAMPLE_INPUT_HEIGHT}x{EXAMPLE_INPUT_WIDTH} \\")
        print(f"           # --maxShapes=input_image:1x3x{EXAMPLE_INPUT_HEIGHT*2}x{EXAMPLE_INPUT_WIDTH*2}")


    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    main()