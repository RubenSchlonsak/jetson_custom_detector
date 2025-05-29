# deployment/inference_jetson.py
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Important for CUDA context initialization
import os
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "your_model.engine" # Should be in the same directory as this script in the Docker container

# --- ADJUST THESE PARAMETERS FOR YOUR CUSTOM MODEL ---
# This should match the input resolution your TensorRT engine was built for.
# For example, if your ONNX was exported with dummy_input_shape=(1, 3, 320, 320)
MODEL_INPUT_C, MODEL_INPUT_H, MODEL_INPUT_W = 3, 320, 320 # Example C, H, W

NUM_CLASSES_OUTPUT = 1 # Number of classes your model detects (e.g., 1 for 'face')
# Background class is often implicit or handled by model output structure.

# Detection thresholds
CONF_THRESHOLD = 0.5  # Confidence score to consider a detection valid
IOU_THRESHOLD = 0.4   # IOU for Non-Maximum Suppression

# Optional: Class names if your model outputs class IDs
# CLASS_NAMES = ["background", "face"] # If your model outputs class ID 1 for face
CLASS_NAMES = ["face"] # If model outputs ID 0 for face and no background class directly

class HostDeviceMem(object):
    # (Keep this class as provided before)
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self): return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self): return self.__str__()

def allocate_buffers(engine, batch_size=1):
    # (Keep this function largely as provided before, ensure batch_size handling)
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding_name in engine:
        binding_idx = engine.get_binding_index(binding_name)
        shape = list(engine.get_binding_shape(binding_idx))
        
        is_dynamic_shape = any(dim == -1 for dim in shape)
        
        # Try to resolve dynamic batch size
        if shape[0] == -1:
            shape[0] = batch_size
        
        # For other dynamic dimensions, this is tricky without knowing the model.
        # If a dimension is still -1, TensorRT needs an optimization profile
        # and context.set_binding_shape() to be called.
        # For this allocation, we need a concrete max size.
        if any(dim == -1 for dim in shape): # If still dynamic after setting batch
            if engine.binding_is_input(binding_idx):
                # For input, use the predefined MODEL_INPUT_ C,H,W if shape is fully dynamic
                # This assumes engine was built to support this max shape.
                current_shape = (batch_size, MODEL_INPUT_C, MODEL_INPUT_H, MODEL_INPUT_W)
                print(f"Warning: Input binding '{binding_name}' has dynamic dims other than batch. Using predefined shape {current_shape} for allocation.")
            else: # Output
                # This is the hardest part for dynamic output shapes.
                # You MUST know the maximum possible output size.
                # For object detection, it's often num_detections * (box_coords + conf + classes).
                # Set a large enough buffer, but this is risky.
                # Example: max 100 detections, 4 box coords, 1 conf, 1 class_id = 100 * 6 floats
                max_output_elements = 100 * (4 + 1 + NUM_CLASSES_OUTPUT) # Adjust this heuristic
                print(f"Warning: Output binding '{binding_name}' has dynamic dims. Allocating for ~{max_output_elements} elements. Adjust if incorrect.")
                # This assumes float32, adjust dtype logic if necessary
                shape = (batch_size, max_output_elements) # Simplified, actual shape might be more complex
        
        vol = trt.volume(shape)
        if vol == 0: # If volume is zero (e.g. shape was (1,0) for some reason)
            print(f"Warning: Calculated volume for binding '{binding_name}' is 0 with shape {shape}. This might be an issue.")
            # Allocate a minimal buffer to avoid errors, but this needs investigation.
            vol = 1 # Smallest possible non-zero
            
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        
        try:
            host_mem = cuda.pagelocked_empty(int(vol), dtype) # Ensure vol is int
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        except Exception as e:
            print(f"Error allocating memory for binding {binding_name} with vol {vol}, dtype {dtype}, shape {shape}: {e}")
            raise # Re-raise the exception

        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding_idx):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    return inputs, outputs, bindings, stream

def load_tensorrt_engine(engine_path):
    # (Keep this function as provided before)
    if not os.path.exists(engine_path):
        print(f"Error: TensorRT Engine file not found: {engine_path}")
        return None
    print(f"Loading TensorRT Engine from: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def preprocess_webcam_frame(frame, target_c, target_h, target_w):
    """
    Preprocesses a frame from the webcam for inference.
    This MUST match the preprocessing used during your model's training.
    """
    # Example: Resize, BGR to RGB, HWC to CHW, normalization
    img_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1] and then potentially subtract mean / divide by std if your model expects that
    # Example for [0,1] normalization:
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # If your model was trained with specific mean/std normalization (e.g., ImageNet stats):
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # img_normalized = (img_normalized - mean) / std
    
    # Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    img_chw = img_normalized.transpose((2, 0, 1))
    
    # Add batch dimension and ensure it's C-contiguous for TensorRT
    input_tensor = np.expand_dims(img_chw, axis=0) # Batch size 1
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
    
    return input_tensor

def postprocess_model_outputs(outputs_host_list, original_frame_shape_hw, model_input_shape_hw):
    """
    Postprocesses the raw output tensors from your TensorRT model.
    This is HIGHLY SPECIFIC to your model's architecture (e.g., SSD, YOLO, FasterRCNN).

    Args:
        outputs_host_list: A list of numpy arrays, each corresponding to an output tensor from the model.
        original_frame_shape_hw: Tuple (height, width) of the original webcam frame.
        model_input_shape_hw: Tuple (height, width) the model expects as input.

    Returns:
        A list of detections. Each detection is a dictionary:
        e.g., {"box": [x1, y1, x2, y2], "confidence": score, "class_id": id}
               where box coordinates are scaled to the original_frame_shape.
    """
    detections = []
    
    # --- YOU MUST IMPLEMENT THIS BASED ON YOUR MODEL ---
    # Example structure for a common object detector (like SSD or YOLO outputs after some parsing):
    # Let's assume your model (after potential ONNX graph modifications or internal parsing)
    # gives one primary output tensor that contains [N, K, 6] or [N, K, 7] where:
    # N = batch size (usually 1 here)
    # K = number of detected objects
    # 6 or 7 = [x1, y1, x2, y2, confidence, class_id] or [img_idx, x1, y1, x2, y2, conf, cls_id]

    # Placeholder: Assume outputs_host_list[0] is the main detection output
    # This shape and interpretation is purely hypothetical!
    # Shape might be (batch_size, num_detections, 6) where 6 = (x1,y1,x2,y2,score,class_id)
    # Or for some models, you might get separate tensors for boxes, scores, classes.

    if not outputs_host_list:
        return detections

    # Example: If your first output tensor has shape (1, num_dets, 6)
    # where 6 is [x1_norm, y1_norm, x2_norm, y2_norm, score, class_id]
    # and x1,y1,x2,y2 are normalized to model input dimensions (0-1 range)
    
    # This is a generic example; consult your model's output specification.
    # For torchvision FasterRCNN, the output is a list of dicts already.
    # If your ONNX/TRT model gives raw tensors, you need to parse them.

    # Let's assume `detection_output = outputs_host_list[0]` contains the detections.
    # And its shape is something like (1, MAX_DETECTIONS, 6)
    # where the 6 columns are [x_center, y_center, width, height, confidence, class_id]
    # (This is a common pattern for some YOLO versions *after* some initial decoding)

    # --- START OF HYPOTHETICAL POSTPROCESSING ---
    # This section is highly dependent on your model and needs to be replaced.
    # For a torchvision FasterRCNN model, if the TRT engine preserves its output structure,
    # it might be a list of dictionaries directly. More likely, it's raw tensors.

    # Let's assume outputs_host_list[0] contains bounding boxes (e.g., shape [num_boxes, 4])
    # outputs_host_list[1] contains scores (e.g., shape [num_boxes])
    # outputs_host_list[2] contains class IDs (e.g., shape [num_boxes])
    # (This is a common pattern for EfficientDet or some SSD post-NMS outputs)
    
    # Number of output tensors can vary. Check your model.
    if len(outputs_host_list) >= 3: # Example: boxes, scores, classes
        raw_boxes = outputs_host_list[0] # Shape: [num_detections, 4] e.g., [y1,x1,y2,x2] normalized 0-1
        raw_scores = outputs_host_list[1] # Shape: [num_detections]
        raw_classes = outputs_host_list[2] # Shape: [num_detections]
        # num_detections_tensor = outputs_host_list[3] # Some models output num_detections explicitly

        # num_actual_detections = int(num_detections_tensor[0]) if len(outputs_host_list) > 3 else raw_scores.shape[0]
        num_actual_detections = raw_scores.shape[0]


        original_h, original_w = original_frame_shape_hw
        # model_h, model_w = model_input_shape_hw # Not always needed if boxes are 0-1 normalized

        for i in range(num_actual_detections):
            score = raw_scores[i]
            if score >= CONF_THRESHOLD:
                class_id = int(raw_classes[i])
                
                # Assuming boxes are [y1, x1, y2, x2] and normalized to [0,1]
                # Adjust if your model outputs [x1,y1,x2,y2] or center_x,center_y,w,h
                y1_norm, x1_norm, y2_norm, x2_norm = raw_boxes[i]

                # Scale to original image dimensions
                x1 = int(x1_norm * original_w)
                y1 = int(y1_norm * original_h)
                x2 = int(x2_norm * original_w)
                y2 = int(y2_norm * original_h)
                
                # Ensure box coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_w - 1, x2)
                y2 = min(original_h - 1, y2)

                if x2 > x1 and y2 > y1: # Valid box
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": float(score),
                        "class_id": class_id
                    })
    else:
        print(f"Warning: Unexpected number of output tensors ({len(outputs_host_list)}). Implement postprocessing accordingly.")
        # Fallback: print raw output shapes for debugging
        for i, out_tensor in enumerate(outputs_host_list):
            print(f"  Output {i} shape: {out_tensor.shape}, dtype: {out_tensor.dtype}")


    # Apply Non-Maximum Suppression (NMS) if not already done by the model.
    # This placeholder assumes NMS is needed.
    if detections:
        boxes_for_nms = np.array([d["box"] for d in detections], dtype=np.float32)
        confidences_for_nms = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_ids_for_nms = np.array([d["class_id"] for d in detections]) # To perform NMS per class if needed

        final_detections = []
        # Perform NMS per class if your model detects multiple classes
        unique_class_ids = np.unique(class_ids_for_nms)
        for cid in unique_class_ids:
            class_indices = np.where(class_ids_for_nms == cid)[0]
            class_boxes = boxes_for_nms[class_indices]
            class_confidences = confidences_for_nms[class_indices]

            if len(class_boxes) > 0:
                # OpenCV's NMSBoxes returns indices of kept boxes *within the current class_boxes/class_confidences*
                kept_indices_for_class = cv2.dnn.NMSBoxes(class_boxes.tolist(), class_confidences.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
                if len(kept_indices_for_class) > 0:
                    # Convert these local indices back to original detection indices for this class
                    original_indices_for_kept_boxes = class_indices[kept_indices_for_class.flatten()]
                    for original_idx in original_indices_for_kept_boxes:
                        final_detections.append(detections[original_idx])
        detections = final_detections
    # --- END OF HYPOTHETICAL POSTPROCESSING ---
            
    return detections


def draw_bounding_boxes(frame, detections):
    """Draws bounding boxes and labels on the frame."""
    if not detections:
        return frame
        
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = map(int, box) # Ensure coordinates are integers
        
        confidence = det["confidence"]
        class_id = det["class_id"]
        
        label_text = f"{CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else 'Unknown'}: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
        
        # Put label above the box
        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10) # Ensure label is not cut off at top
        cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10), 
                               (x1 + label_size[0], y1_label - base_line -10), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label_text, (x1, y1_label - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text

    return frame

def main_inference():
    engine = load_tensorrt_engine(ENGINE_PATH)
    if not engine:
        print("Failed to load TensorRT engine. Exiting.")
        return

    # Setup GStreamer pipeline for CSI camera or use default for USB
    # (Same gstreamer_pipeline function as before if needed)
    # cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0) # Default to USB camera index 0

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create an execution context
    with engine.create_execution_context() as context:
        # If your model has dynamic input shapes, especially batch size,
        # you might need to set the binding shape for the context.
        # This assumes the engine was built with an optimization profile that supports this.
        # Example: if first binding (index 0) is input:
        # context.set_binding_shape(0, (1, MODEL_INPUT_C, MODEL_INPUT_H, MODEL_INPUT_W)) # Batch size 1
        
        inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size=1)

        print("Starting webcam inference... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting.")
                break
            
            original_frame_h, original_frame_w = frame.shape[:2]
            
            # Preprocess the frame
            input_tensor = preprocess_webcam_frame(frame, MODEL_INPUT_C, MODEL_INPUT_H, MODEL_INPUT_W)
            
            # Copy input data to pagelocked host memory and then to GPU
            np.copyto(inputs[0].host, input_tensor.ravel()) # Assuming first input buffer
            cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)

            # Run inference
            # For explicit batch engines, use execute_async_v2
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            # Copy output data from GPU to pagelocked host memory
            outputs_host_data = []
            for output_buffer in outputs:
                cuda.memcpy_dtoh_async(output_buffer.host, output_buffer.device, stream)
                outputs_host_data.append(output_buffer.host)
            stream.synchronize() # Wait for all GPU operations to complete

            # Postprocess the model outputs
            detections = postprocess_model_outputs(outputs_host_data,
                                                   (original_frame_h, original_frame_w),
                                                   (MODEL_INPUT_H, MODEL_INPUT_W))
            
            # Draw detections on the frame
            output_frame = draw_bounding_boxes(frame.copy(), detections) # Work on a copy

            # Display FPS (Optional)
            # (You can add FPS calculation here if needed)

            cv2.imshow("Jetson Custom Object Detector - Webcam", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == '__main__':
    # Ensure PyCUDA context is available if not using pycuda.autoinit implicitly
    # import pycuda.driver # If autoinit is not used or context needs manual management
    # pycuda.driver.init() # Initialize CUDA driver
    # ctx = pycuda.driver.Device(0).make_context() # Create context on device 0
    # try:
    main_inference()
    # finally:
    #     ctx.pop() # Detach context
    #     ctx.detach()