# training/train_model.py
import torch
import torchvision # For models
from torch.utils.data import DataLoader

# Import from your dataset utility script
from dataset_utils import FaceDataset, get_transform, collate_fn # Make sure this is correct

# --- Configurations (Examples) ---
# Path to your processed annotation files
TRAIN_ANNOTATIONS = "../data/processed_data/train_annotations.csv" # You need to create this
VAL_ANNOTATIONS = "../data/processed_data/val_annotations.csv"   # You need to create this
# Base directory where images (e.g., WIDER_train/images, WIDER_val/images) are located
# The paths in your annotation CSVs should be relative to these base directories or be absolute.
# If your CSVs store paths like "0--Parade/image.jpg", then IMG_DIR_TRAIN should be "../data/wider_face/WIDER_train/images/"
IMG_DIR_TRAIN = "../data/wider_face/WIDER_train/images/" # Adjust to your WIDER FACE image path
IMG_DIR_VAL = "../data/wider_face/WIDER_val/images/"     # Adjust to your WIDER FACE image path


MODEL_OUTPUT_PATH = "../models/source_model/face_detector_model.pth"
NUM_CLASSES = 1 + 1  # 1 class (face) + 1 background class for many object detection models
NUM_EPOCHS = 25      # Reduce for faster hackathon iterations, increase for better results
BATCH_SIZE = 4       # Adjust based on your GPU memory (Jetson Nano might require smaller)
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

def get_object_detection_model(num_classes):
    print("Defining object detection model...")
    # Example: Faster R-CNN with MobileNetV3-Large FPN backbone (pre-trained on COCO)
    # This is a good balance for performance on edge devices if fine-tuned.
    try:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT # Recommended for consistency
        )
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one (for your number of classes)
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    except Exception as e:
        print(f"Could not load pre-trained FasterRCNN with MobileNetV3: {e}")
        print("Falling back to a simpler SSD or ensure torchvision version is compatible.")
        # Fallback or alternative: SSDlite with MobileNetV3 Small (if available and torchvision supports it easily)
        # Or you might need to implement a simpler model or use a YOLO library.
        # For simplicity, this example relies on torchvision's offerings.
        # Consider `torchvision.models.detection.ssd300_vgg16(weights=...)` and modify head,
        # or `torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=...)`
        # For a hackathon, a pre-built YOLOv5/YOLOv8 script might be faster to get running if torchvision is tricky.
        # This example sticks to torchvision for now.
        # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        # from torchvision.models.detection.ssdlite import SSDLiteHead
        # num_anchors = model.anchor_generator.num_anchors_per_location()
        # model.head = SSDLiteHead(in_channels=..., num_anchors=num_anchors, num_classes=num_classes) # in_channels need to be found
        return None # Indicate model loading failure

    print(f"Using FasterRCNN with MobileNetV3-Large FPN, configured for {num_classes} classes.")
    return model

def main_training_loop():
    print("--- Starting Model Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare Datasets and DataLoaders
    print("Loading datasets...")
    try:
        dataset_train = FaceDataset(
            annotations_file=TRAIN_ANNOTATIONS,
            img_dir=IMG_DIR_TRAIN,
            transform=get_transform(train=True)
        )
        dataset_val = FaceDataset(
            annotations_file=VAL_ANNOTATIONS,
            img_dir=IMG_DIR_VAL,
            transform=get_transform(train=False)
        )

        # Ensure datasets are not empty
        if len(dataset_train) == 0 or len(dataset_val) == 0:
            print("Error: One or both datasets are empty. Please check annotation files and image paths.")
            print(f"TRAIN_ANNOTATIONS: {TRAIN_ANNOTATIONS} (exists: {os.path.exists(TRAIN_ANNOTATIONS)})")
            print(f"VAL_ANNOTATIONS: {VAL_ANNOTATIONS} (exists: {os.path.exists(VAL_ANNOTATIONS)})")
            print(f"IMG_DIR_TRAIN: {IMG_DIR_TRAIN} (exists: {os.path.exists(IMG_DIR_TRAIN)})")
            print(f"IMG_DIR_VAL: {IMG_DIR_VAL} (exists: {os.path.exists(IMG_DIR_VAL)})")
            return

        print(f"Training samples: {len(dataset_train)}, Validation samples: {len(dataset_val)}")

        # DataLoaders
        # Note: For training on Jetson Nano, num_workers > 0 can cause issues. Start with 0.
        # On a PC for training, you can increase num_workers (e.g., 2 or 4).
        data_loader_train = DataLoader(
            dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn
        )
        data_loader_val = DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn # Batch size 1 for val
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Make sure your annotation CSVs (train_annotations.csv, val_annotations.csv) exist in data/processed_data/")
        print("and that dataset_utils.py can parse them correctly. Also check image paths.")
        return

    # Get the model
    model = get_object_detection_model(NUM_CLASSES)
    if model is None:
        print("Failed to initialize model. Exiting training.")
        return
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler (optional, but good practice)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader_train):
            if images is None or targets is None: # Handle cases where batch is None from collate_fn
                print(f"Skipping batch {i} due to loading error.")
                continue

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets) # In training, model returns a dict of losses
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            if (i+1) % 10 == 0: # Print status every 10 batches
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(data_loader_train)}], Loss: {losses.item():.4f}")

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {epoch_loss/len(data_loader_train):.4f}")

        # Validation (simplified, proper evaluation would use mAP)
        model.eval() # Set model to evaluation mode
        val_loss_total = 0
        with torch.no_grad():
            for images_val, targets_val in data_loader_val:
                if images_val is None or targets_val is None: continue
                images_val = list(img.to(device) for img in images_val)
                targets_val = [{k: v.to(device) for k, v in t.items()} for t in targets_val]
                
                # Note: During evaluation, some models might not return losses directly when targets are provided.
                # For a simple validation loss check during training, we can still pass targets.
                # For proper mAP, you'd get predictions and compare with ground truth separately.
                # This behavior can vary between torchvision versions / models.
                # If it errors here, it means model(images, targets) is not intended for loss calculation in eval mode.
                # In that case, you'd do: predictions = model(images_val) and then compare predictions to targets_val.
                try:
                    loss_dict_val = model(images_val, targets_val)
                    losses_val = sum(loss for loss in loss_dict_val.values())
                    val_loss_total += losses_val.item()
                except Exception as e_val: # Catch if model doesn't return loss in eval with targets
                    print(f"Note: Could not get validation loss directly from model: {e_val}")
                    print("Skipping validation loss for this epoch. Consider implementing mAP or separate prediction eval.")
                    val_loss_total = -1 # Indicate no loss calculated
                    break 

        if val_loss_total != -1 and len(data_loader_val) > 0:
             print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {val_loss_total/len(data_loader_val):.4f}")
        elif len(data_loader_val) == 0:
            print("Validation data loader is empty.")


        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save checkpoint (optional)
        # torch.save(model.state_dict(), f"../models/source_model/face_detector_epoch_{epoch+1}.pth")

    print("Training finished.")
    print(f"Saving final model to: {MODEL_OUTPUT_PATH}")
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print("Model saved.")

if __name__ == '__main__':
    # Crucial: Create your processed annotation files (train_annotations.csv, val_annotations.csv)
    # in data/processed_data/ before running this script.
    # These files should list image paths and their bounding box coordinates.
    # The `dataset_utils.py` placeholder shows an example of how to create a dummy one.
    if not (os.path.exists(TRAIN_ANNOTATIONS) and os.path.exists(VAL_ANNOTATIONS)):
        print("Error: Processed annotation files not found!")
        print(f"Please ensure '{TRAIN_ANNOTATIONS}' and '{VAL_ANNOTATIONS}' exist.")
        print("You might need to run a script to parse your downloaded dataset (e.g., WIDER FACE .mat files)")
        print("and create these CSVs with columns like: image_path,x1,y1,x2,y2,label_name")
        print("Refer to `dataset_utils.py` for a conceptual structure or to create dummy files for testing.")
    else:
        main_training_loop()