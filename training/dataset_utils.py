# training/dataset_utils.py
import os
import pandas as pd # Example if using CSV annotations
from PIL import Image # For loading images
import torch # Assuming PyTorch
from torch.utils.data import Dataset
# Add imports for torchvision.transforms, etc.

class FaceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
                                     Example format: image_name,x1,y1,x2,y2,label
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        # Ensure you have a consistent mapping from class names to integer labels if needed
        # self.class_to_idx = {"face": 0} # Example for single class detection

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert("RGB") # Ensure RGB
        except FileNotFoundError:
            print(f"Warning: Image not found {img_name}, skipping item or returning None.")
            # You might want to handle this more robustly, e.g., by returning a placeholder
            # or ensuring your annotations file is perfectly clean.
            return None # Or raise an error, or return a dummy sample

        # Get bounding boxes and labels for the image
        # This part is highly dependent on your annotations_file format
        # Assuming one face per row for simplicity here. If multiple faces per image,
        # you'll need to group annotations by image_name.
        
        boxes = []
        labels = []
        
        # Example: if your CSV has one row per bounding box
        # This assumes your CSV is structured such that you can easily get all boxes for `img_name`
        # For a more robust solution, you might pre-process annotations to group boxes per image.
        image_annotations = self.img_labels[self.img_labels.iloc[:, 0] == self.img_labels.iloc[idx, 0]]
        
        for _, row in image_annotations.iterrows():
            # Assuming columns x1, y1, x2, y2, and a label column
            # Make sure these column names/indices match your CSV
            box = [row['x1'], row['y1'], row['x2'], row['y2']] # Or row.iloc[1:5]
            boxes.append(box)
            # Assuming 'face' is the only class, label is 0
            # If you have multiple classes, map them to integers
            labels.append(0) # Example: self.class_to_idx[row['label']]

        # Convert to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # You might also need area, iscrowd, image_id for some models like FasterRCNN
        # target["image_id"] = torch.tensor([idx])
        # target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)


        if self.transform:
            # Note: Transforms for object detection need to handle both image and bounding boxes.
            # Standard torchvision transforms might only work on the image.
            # You might need libraries like 'albumentations' or write custom transform logic.
            image, target = self.transform(image, target) # transform needs to accept (image, target)

        return image, target

def get_transform(train):
    """
    Define transforms. For object detection, transforms need to adjust bounding boxes too.
    Libraries like Albumentations are great for this.
    This is a very basic placeholder.
    """
    import torchvision.transforms.v2 as T # Use v2 for object detection transforms

    transforms = []
    transforms.append(T.ToImageTensor()) # Convert PIL image to tensor (better than ToTensor for PIL)
    transforms.append(T.ConvertImageDtype(torch.float30)) # Normalize to [0,1]

    if train:
        # Add augmentation for training, e.g., horizontal flip
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Add more augmentations: RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop etc.

    # Note: Bounding box adjustments are handled by T.Compose if transforms support it.
    # If not using v2 transforms that handle targets, you'll need custom logic or Albumentations.
    return T.Compose(transforms)


def collate_fn(batch):
    """
    Collate function for the DataLoader, especially if images in a batch can have
    different numbers of objects. This function handles padding or list collation.
    Returns a tuple of images and targets.
    """
    # Filter out None items that may have resulted from missing images in __getitem__
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: # If all items were None
        return None, None 
    return tuple(zip(*batch))

# Example usage (you'd call this from train_model.py)
if __name__ == '__main__':
    # These paths need to point to your actual processed annotation files and image directories
    # You would create these processed_train_annotations.csv from the WIDERFACE .mat files or other dataset format.
    # For WIDERFACE, the images are in subfolders, so img_dir logic might need adjustment or
    # your CSV should contain the full relative path to each image.
    
    # Placeholder: Create dummy annotation files for testing the Dataset class structure
    dummy_train_ann_path = "../data/processed_data/dummy_train_annotations.csv"
    dummy_val_ann_path = "../data/processed_data/dummy_val_annotations.csv"
    dummy_img_dir = "../data/wider_face/WIDER_train/images/0--Parade/" # Point to a dir with a few images

    if not os.path.exists("../data/processed_data"):
        os.makedirs("../data/processed_data")

    # Create dummy CSV if it doesn't exist
    if not os.path.exists(dummy_train_ann_path) and os.path.exists(dummy_img_dir):
        print(f"Creating dummy annotation file at {dummy_train_ann_path}")
        # List a few images from the dummy_img_dir
        sample_images = [f for f in os.listdir(dummy_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:2]
        if sample_images:
            dummy_data = []
            for img_file in sample_images:
                # Create dummy bounding boxes for these images
                # Format: image_filename,x1,y1,x2,y2,label_name (or just class_id if using map)
                # These are completely made up box coordinates
                dummy_data.append([os.path.join("0--Parade", img_file), 50, 50, 150, 150, "face"])
                dummy_data.append([os.path.join("0--Parade", img_file), 160, 60, 250, 180, "face"]) # Second box for same image
            
            dummy_df = pd.DataFrame(dummy_data, columns=["image_path", "x1", "y1", "x2", "y2", "label"])
            dummy_df.to_csv(dummy_train_ann_path, index=False)
            print(f"Dummy annotation file created with {len(sample_images)} images.")
        else:
            print(f"No images found in {dummy_img_dir} to create dummy annotations.")

    if os.path.exists(dummy_train_ann_path):
        print(f"\nTesting FaceDataset with dummy annotations:")
        # For WIDER FACE, the image paths in the CSV might be relative to a base image folder.
        # e.g., if img_dir is "../data/wider_face/WIDER_train/images/"
        # and your CSV has "0--Parade/0_Parade_marchingband_1_23.jpg"
        # Then os.path.join(img_dir, "0--Parade/0_Parade_marchingband_1_23.jpg") will work.
        # Ensure your `img_dir` is the base from which paths in the CSV are relative.
        # For this dummy example, image_path already contains the subfolder.
        
        train_dataset = FaceDataset(
            annotations_file=dummy_train_ann_path,
            img_dir="../data/wider_face/WIDER_train/images/", # Base directory for images
            transform=get_transform(train=True)
        )
        
        if len(train_dataset) > 0:
            # Test DataLoader
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0, # Set to 0 for initial testing, >0 for performance
                collate_fn=collate_fn # Important for object detection
            )

            print(f"Number of samples in dummy training dataset: {len(train_dataset)}")
            # Get one batch
            images, targets = next(iter(train_dataloader))
            if images is not None and targets is not None:
                print(f"Batch of images shape (example): {type(images)}, len: {len(images)}")
                print(f"First image in batch tensor shape: {images[0].shape if images else 'N/A'}")
                print(f"Batch of targets (example): {type(targets)}, len: {len(targets)}")
                print(f"First target in batch: {targets[0] if targets else 'N/A'}")
            else:
                print("Failed to load a batch, likely due to missing images or empty dataset.")
        else:
            print("Dummy training dataset is empty. Check paths and dummy data creation.")
    else:
        print(f"Dummy annotation file not found at {dummy_train_ann_path}. Cannot test dataset loading.")
        print("Please create the dummy annotation file or point to a real one.")