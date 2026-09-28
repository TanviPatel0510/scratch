import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DAGMDataset(Dataset):
    def __init__(self, root_dir, classes=None, split='Train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes (e.g. DAGM_KaggleUpload).
            classes (list): List of class folders to include, e.g., ['Class1', 'Class2']. 
                            If None, reads all Class folders.
            split (string): 'Train' or 'Test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.samples = [] # List of tuples: (image_path, mask_path)
        
        if classes is None:
            classes = [d for d in os.listdir(root_dir) if d.startswith('Class') and os.path.isdir(os.path.join(root_dir, d))]
            
        for cls in classes:
            split_dir = os.path.join(root_dir, cls, split)
            if not os.path.exists(split_dir):
                continue
                
            label_dir = os.path.join(split_dir, 'Label')
            
            for file_name in os.listdir(split_dir):
                if file_name.endswith('.PNG') and not file_name.endswith('_label.PNG'):
                    img_path = os.path.join(split_dir, file_name)
                    
                    # Construct potential mask path
                    base_name = os.path.splitext(file_name)[0]
                    mask_name = f"{base_name}_label.PNG"
                    mask_path = os.path.join(label_dir, mask_name)
                    
                    # If mask doesn't exist, we will handle it as a defect-free image (all zeros)
                    if not os.path.exists(mask_path):
                        mask_path = None
                        
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Read image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Convert to RGB (segmentation models usually expect 3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Threshold to make it strictly binary (0 or 1)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        else:
            # Create a zero mask for defect-free images
            mask = np.zeros_like(image[:,:,0])
            
        # Add channel dimension to mask (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform:
            image = self.transform(image)
            # transform for mask? 
            # If using torchvision transforms, they work on PIL Image or Tensors.
            # Usually we use albumentations for simultaneous img/mask transforms.
            # But let's keep it simple with ToTensor manually for now.
            
        # Convert to tensors
        # image to (C, H, W) and scale to [0, 1]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        # mask to (C, H, W)
        mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1)
            
        return image_tensor, mask_tensor
