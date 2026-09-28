import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import json

from dataset import DAGMDataset

def calculate_metrics(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    
    # Flatten arrays
    preds = preds.view(-1)
    masks = masks.view(-1)
    
    tp = (preds * masks).sum().item()
    fp = (preds * (1 - masks)).sum().item()
    fn = ((1 - preds) * masks).sum().item()
    tn = ((1 - preds) * (1 - masks)).sum().item()
    
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

def train_model():
    # Hyperparameters
    batch_size = 8
    epochs = 5 # Small number for testing/phase 1
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = "c:/Computer/scratch/DAGM_KaggleUpload"
    
    # We will use Class1 and Class2 for quick training, or None for all
    classes = ['Class1', 'Class2'] 

    # Datasets and Loaders
    print("Loading datasets...")
    train_dataset = DAGMDataset(root_dir=root_dir, classes=classes, split='Train')
    test_dataset = DAGMDataset(root_dir=root_dir, classes=classes, split='Test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # Model definition
    # U-Net with ResNet34 encoder
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    ).to(device)

    # Loss and Optimizer
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_iou = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # Apply sigmoid as output is logits
                preds = torch.sigmoid(outputs)
                batch_metrics = calculate_metrics(preds, masks)
                for k in all_metrics:
                    all_metrics[k] += batch_metrics[k]
                    
        val_loss = val_loss / len(test_loader.dataset)
        for k in all_metrics:
            all_metrics[k] /= num_batches
            
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val IoU: {all_metrics['iou']:.4f}")
        
        # Save best model
        if all_metrics['iou'] > best_iou:
            best_iou = all_metrics['iou']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with IoU: {best_iou:.4f}")

    print("Training completed.")
    print("Final Validation Metrics:")
    for k, v in all_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f)

if __name__ == '__main__':
    train_model()
