import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# 1. First define the model class
class ImagenetteClassifier(nn.Module):
    def __init__(self, use_batchnorm=False, use_dropout=False):
        super().__init__()
        
        # Conv Layer 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        
        # Conv Layer 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        
        # Conv Layer 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        
        # Pool layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate size after convolutions and pooling
        # Input: 160x160 -> After 3 pools: 20x20
        feature_size = 256 * 20 * 20
        
        # Linear Layers
        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes
        
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

    def forward(self, x):
        # Conv Layers with ReLU, BatchNorm (optional), and MaxPool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 160x160 -> 80x80
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 80x80 -> 40x40
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 40x40 -> 20x20
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Linear Layers with Dropout (optional)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 2. Define the transforms function
def get_transforms(use_augmentation=False):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Fixed size resize
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Fixed size resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Fixed size resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


# 3. Define the training function
def train_model(model, train_loader, val_loader, epochs=20, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize lists to store metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate validation metrics
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        print('-' * 60)
    
    return train_losses, train_accs, val_losses, val_accs


# 4. Define plotting function
def plot_results(results_dict, metric='loss'):
    plt.figure(figsize=(10, 6))
    
    for name, results in results_dict.items():
        if metric == 'loss':
            plt.plot(results[0], label=f'{name} Train Loss')
            plt.plot(results[2], label=f'{name} Val Loss')
        else:
            plt.plot(results[1], label=f'{name} Train Acc')
            plt.plot(results[3], label=f'{name} Val Acc')
    
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Define confusion matrix function
def get_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# Setup data loading
def setup_data(data_dir, batch_size=32, use_augmentation=False):
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip() if use_augmentation else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(10) if use_augmentation else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(0.2, 0.2, 0.2) if use_augmentation else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
        
        # Create dataloaders with appropriate settings for CPU
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for CPU
            pin_memory=False  # Set to False for CPU
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for CPU
            pin_memory=False  # Set to False for CPU
        )
        
        return train_loader, val_loader, train_dataset.classes
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

# Main training code
if __name__ == "__main__":
    print("Cuda available?: " + str(torch.cuda.is_available()))
    # Set device - now defaulting to CPU
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Set data directory
    data_dir = 'imagenette2-160'
    
    # Setup data
    train_loader, val_loader, classes = setup_data(data_dir, batch_size=32)
    
    if train_loader is None:
        print("Failed to load data. Exiting.")
        exit()
    
    # Create and train model
    print("Training base model...")
    model_base = ImagenetteClassifier()  # Create model
    
    # Train model
    results = {}
    results['base'] = train_model(model_base, train_loader, val_loader, device=device)
    
    # Save the trained model
    torch.save(model_base.state_dict(), 'imagenette_model.pth')
    print("Model saved successfully!")
