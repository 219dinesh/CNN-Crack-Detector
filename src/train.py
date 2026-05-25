import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

Import your model from the other file
from model import CrackDetectorCNN

# ==========================================
# 1. Data Loading & Splitting (Train/Val)
# ==========================================
print("Loading and splitting data...")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip left/right
    transforms.RandomVerticalFlip(p=0.5),   # 50% chance to flip up/down
    transforms.RandomRotation(degrees=15),  # Randomly tilt the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Random lighting changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = 'my_dataset' 
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

Splitting the dataset 90% Train, 10% Validation ---
total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create separate DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # No need to shuffle val data

print(f"Total images: {total_size} | Training on: {train_size} | Validating on: {val_size}")


# ==========================================
# 2. Setup Training
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = CrackDetectorCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 60

# Tracking arrays for both Train and Val 
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

# ==========================================
# 2. The Training & Validation Loop
# ==========================================
print("\nStarting Training...")
for epoch in range(num_epochs):
    
    # -----------------------
    # PHASE 1: TRAINING
    # -----------------------
    model.train() # Turns on Dropout and BatchNorm
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Calculate Accuracy
        probs = torch.sigmoid(predictions)
        preds = (probs >= 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = (train_correct / train_total) * 100.0
    
    # -----------------------
    # PHASE 2: VALIDATION
    # -----------------------
    model.eval() # Turns OFF Dropout so we get clean predictions
    val_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad(): # Don't track gradients (saves memory & speeds up)
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            predictions = model(images)
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            
            # Calculate Accuracy
            probs = torch.sigmoid(predictions)
            preds = (probs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = (val_correct / val_total) * 100.0
    
    # Save metrics to history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_acc'].append(avg_val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.1f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.1f}%")

print("\nTraining Complete!")

# Save the trained model
os.makedirs('../saved_models', exist_ok=True)
model_path = "../saved_models/crack_detector_with_validation_cnn.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to '{model_path}'")

# ==========================================
# 4. Plotting and Saving the Curves
# ==========================================
# Create a figure with 2 side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, num_epochs + 1)

# Plot 1: Loss Curve
ax1.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', marker='o')
ax1.plot(epochs_range, history['val_loss'], 'r--', label='Validation Loss', marker='s')
ax1.set_title('Training vs Validation Loss', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (BCE)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(fontsize=12)

# Plot 2: Accuracy Curve
ax2.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy', marker='o')
ax2.plot(epochs_range, history['val_acc'], 'g--', label='Validation Accuracy', marker='s')
ax2.set_title('Training vs Validation Accuracy', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=12)

plt.tight_layout()

# Save the plot to a file
os.makedirs('../output_graphs', exist_ok=True)
plot_filename = "../output_graphs/training_validation_metrics.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to '{plot_filename}'")

# Display the plot on screen
plt.show()
