import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

Import model from the other file
from model import CrackDetectorCNN

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model
model = CrackDetectorCNN().to(device)
model_path = "../saved_models/crack_detector_with_validation_cnn.pth" # Updated path
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

#  Image Preprocessing Pipeline

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Because ImageFolder sorts alphabetically during training:
# Class 0 = 'cracked'
# Class 1 = 'uncracked'
class_names = {0: "Cracked", 1: "Uncracked"}

#  The Inference Function
def predict_image(image_path):
    try:
        # Load the raw image
        raw_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Could not load image {image_path}: {e}")
        return

    # Apply preprocessing (Shape becomes [3, 256, 256])
    img_tensor = preprocess(raw_img)
    
    # Add the Batch Dimension (Shape becomes [1, 3, 256, 256])
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run the model without tracking gradients (saves memory/speed)
    with torch.no_grad():
        raw_output = model(img_tensor)
        
        # Apply Sigmoid to convert raw output to a probability (0.0 to 1.0)
        probability = torch.sigmoid(raw_output).item()

    # Determine the class based on a 50% threshold
    if probability >= 0.5:
        predicted_class = 1  # Uncracked
        confidence = probability * 100
    else:
        predicted_class = 0  # Cracked
        # If prob is 0.1, it is 90% confident it is class 0
        confidence = (1.0 - probability) * 100 

    label = class_names[predicted_class]

    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.imshow(raw_img)
    plt.axis('off')
    
    # Color code the title: Red for cracked, Green for uncracked
    title_color = 'red' if predicted_class == 0 else 'green'
    plt.title(f"Prediction: {label}\nConfidence: {confidence:.2f}%", 
              fontsize=16, color=title_color, fontweight='bold')
    plt.tight_layout()
    plt.show()

#  Test it on a new image
app = QApplication(sys.argv)

# Put the path to a new, unseen image here
test_image_path, _ = QFileDialog.getOpenFileName(None, "Select File", "", "All Files (*)") 

predict_image(test_image_path)
