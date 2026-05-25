# CNN-Crack-Detector
# 🏗️ Concrete Crack Detector CNN
A deep learning project utilizing a custom Convolutional Neural Network (CNN) built in PyTorch to detect structural cracks in concrete surfaces. This repository includes a full training pipeline with data augmentation, validation tracking, and a PyQt5 desktop application for easy, click-and-point inference.

# ✨ Features
* Custom CNN Architecture: A lightweight, 4-block convolutional network optimized for binary classification (Cracked vs. Uncracked).
* Robust Training Pipeline: Includes on-the-fly data augmentation (flips, rotations, color jitter), dropout regularization, and real-time metric tracking to prevent overfitting.
* Desktop GUI: A standalone PyQt5 application allowing users to upload local images and get instant predictions with confidence percentages.
* Hardware Agnostic: Automatically detects and utilizes NVIDIA CUDA GPUs for training/inference, but safely falls back to CPU if no GPU is available.

# 📂 Project Structure
```Plaintext 
cnn-crack-detector/
│
├── my_dataset/             # (Ignored in Git) User-provided image dataset
├── saved_models/           # (Ignored in Git) Trained .pth weight files
├── output_graphs/          # (Ignored in Git) Training metrics visualizations
│
├── src/                    
│   ├── model.py            # CNN Class definition
│   ├── train.py            # Training loop, validation, and graph plotting
│   └── test.py              # PyQt5 GUI inference application
│
├── .gitignore              # Prevents heavy data/models from uploading
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

# ⚙️ Installation

Clone the repository:
```bash 
git clone https://github.com/YOUR_USERNAME/cnn-crack-detector.git
cd cnn-crack-detector
```

Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

# 📊 Dataset Preparation

* Because image datasets are too large for GitHub, you must provide your own data to train the model from scratch.
* Create a folder named my_dataset in the root directory.
* Inside my_dataset, create two subfolders exactly named cracked and uncracked.
* Place your .jpg or .png images into their respective folders.

Your directory should look like this:
```Plaintext
my_dataset/
├── cracked/
│   ├── image1.jpg
│   └── image2.jpg
└── uncracked/
    ├── image1.jpg
    └── image2.jpg
```
# 🚀 Usage

1. Training the Model
Once your dataset is in place, run the training script. The script automatically splits your data (80% Train / 20% Validation), trains the network, saves the best weights, and generates a loss/accuracy graph.

```bash
python src/train.py
```
Outputs: * saved_models/crack_detector_with_validation_cnn.pth

  output_graphs/training_validation_metrics.png

2. Running the GUI (Inference)
To test the model on new, unseen images, launch the desktop application:

```bash
python src/app.py
```
A window will appear. Click "Select File", choose an image of a concrete surface, and the model will pop up a window displaying the image, its prediction, and its exact confidence percentage.

# 🧠 Pre-Trained Weights

Don't want to train the model yourself? You can download my pre-trained weights here:
👉 [Download crack_detector_cnn.pth](Insert Your Link Here)

After downloading, create a saved_models/ folder in the root directory and place the .pth file inside before running src/app.py.

# 🛠️ Built With
* PyTorch - The core Deep Learning framework.
* Torchvision - Image processing and augmentation.
* PyQt5 - For selecting the image using select window.
* Matplotlib - Visualization and metric plotting.
