# Railway-Track-Crack-Detection-Using-CNN

This project focuses on detecting cracks in railway track images using a deep learning model built on the **VGG16 architecture**. By leveraging **transfer learning**, **data augmentation**, and **K-Fold Cross-Validation**, the model achieves high accuracy in classifying **defective** and **non-defective** railway track images. This solution aims to assist autonomous railway inspection systems in ensuring safer operations.

---

## Problem Statement

Manual inspection of railway tracks is:

- Labor-intensive
- Error-prone
- Inefficient for large-scale rail networks

To address these challenges, we propose an **automated image-based system** using a **CNN model** to classify track images, enabling **real-time**, **scalable**, and **reliable crack detection**.

---

## Dataset
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("salmaneunus/railway-track-fault-detection")

print("Path to dataset files:", path)
```

### Categories

- **Defective**: Images containing cracks on railway tracks
- **Non-Defective**: Images of intact tracks

### Preprocessing

- Resized to **128x128**
- Converted to **Grayscale**

### Augmentation

- Random **rotations**
- **Width/Height shift**
- **Shear**, **Zoom**, and **Horizontal flip**

---

## Methodology

### 1️. Model Architecture

- **Base**: Pre-trained `VGG16` (without top layer)
- **Custom Layers**:
  - Global Average Pooling (GAP)
  - Dense layer with **ReLU**
  - Output layer with **Sigmoid** activation (binary classification)

### 2️. Cross-Validation

- **5-Fold Cross Validation** used to improve generalization and reduce overfitting.

### 3️. Training Details

| Parameter       | Value         |
|----------------|---------------|
| Optimizer      | Adam          |
| Loss Function  | Binary Crossentropy |
| Metric         | Accuracy      |
| Batch Size     | 32            |
| Epochs         | 50 (per fold) |

---

## Results

- **Validation Accuracy**: 75%
- Low false positives and false negatives
- Well-generalized to unseen data
- Suitable for real-time autonomous inspection systems

---

## Installation & Usage

### Clone the Repository

```
git clone https://github.com/your-username/railway-track-crack-detection.git
cd railway-track-crack-detection
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the notebook
Open and run Crack_detection_project.ipynb in Jupyter or Colab.

## Prediction on New Images
```
image_path = "<Replace with actual image_path>" 
result = identify_crack(image_path, 'crack_detection_model.keras')
print(result)
```
## Key Features
- Transfer learning using VGG16
- Real-time defect classification
- Data augmentation to boost robustness
- Cross-validation for model reliability
- Designed for scalable railway inspection


