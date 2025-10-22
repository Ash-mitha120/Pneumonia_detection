# 🩺 Pneumonia Detection Using Transfer Learning (EfficientNet-B0)

This repository contains an end-to-end deep learning project for **Pneumonia Detection** using **Chest X-ray images**.  
The model leverages **Transfer Learning (EfficientNet-B0)**, **CutMix augmentation**, **Grad-CAM++ visualization**, and **Optuna-based hyperparameter optimization** to achieve high diagnostic accuracy.

---

## 📂 Project Structure

pneumonia-detection/
│
├── augmentations.py # Data augmentation transformations
├── cutmix_utils.py # CutMix implementation
├── evaluate_model.py # Model evaluation and metrics
├── gradcam_plus_plus.py # Grad-CAM++ visualization
├── merge_datasets.py # Dataset preprocessing and merging
├── model_utils.py # Utility functions for model handling
├── optuna_tuner.py # Optuna-based hyperparameter tuning
├── train_model.py # Model training script
│
├── result of gradcam/ # Grad-CAM visualization results
├── confusing_images/ # Misclassified sample visualization
├── venv/ # Virtual environment (excluded in .gitignore)
├── best_model_optuna.pth # Saved best model weights
│
└── README.md # Project documentation


---

## 🧠 Architecture Overview

The following diagram represents the architecture flow used in this project:



Input Layer → Data Augmentation Layer → Normalization Layer → No Tensor Layer →
CutMix Augmentation Layer → EfficientNet-B0 Layer → Dropout Layer →
Grad-CAM++ Visualization Layer → Hyperparameter Optimization Layer → Output Layer


---

## 📊 Dataset

The model was trained and validated on publicly available pneumonia datasets:

- **RSNA Pneumonia Detection Challenge (2018)**:  
  [https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
- **NIH Chest X-Ray Dataset**
- **Kaggle Pneumonia Dataset**

**Dataset Split:**
| Split | Normal Images | Pneumonia Images | Total |
|:------|:---------------|:-----------------|:------|
| Train | 7,518 | 7,514 | 15,032 |
| Test  | 234 | 390 | 624 |
| Val   | 8 | 8 | 16 |

> ⚠️ The dataset is **not included** in this repository due to size limitations.  
> Please download it from the provided links and place it in a folder named `chest_xray/` inside the project root.

---

## ⚙️ Features Implemented

- ✅ **Transfer Learning** using `EfficientNet-B0`
- 🧩 **CutMix Data Augmentation**
- 📈 **Optuna Hyperparameter Optimization**
- 🧠 **Grad-CAM++** for model explainability
- 🛠️ **Early Stopping** & Learning Rate Scheduling
- 🔥 **Mixed Precision Training** for performance boost (using `torch.amp`)

---

## 🏗️ Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/pneumonia-detection.git
cd pneumonia-detection

2️⃣  **Create and Activate Virtual Environment**

```bash
python -m venv venv
# For Linux/Mac
source venv/bin/activate
# For Windows
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🚀 Training the Model

To train the model, run:

python train_model.py


To tune hyperparameters with Optuna:

python optuna_tuner.py


To evaluate the model:

python evaluate_model.py


Grad-CAM++ Visualization Example:

Visual heatmaps highlighting pneumonia-infected lung regions are stored under
result of gradcam/ directory.

🧩 Model Explainability (Grad-CAM++)

Grad-CAM++ provides visual explanations for model predictions, helping to validate that the network focuses on the correct lung regions for pneumonia detection.

Example output:

Grad-CAM++ → Highlights infected lung regions → Improves model transparency.

🧪 Technologies Used
Category	Tools / Libraries
Framework	PyTorch
Model	EfficientNet-B0
Optimization	Optuna
Visualization	Grad-CAM++
Augmentation	CutMix, Torchvision
Logging	Matplotlib, tqdm

### 📘 **References**

RSNA Pneumonia Detection Challenge 2018
NIH Chest X-ray Dataset
EfficientNet Paper (Tan & Le, 2019)
CutMix: Regularization Strategy to Train Strong Classifiers
Grad-CAM++: Improved Visual Explanations for CNNs

🤝 Contributing

Contributions are welcome!
If you find bugs or improvements, please open an issue or submit a pull request.

📄 License

This project is released under the MIT License.
Feel free to use, modify, and share with proper attribution.

---



