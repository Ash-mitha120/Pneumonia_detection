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

## 1️⃣ Clone the Repository

Open a terminal and run:

git clone <your-repo-url>
cd pneumonia-detection
Replace <your-repo-url> with the actual GitHub URL of your repository.

## 2️⃣ Create and Activate Virtual Environment

Create a virtual environment to manage dependencies:

# For Linux / macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
You should see your terminal prompt change to indicate that the virtual environment is active.

3️⃣ Install Required Packages

## Install all dependencies listed in requirements.txt:

pip install -r requirements.txt
If you don’t have a requirements.txt, create one by running:

pip freeze > requirements.txt

## 4️⃣ Prepare the Dataset
 Place your dataset in the chest_xray folder. The folder structure should look like this:

chest_xray/
    train/
    test/
    val/
Make sure your dataset is structured according to the expected training/testing split.

## 5️⃣ Run the Training Script
Train the model by running:

python train_model.py

This will:
Load the dataset from chest_xray/

Apply augmentations from augmentations.py

Train the model and save the best checkpoint (e.g., pneumonia_model.pth or best_model_optuna.pth)

## 6️⃣ Evaluate the Model

After training, evaluate the model using:

python evaluate_model.py


This will generate metrics and confusion matrices for your model.

## 7️⃣ Hyperparameter Tuning

If you want to perform hyperparameter optimization, run:

python optuna_tuner.py

## 8️⃣ View Grad-CAM Results

Visualize model attention using Grad-CAM:

python gradcam_plus_plus.py


Results will be saved in the result of gradcam folder.

## 9️⃣ Notes

Make sure torch is installed with CUDA support if you want GPU acceleration.

Use the pretrained models (pneumonia_model.pth, best_model_optuna.pth) for inference without retraining.
## 🧩 Model Explainability (Grad-CAM++)

Grad-CAM++ provides visual explanations for model predictions, helping to validate that the network focuses on the correct lung regions for pneumonia detection.

Example output:

Grad-CAM++ → Highlights infected lung regions → Improves model transparency.

## 🧪 Technologies Used
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



