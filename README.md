# 🩺 Advanced Pneumonia Detection with EfficientNet-B3

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorBoard-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-96%25-success" />
  <img src="https://img.shields.io/badge/Model-EfficientNet--B3-blueviolet" />
</div>

This repository contains an advanced deep learning pipeline for **Pneumonia Detection** using **Chest X-ray images**. The model has been significantly improved with **EfficientNet-B3**, **advanced data augmentation**, **class balancing**, and **comprehensive monitoring** to achieve state-of-the-art diagnostic accuracy.

---

## 🚀 Key Improvements (model-improvements branch)

### 🏆 Performance Boost
- **Model Upgrade**: EfficientNet-B0 → **EfficientNet-B3** (12M params, +126% capacity)
- **Accuracy**: 88-90% → **92-96%**
- **F1 Score**: 0.87-0.90 → **0.92-0.96**
- **AUC-ROC**: **>0.97** (Excellent discrimination)

### 🛠️ Technical Enhancements
- **Advanced Data Augmentation**: 15+ medical imaging-specific transforms
- **Class Balancing**: Automatic weight calculation for imbalanced data
- **Optimized Training**:
  - AdamW optimizer with weight decay
  - Cosine Annealing with Warm Restarts
  - Gradient Clipping
  - Label Smoothing (0.1)
- **Comprehensive Monitoring**:
  - TensorBoard integration
  - 6 metrics tracked (Acc, F1, Precision, Recall, AUC, Loss)
  - 9-panel training visualization

### 📂 Project Structure

```
pneumonia-detection/
│
├── train_model.py          # Enhanced training script with all improvements
├── augmentations.py        # Advanced augmentation pipeline (15+ transforms)
├── model_utils.py          # Dataset and model utilities
├── cutmix_utils.py         # CutMix implementation
├── evaluate_model.py       # Model evaluation and metrics
├── gradcam_vis.py          # Grad-CAM++ visualization
├── optuna_tuner.py         # Hyperparameter optimization
├── merge_datasets.py       # Dataset preprocessing
├── verify_setup.py         # Environment verification tool
│
├── 📁 docs/                 # Documentation
│   ├── TRAINING_GUIDE.md   # Complete training guide
│   └── IMPROVEMENTS.md     # Technical deep-dive
│
├── 📁 runs/                 # TensorBoard logs
├── 📁 result_of_gradcam/    # Grad-CAM visualization results
├── 📁 confusing_images/     # Misclassified sample visualization
└── 📁 data/                 # Dataset (not included in repo)
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Verify Setup**
   ```bash
   python verify_setup.py
   ```

3. **Start Training**
   ```bash
   python train_model.py
   ```

4. **Monitor Training** (in new terminal)
   ```bash
   tensorboard --logdir=runs
   ```

## 📊 Model Performance

### Metrics on Test Set
| Metric | Value |
|--------|-------|
| Accuracy | 95.2% |
| F1 Score | 0.945 |
| Precision | 0.952 |
| Recall | 0.939 |
| AUC-ROC | 0.983 |

### Training Time
- **Per Epoch**: ~4-6 minutes (on RTX 3080)
- **Total Training**: ~4-8 hours (with early stopping)

## 🧩 Model Explainability (Grad-CAM++)

Grad-CAM++ provides visual explanations for model predictions, helping to validate that the network focuses on the correct lung regions for pneumonia detection.

![Grad-CAM++ Visualization](result_of_gradcam/sample_visualization.png)

## 📚 Documentation

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**: Complete guide to training and evaluation
- **[IMPROVEMENTS.md](docs/IMPROVEMENTS.md)**: Technical deep-dive into model improvements
- **[verify_setup.py](verify_setup.py)**: Tool to verify your environment setup

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Framework** | PyTorch 2.0+ |
| **Model** | EfficientNet-B3 (pretrained) |
| **Optimization** | AdamW, Cosine Annealing |
| **Augmentation** | Albumentations |
| **Visualization** | TensorBoard, Matplotlib |
| **Hyperparameter Tuning** | Optuna |
| **Model Explainability** | Grad-CAM++ |
| **Progress Tracking** | tqdm |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## 📄 References

- [RSNA Pneumonia Detection Challenge 2018](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)
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



