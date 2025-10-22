import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import CustomImageDataset
from augmentations import get_val_transforms
from efficientnet_pytorch import EfficientNet

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# ===== Paths =====
data_dir = "chest_xray"
test_dir = os.path.join(data_dir, "test")

# Automatically find latest model file
model_files = [f for f in os.listdir() if f.endswith(".pth")]
if not model_files:
    raise FileNotFoundError("❌ No .pth model file found in current directory.")
model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
model_path = model_files[0]
print(f"📦 Loading latest model: {model_path}")

def evaluate():
    # ===== Dataset =====
    test_dataset = CustomImageDataset(test_dir, get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)  # avoid multiprocessing error

    print(f"📂 Test images: {len(test_dataset)}")

    # ===== Load Model =====
    model = EfficientNet.from_name('efficientnet-b0')  # architecture
    model._fc = nn.Linear(model._fc.in_features, 2)   # binary output
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # ===== Inference =====
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ===== Metrics =====
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n📊 Test Results:")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))

    # ===== Confusion Matrix =====
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_test.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
