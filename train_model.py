import os
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

from augmentations import get_train_transforms, get_val_transforms
from model_utils import CustomImageDataset
from cutmix_utils import cutmix_data, cutmix_criterion

def main():
    # ===== Device setup =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler()  # Mixed precision scaler
    print(f"🔹 Using device: {device}")

    # ===== Paths =====
    train_dir = r"D:\pneumonia detection\chest_xray\train"
    val_dir = r"D:\pneumonia detection\chest_xray\val"

    # ===== Dataset & DataLoader =====
    train_dataset = CustomImageDataset(train_dir, get_train_transforms())
    val_dataset = CustomImageDataset(val_dir, get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)

    print(f"📂 Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    # ===== Model =====
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 2)  # Binary classification
    model = model.to(device)

    # ===== Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ===== Tracking metrics =====
    train_losses, train_accuracies = [], []
    val_accuracies, val_f1s = [], []

    # ===== Early stopping setup =====
    best_f1 = 0
    patience = 5
    patience_counter = 0

    # ===== Training Loop =====
    epochs = 30
    for epoch in range(epochs):
        print(f"\n🟣 Starting Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images, ya, yb, lam = cutmix_data(images, labels)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = cutmix_criterion(criterion, outputs, ya, yb, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Training accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        print(f"✅ Epoch {epoch+1} — Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")

        # ===== Validation =====
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        val_accuracies.append(accuracy)
        val_f1s.append(f1)

        print(f"📊 Validation — Acc: {accuracy*100:.2f}% | F1: {f1:.4f}")

        # ===== Early stopping =====
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), "pneumonia_model_amp.pth")
            print("💾 Model improved & saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered.")
                break

    # ===== Plot Accuracy & F1 Curves =====
    epochs_range = range(1, len(train_accuracies)+1)

    plt.figure(figsize=(12, 5))

    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='red')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # F1 curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_f1s, marker='o', color='green', label="Validation F1")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

if __name__ == '__main__':
    main()

