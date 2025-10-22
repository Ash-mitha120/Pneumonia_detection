import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score
import optuna
from augmentations import get_train_transforms, get_val_transforms
from model_utils import CustomImageDataset
from cutmix_utils import cutmix_data, cutmix_criterion

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# ===== Paths =====
train_dir = r"D:\pneumonia detection\chest_xray\train"
val_dir = r"D:\pneumonia detection\chest_xray\val"

# ===== Global variable to save best model =====
best_model_path = "best_model_optuna.pth"
global_best_f1 = 0

# ===== Objective function =====
def objective(trial):
    global global_best_f1

    # Hyperparameters to tune
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    cutmix_prob = trial.suggest_uniform("cutmix_prob", 0.3, 0.7)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.2, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # ===== Dataset & Dataloader =====
    train_dataset = CustomImageDataset(train_dir, get_train_transforms())
    val_dataset = CustomImageDataset(val_dir, get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ===== Model =====
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model._fc.in_features, 2)
    )
    model = model.to(device)

    # ===== Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler()

    # ===== Training =====
    best_val_f1 = 0
    epochs = 15  # Short training for Optuna
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images, ya, yb, lam = cutmix_data(images, labels, cutmix_prob)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = cutmix_criterion(criterion, outputs, ya, yb, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ===== Validation =====
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            # ===== Save model if it's the global best =====
            if val_f1 > global_best_f1:
                global_best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"💾 New global best model saved! F1: {val_f1:.4f}")

    return best_val_f1

# ===== Run Optuna study =====
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("🏆 Best trial:")
    trial = study.best_trial
    print(f"  F1-score: {trial.value}")
    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"✅ Best model saved as: {best_model_path}")
