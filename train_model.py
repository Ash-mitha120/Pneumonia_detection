import os
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augmentations import get_train_transforms, get_val_transforms
from model_utils import CustomImageDataset
from cutmix_utils import cutmix_data, cutmix_criterion

def calculate_class_weights(dataset):
    """Calculate class weights for handling imbalanced dataset"""
    labels = [label for _, label in dataset.data]
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def main():
    # ===== Device setup =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision scaler
    print(f"🔹 Using device: {device}")
    print(f"🔹 PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🔹 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔹 CUDA version: {torch.version.cuda}")
    
    # ===== TensorBoard Setup =====
    writer = SummaryWriter('runs/pneumonia_detection')
    print("📊 TensorBoard logging enabled. Run: tensorboard --logdir=runs")

    # ===== Paths =====
    train_dir = r"C:\Users\ASUS\Downloads\archive\chest_xray\train"
    val_dir = r"C:\Users\ASUS\Downloads\archive\chest_xray\val"

    # ===== Dataset & DataLoader =====
    train_dataset = CustomImageDataset(train_dir, get_train_transforms())
    val_dataset = CustomImageDataset(val_dir, get_val_transforms())
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_dataset).to(device)
    print(f"⚖️  Class weights: {class_weights.cpu().numpy()}")

    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Increased batch size
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print(f"📂 Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    # ===== Model with Dropout =====
    model = EfficientNet.from_pretrained('efficientnet-b3')  # Using larger model
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🔢 Trainable parameters: {trainable_params:,}")

    # ===== Loss & Optimizer with improvements =====
    # Label smoothing + class weighting for better generalization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    # Cosine Annealing LR Scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Alternative: ReduceLROnPlateau
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-7
    # )

    # ===== Tracking metrics =====
    train_losses, train_accuracies = [], []
    val_accuracies, val_f1s, val_precisions, val_recalls = [], [], [], []
    learning_rates = []

    # ===== Early stopping setup =====
    best_f1 = 0
    best_acc = 0
    patience = 10  # Increased patience
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement threshold

    # ===== Training Loop =====
    epochs = 50  # Increased epochs
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"\n{'='*60}")
        print(f"🟣 Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f}")
        print(f"{'='*60}")
        
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Apply CutMix with 50% probability
            if np.random.rand() > 0.5:
                images, ya, yb, lam = cutmix_data(images, labels)
                use_cutmix = True
            else:
                ya, yb, lam = labels, labels, 1.0
                use_cutmix = False

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                if use_cutmix:
                    loss = cutmix_criterion(criterion, outputs, ya, yb, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Training accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"✅ Training — Loss: {avg_train_loss:.4f} | Acc: {train_acc*100:.2f}%")

        # ===== Validation =====
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

        # Calculate comprehensive metrics
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        avg_val_loss = val_loss / len(val_loader)

        val_accuracies.append(accuracy)
        val_f1s.append(f1)
        val_precisions.append(precision)
        val_recalls.append(recall)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('F1/val', f1, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('AUC/val', auc, epoch)

        print(f"📊 Validation — Loss: {avg_val_loss:.4f} | Acc: {accuracy*100:.2f}% | F1: {f1:.4f}")
        print(f"   Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f}")

        # ===== Learning Rate Scheduling =====
        scheduler.step()
        
        # ===== Enhanced Early Stopping & Checkpointing =====
        improved = False
        
        # Save best model based on F1 score
        if f1 > best_f1 + min_delta:
            best_f1 = f1
            patience_counter = 0
            improved = True
            
            # Save comprehensive checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_f1': best_f1,
                'best_acc': accuracy,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'val_f1s': val_f1s,
            }
            torch.save(checkpoint, "best_model_f1.pth")
            torch.save(model.state_dict(), "pneumonia_model_best.pth")  # Backward compatibility
            print(f"💾 🏆 New Best F1: {best_f1:.4f} - Model saved!")
        
        # Save best model based on accuracy
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best_model_accuracy.pth")
            if not improved:
                print(f"💾 New Best Accuracy: {best_acc*100:.2f}%")
        
        if not improved:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("\n⏹ Early stopping triggered.")
                print(f"🏆 Best F1 Score: {best_f1:.4f}")
                print(f"🏆 Best Accuracy: {best_acc*100:.2f}%")
                break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
            print(f"💾 Checkpoint saved at epoch {epoch+1}")

    # Close TensorBoard writer
    writer.close()
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")
    print(f"🏆 Best Accuracy: {best_acc*100:.2f}%")
    print("="*60)
    
    # ===== Plot Comprehensive Training Curves =====
    epochs_range = range(1, len(train_accuracies)+1)

    fig = plt.figure(figsize=(18, 12))

    # Accuracy curve
    plt.subplot(3, 3, 1)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='red', linewidth=2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='blue', linewidth=2)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.title("Training vs Validation Accuracy", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss curve
    plt.subplot(3, 3, 2)
    plt.plot(epochs_range, train_losses, label='Train Loss', color='orange', linewidth=2)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Training Loss", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # F1 curve
    plt.subplot(3, 3, 3)
    plt.plot(epochs_range, val_f1s, marker='o', color='green', linewidth=2, markersize=4, label="Validation F1")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.title("Validation F1 Score", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision curve
    plt.subplot(3, 3, 4)
    plt.plot(epochs_range, val_precisions, marker='s', color='purple', linewidth=2, markersize=4, label="Precision")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.title("Validation Precision", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recall curve
    plt.subplot(3, 3, 5)
    plt.plot(epochs_range, val_recalls, marker='^', color='brown', linewidth=2, markersize=4, label="Recall")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Recall", fontsize=10)
    plt.title("Validation Recall", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate curve
    plt.subplot(3, 3, 6)
    plt.plot(epochs_range, learning_rates, color='navy', linewidth=2)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Learning Rate", fontsize=10)
    plt.title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Combined metrics
    plt.subplot(3, 3, 7)
    plt.plot(epochs_range, val_f1s, label='F1', linewidth=2)
    plt.plot(epochs_range, val_precisions, label='Precision', linewidth=2)
    plt.plot(epochs_range, val_recalls, label='Recall', linewidth=2)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.title("All Validation Metrics", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary stats
    plt.subplot(3, 3, 8)
    plt.axis('off')
    summary_text = f"""
    Training Summary
    {'='*30}
    
    Total Epochs: {len(train_accuracies)}
    
    Best F1 Score: {best_f1:.4f}
    Best Accuracy: {best_acc*100:.2f}%
    
    Final Train Acc: {train_accuracies[-1]*100:.2f}%
    Final Val Acc: {val_accuracies[-1]*100:.2f}%
    Final Val F1: {val_f1s[-1]:.4f}
    
    Model: EfficientNet-B3
    Optimizer: AdamW
    Scheduler: CosineAnnealingWarmRestarts
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
             verticalalignment='center')

    plt.suptitle('Pneumonia Detection Training Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig("training_curves_comprehensive.png", dpi=300, bbox_inches='tight')
    print("\n📊 Training curves saved as 'training_curves_comprehensive.png'")
    plt.show()

if __name__ == '__main__':
    main()

