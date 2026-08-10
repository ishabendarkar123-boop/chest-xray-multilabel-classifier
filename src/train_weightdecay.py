"""
Training script for DenseNet-121 multi-label chest X-ray classification.
Full dataset run -- WITH weight decay (L2 regularization) to address the
overfitting pattern observed in the original run (peaked epoch 4, declined after).
"""

import sys
import time
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append('.')
from dataset import build_image_path_map, ChestXrayDataset, DISEASE_LIST
from model import build_densenet121

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# --- Config ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4   # NEW -- the only real change from the original run

# --- Data (full dataset) ---
image_map = build_image_path_map('../data/raw', cache_path='../data/processed/image_path_map.json')

train_dataset = ChestXrayDataset('../data/processed/train_split.csv', image_map, split='train')
val_dataset = ChestXrayDataset('../data/processed/val_split.csv', image_map, split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- Model, loss, optimizer ---
model = build_densenet121(num_classes=14, pretrained=True).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

def compute_macro_auroc(all_labels, all_preds):
    aurocs = []
    for i in range(len(DISEASE_LIST)):
        if len(np.unique(all_labels[:, i])) < 2:
            continue
        aurocs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    return np.mean(aurocs), aurocs

best_val_auroc = 0.0
epochs_without_improvement = 0
total_training_start = time.time()

for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    val_loss /= len(val_loader)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    val_macro_auroc, _ = compute_macro_auroc(all_labels, all_preds)

    scheduler.step(val_macro_auroc)
    elapsed = time.time() - epoch_start

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Macro AUROC: {val_macro_auroc:.4f} | "
          f"Time: {elapsed/60:.1f} min")

    # --- Checkpointing ---
    if val_macro_auroc > best_val_auroc:
        best_val_auroc = val_macro_auroc
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_macro_auroc': val_macro_auroc,
        }, '../models/densenet121_weightdecay_best.pth')
        print(f"  -> New best model saved (AUROC: {val_macro_auroc:.4f})")
    else:
        epochs_without_improvement += 1

    # --- Early stopping ---
    if epochs_without_improvement >= PATIENCE:
        print(f"\nNo improvement for {PATIENCE} epochs. Stopping early.")
        break

total_training_time = time.time() - total_training_start

print(f"\n{'='*50}")
print(f"COMPARISON SUMMARY (DenseNet-121 + Weight Decay)")
print(f"{'='*50}")
print(f"Best Val Macro AUROC: {best_val_auroc:.4f}")
print(f"Total training time: {total_training_time/60:.1f} minutes")
print(f"\nOriginal DenseNet-121 (no weight decay) for reference: Best AUROC 0.8402 at epoch 4")