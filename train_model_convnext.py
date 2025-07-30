import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

SEED = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASET_ROOT = "dataset_more_none"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
MODEL_SAVE_PATH = "kendo_classifier_convnext.pth"

#Data Augmentation (keep left/right orientation!)
# train_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# val_test_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),

    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02
        )
    ], p=0.5),

    transforms.RandomApply([
        transforms.Lambda(lambda img: img.convert("RGB")),  # 保證是RGB
        transforms.Lambda(lambda img: transforms.functional.adjust_jpeg_quality(img, quality=80))
    ], p=0.3),

    transforms.RandomApply([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0))
    ], p=0.4),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset (initially no augmentation)
full_dataset = datasets.ImageFolder(DATASET_ROOT, transform=val_test_transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# Extract labels for stratified splitting
all_labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]

# Stratified split: 80% train, 10% val, 10% test
sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # 80/20
train_idx, temp_idx = next(sss_temp.split(np.zeros(len(all_labels)), all_labels))

temp_labels = [all_labels[i] for i in temp_idx]
sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)  # split 10%/10%
val_idx, test_idx = next(sss_val_test.split(np.zeros(len(temp_labels)), temp_labels))

# map val/test back to original index
val_idx = [temp_idx[i] for i in val_idx]
test_idx = [temp_idx[i] for i in test_idx]

print(f"Stratified split done:")
print(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

# Subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

def count_labels(indices, labels):
    subset_labels = [labels[i] for i in indices]
    counter = Counter(subset_labels)
    return {class_names[k]: counter.get(k, 0) for k in range(len(class_names))}

train_dist = count_labels(train_idx, all_labels)
val_dist = count_labels(val_idx, all_labels)
test_dist = count_labels(test_idx, all_labels)

print("\nClass distribution:")
print("Train:", train_dist)
print("Val:  ", val_dist)
print("Test: ", test_dist)

# Augmentation
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Category: {class_names}")

# === Build ConvNeXt Tiny ===
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# fine-tune last ~100 params
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in list(model.named_parameters())[-100:]:
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

def train_one_epoch(epoch):
    model.train()
    total_loss, total_correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} Acc: {acc:.4f}")

def validate(epoch):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(val_loader.dataset)
    acc = total_correct / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_loss:.4f} Acc: {acc:.4f}")

def test_model():
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nAccuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(class_names)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# === Training loop ===
for epoch in range(EPOCHS):
    train_one_epoch(epoch)
    validate(epoch)

test_model()

torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": full_dataset.class_to_idx
}, MODEL_SAVE_PATH)
print(f"\nModel saved at {MODEL_SAVE_PATH}")
