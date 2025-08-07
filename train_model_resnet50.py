import os
import random
from collections import Counter
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setting
DATASET_ROOT = "dataset_aug"   # from offline aug
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
MODEL_SAVE_PATH = "kendo_resnet50.pth"

# Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# load dataset
full_dataset = datasets.ImageFolder(DATASET_ROOT, transform=val_test_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

# Stratified Split
labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]
sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, temp_idx = next(sss_temp.split(np.zeros(len(labels)), labels))

temp_labels = [labels[i] for i in temp_idx]
sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(sss_val_test.split(np.zeros(len(temp_labels)), temp_labels))
val_idx = [temp_idx[i] for i in val_idx]
test_idx = [temp_idx[i] for i in test_idx]

def label_dist(indices):
    c = Counter([labels[i] for i in indices])
    return {class_names[k]: c.get(k, 0) for k in range(len(class_names))}

print("Train Dist:", label_dist(train_idx))
print("Val Dist:", label_dist(val_idx))
print("Test Dist:", label_dist(test_idx))

# class weight
train_labels = [labels[i] for i in train_idx]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", dict(zip(class_names, class_weights.numpy())))

# Dataset & DataLoader
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ConvNeXt Tiny + ImageNet
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
class_weights = class_weights.to(device)

#fine-tune
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in list(model.named_parameters())[-20:]:  # fine tune the last few layer
    param.requires_grad = True

# Use class weight to balance loss
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# train and val
def run_epoch(loader, training=True):
    if training:
        model.train()
    else:
        model.eval()
    total_loss, total_correct = 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc

# loop
best_val_acc = 0
for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    print(f"[Epoch {epoch+1}/{EPOCHS}] "
          f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
          f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

    #save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": full_dataset.class_to_idx
        }, MODEL_SAVE_PATH)
        print(f"Best model saved (Val Acc={val_acc:.4f})")

# test
print("\nTest:")
checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
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
print(f"Test Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Best model is saved at:", MODEL_SAVE_PATH)