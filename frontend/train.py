import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import numpy as np
import os

# ================= CONFIG =================
BASE_DIR = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\data\samples"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ================= DATA =================
train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

# ===== SPLIT TRAIN → TRAIN + VAL =====
train_size = int(0.85 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size

train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

# ================= MODEL =================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ================= CLASS WEIGHTS =================
targets = [label for _, label in train_dataset_full]
class_counts = np.bincount(targets)
weights = 1. / class_counts
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    train_acc = 100 * correct / total

    # ===== VALIDATION =====
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            _, pred = torch.max(out, 1)

            total += y.size(0)
            correct += (pred == y).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

    # ===== SAVE BEST =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Best model saved")

# ================= TEST EVALUATION =================
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1]
        _, pred = torch.max(out, 1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(all_labels, all_preds)

os.makedirs("plots", exist_ok=True)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test)")
plt.savefig("plots/confusion_matrix.png")
plt.show()

# ================= REPORT =================
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))

# ================= ROC =================
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend()
plt.savefig("plots/roc_curve.png")
plt.show()

print(f"\n🔥 Test ROC AUC: {roc_auc:.4f}")
