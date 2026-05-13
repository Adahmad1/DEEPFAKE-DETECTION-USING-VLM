import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIG =================
DATA_DIR = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\data\samples\test"
MODEL_PATH = r"models/best_model.pth"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ================= DATA =================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Testing on {len(dataset)} images")

# ================= MODEL =================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Loaded best model")

# ================= EVALUATION =================
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:,1]  # probability of REAL
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ================= CONFUSION MATRIX =================
os.makedirs("plots", exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.show()

# ================= CLASSIFICATION REPORT =================
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))

# ================= ROC CURVE =================
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("plots/roc_curve.png")
plt.show()

print(f"🔥 ROC AUC Score: {roc_auc:.4f}")