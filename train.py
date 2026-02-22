import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


DATA_DIR = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\data\samples\train"
BATCH_SIZE = 16
NUM_EPOCHS = 5   
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Found {len(train_dataset)} images in {DATA_DIR}")


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Accuracy: {100*correct/total:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/deepfake_classifier.pth")
print("Training complete. Model saved to models/deepfake_classifier.pth")
import os

os.makedirs("models", exist_ok=True)

SAVE_PATH = "models/deepfake_classifier.pth"
torch.save(model.state_dict(), SAVE_PATH)

print("✅ MODEL SAVED SUCCESSFULLY AT:")
print(os.path.abspath(SAVE_PATH))