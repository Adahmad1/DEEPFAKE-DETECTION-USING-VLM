import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import os


MODEL_PATH = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\models\deepfake_classifier.pth"
VIDEO_PATH = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\newsf.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")


def extract_frames(video_path, step=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
        
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        idx += 1

    cap.release()
    return frames

frames = extract_frames(VIDEO_PATH)

if len(frames) == 0:
    raise RuntimeError("❌ No frames extracted from video")


LABELS = {0: "FAKE", 1: "REAL"}

votes = {"FAKE": 0, "REAL": 0}

with torch.no_grad():
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img = transform(img).unsqueeze(0).to(DEVICE)
        output = model(img)



        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = LABELS[pred_idx]
        votes[pred_label] += 1

    
        print(f"Frame {i}: {pred_label}")


final_prediction = "FAKE" if votes["FAKE"] > votes["REAL"] else "REAL"

print("🎥 Video Prediction:", final_prediction)
print(f"🧮 Votes → REAL: {votes['REAL']}, FAKE: {votes['FAKE']}")