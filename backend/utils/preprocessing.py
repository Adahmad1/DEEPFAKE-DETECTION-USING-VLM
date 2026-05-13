import cv2
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)
