import torch
import cv2
import numpy as np

def simple_heatmap(image_path):
    img = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        cv2.COLORMAP_JET
    )
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
