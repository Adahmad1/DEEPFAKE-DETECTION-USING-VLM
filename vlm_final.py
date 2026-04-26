"""
Deepfake Detection with VLM + Natural Language Explanation
==========================================================
Pipeline:
  Face detection  ->  ResNet18 (CNN)  +  CLIP (VLM)  ->  Fusion
  ->  Grad-CAM heatmap  ->  Ollama (local LLM) NLP explanation
  ->  Annotated output video  +  Full forensic report

Requirements:
  pip install torch torchvision transformers opencv-python Pillow numpy requests
  Ollama installed from https://ollama.com  +  ollama pull llama3.2
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import requests

# ================= PATHS =================
MODEL_PATH   = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\models\best_model.pth"
VIDEO_PATH   = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\mu.mp4"
OUTPUT_VIDEO = "output_explained.mp4"
REPORT_PATH  = "deepfake_report.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ================= CONFIG =================
LABELS         = {0: "FAKE", 1: "REAL"}
CNN_WEIGHT     = 0.75
VLM_WEIGHT     = 0.25
CONF_THRESHOLD = 0.6
FRAME_STEP     = 5    # analyse every Nth frame (lower = more thorough but slower)

# ================= OLLAMA CONFIG =================
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= LOAD CNN =================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ CNN (ResNet18) loaded")

# ================= LOAD CLIP =================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("✅ CLIP loaded")

# CLIP prompts: index 0 = FAKE, index 1 = REAL  (must match CNN label order)
CLIP_TEXTS = [
    "a deepfake face with blending artifacts, inconsistent lighting, or unnatural skin texture",
    "a real human face with natural lighting and realistic texture"
]

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= GRAD-CAM =================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, face_pil, class_idx):
        """
        Fresh tensor with requires_grad=True so backward() works correctly.
        """
        img_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(img_tensor)
        output[0, class_idx].backward()

        gradients   = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

gradcam = GradCAM(model, model.layer4[-1])


# ================= CLIP INFERENCE =================
def get_vlm_probs(image_pil):
    """Returns np.array([P(FAKE), P(REAL)]) aligned with CNN label order."""
    with torch.no_grad():
        inputs  = clip_processor(
            text=CLIP_TEXTS,
            images=image_pil,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        outputs = clip_model(**inputs)
        probs   = outputs.logits_per_image.softmax(dim=1)
    return probs[0].cpu().numpy()


# ================= CAM REGION ANALYSIS =================
def analyse_cam_regions(cam):
    """
    Divides the 224x224 CAM into a 3x3 spatial grid.
    Returns top-3 regions sorted by average activation intensity.
    """
    h, w           = cam.shape
    cell_h, cell_w = h // 3, w // 3
    row_names      = ["upper", "middle", "lower"]
    col_names      = ["left",  "center", "right"]

    regions = []
    for r in range(3):
        for c in range(3):
            patch = cam[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            regions.append({
                "name":  f"{row_names[r]}-{col_names[c]}",
                "score": float(patch.mean())
            })

    regions.sort(key=lambda x: x["score"], reverse=True)
    return regions[:3]


# ================= OLLAMA LLM CALL =================
def call_llm(prompt):
    """
    Sends a prompt to the local Ollama server and returns the text response.
    Ollama must be running and llama3.2 must be pulled before calling this.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return ("Ollama is not running. Open a terminal and run: ollama serve")
    except Exception as e:
        return f"LLM error: {str(e)}"


# ================= PER-FRAME NLP EXPLANATION =================
def generate_frame_explanation(label, cnn_conf, vlm_conf, hot_regions, frame_id):
    """
    Asks local LLM to produce a forensic-style explanation for
    why this specific frame was classified as FAKE or REAL.
    """
    region_text = ", ".join(
        f"{r['name']} region (intensity {r['score']:.2f})" for r in hot_regions
    )
    fused = CNN_WEIGHT * cnn_conf + VLM_WEIGHT * vlm_conf

    if label == "FAKE":
        prompt = (
            f"You are a deepfake forensic analyst. Frame #{frame_id} is classified as FAKE.\n"
            f"ResNet18 confidence: {cnn_conf:.2f}, CLIP confidence: {vlm_conf:.2f}, "
            f"Fused confidence: {fused:.2f}\n"
            f"Grad-CAM heatmap hotspots: {region_text}\n\n"
            f"Write exactly 2 sentences explaining why this frame is a deepfake. "
            f"Name the specific face regions from the heatmap and describe the visual "
            f"artifacts typically found there in AI-generated faces such as blending seams, "
            f"skin texture inconsistency, lighting mismatch, or hair boundary artifacts. "
            f"Be technical and specific. Do not use bullet points."
        )
    else:
        prompt = (
            f"You are a deepfake forensic analyst. Frame #{frame_id} is classified as REAL.\n"
            f"ResNet18 confidence: {cnn_conf:.2f}, CLIP confidence: {vlm_conf:.2f}, "
            f"Fused confidence: {fused:.2f}\n"
            f"Grad-CAM heatmap hotspots: {region_text}\n\n"
            f"Write exactly 2 sentences explaining why this frame appears authentic. "
            f"Mention the face regions and the natural features found there such as "
            f"consistent skin texture, coherent lighting gradient, or realistic facial geometry. "
            f"Do not use bullet points."
        )

    return call_llm(prompt)


# ================= FINAL VIDEO FORENSIC REPORT =================
def generate_video_report(votes, frame_logs, total_frames_scanned):
    """
    After processing all frames, asks local LLM to write a complete
    forensic analysis report for the entire video.
    """
    fake_logs = [f for f in frame_logs if f["label"] == "FAKE"]
    real_logs = [f for f in frame_logs if f["label"] == "REAL"]

    top_fake = sorted(fake_logs, key=lambda x: x["fused_conf"], reverse=True)[:5]
    suspicious_text = "\n".join(
        f"  Frame {f['frame_id']:04d}: fused={f['fused_conf']:.2f}, "
        f"hotspot={f['hot_regions'][0]['name']} "
        f"(intensity {f['hot_regions'][0]['score']:.2f})"
        for f in top_fake
    ) or "  None"

    region_counts = {}
    for f in fake_logs:
        name = f["hot_regions"][0]["name"]
        region_counts[name] = region_counts.get(name, 0) + 1
    top_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_region_text = ", ".join(
        f"{r} ({c} frames)" for r, c in top_regions
    ) or "N/A"

    verdict  = "FAKE" if votes["FAKE"] > votes["REAL"] else "REAL"
    fake_pct = 100 * len(fake_logs) / max(len(frame_logs), 1)

    prompt = (
        f"You are a deepfake forensic analyst writing an official report.\n\n"
        f"Frames scanned        : {total_frames_scanned}\n"
        f"Frames with faces     : {len(frame_logs)}\n"
        f"Classified as FAKE    : {len(fake_logs)} ({fake_pct:.1f}%)\n"
        f"Classified as REAL    : {len(real_logs)}\n"
        f"FAKE vote sum         : {votes['FAKE']:.2f}\n"
        f"REAL vote sum         : {votes['REAL']:.2f}\n"
        f"Final verdict         : {verdict}\n\n"
        f"Most suspicious frames:\n{suspicious_text}\n\n"
        f"Most flagged face regions in FAKE frames: {top_region_text}\n\n"
        f"Write a 5-sentence forensic report covering:\n"
        f"1. Overall verdict and confidence level\n"
        f"2. Which face regions were most consistently manipulated and what artifacts appear there\n"
        f"3. What deepfake generation method was likely used based on the artifact pattern\n"
        f"4. Whether detection confidence is high or borderline\n"
        f"5. One recommendation for further verification\n\n"
        f"Formal forensic tone. No bullet points. No headers."
    )

    return call_llm(prompt)


# ================= OVERLAY BUILDER =================
def build_overlay(face_rgb, cam, label, confidence, short_text):
    """
    Blends Grad-CAM heatmap onto face and adds label + explanation text.
    Converts heatmap BGR->RGB before blending (color channel bug fix).
    """
    face_resized = cv2.resize(face_rgb, (224, 224))
    heatmap_bgr  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay      = cv2.addWeighted(face_resized, 0.6, heatmap_rgb, 0.4, 0)

    color = (0, 210, 0) if label == "REAL" else (210, 40, 40)
    cv2.putText(overlay, f"{label}  conf={confidence:.2f}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

    words, line, lines = short_text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > 28:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)

    for i, txt in enumerate(lines[:4]):
        cv2.putText(overlay, txt,
                    (4, 196 + i * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (240, 240, 240), 1)

    return overlay


# ================= VIDEO PROCESSING LOOP =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {VIDEO_PATH}")

fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (224, 224))

votes      = {"REAL": 0.0, "FAKE": 0.0}
frame_logs = []
frame_id   = 0
faces_done = 0

print(f"\n▶ Processing : {os.path.basename(VIDEO_PATH)}")
print(f"  FPS        : {fps:.1f}")
print(f"  Sampling   : every {FRAME_STEP} frame(s)")
print(f"  Output     : {OUTPUT_VIDEO}\n")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_STEP != 0:
        frame_id += 1
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces     = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        pad    = 10
        fh, fw = frame_rgb.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
        face_rgb = frame_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            continue

        face_pil = Image.fromarray(face_rgb)

        # CNN
        with torch.no_grad():
            img_t     = transform(face_pil).unsqueeze(0).to(DEVICE)
            cnn_out   = model(img_t)
            cnn_probs = torch.softmax(cnn_out, dim=1).cpu().numpy()[0]

        # CLIP
        vlm_probs = get_vlm_probs(face_pil)

        # Weighted fusion
        final_probs = CNN_WEIGHT * cnn_probs + VLM_WEIGHT * vlm_probs
        pred_idx    = int(final_probs.argmax())
        confidence  = float(final_probs[pred_idx])

        if confidence < CONF_THRESHOLD:
            continue

        label    = LABELS[pred_idx]
        cnn_conf = float(cnn_probs[pred_idx])
        vlm_conf = float(vlm_probs[pred_idx])

        # Grad-CAM
        cam         = gradcam.generate(face_pil, pred_idx)
        hot_regions = analyse_cam_regions(cam)

        # Ollama explanation
        print(f"Frame {frame_id:04d} | {label} ({confidence:.2f}) | "
              f"hotspot: {hot_regions[0]['name']} -> generating explanation...")

        explanation = generate_frame_explanation(
            label, cnn_conf, vlm_conf, hot_regions, frame_id
        )
        print(f"  💬 {explanation}\n")

        short_caption = explanation.split(".")[0].strip() + "."

        overlay_rgb = build_overlay(face_rgb, cam, label, confidence, short_caption)
        out.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        frame_logs.append({
            "frame_id":    frame_id,
            "label":       label,
            "cnn_conf":    cnn_conf,
            "vlm_conf":    vlm_conf,
            "fused_conf":  confidence,
            "hot_regions": hot_regions,
            "explanation": explanation
        })

        votes[label] += confidence
        faces_done   += 1

    frame_id += 1

cap.release()
out.release()

# ================= FINAL VERDICT =================
final_verdict = "FAKE" if votes["FAKE"] > votes["REAL"] else "REAL"

print("=" * 60)
print(f"🎥  FINAL VERDICT   : {final_verdict}")
print(f"📊  Votes -> REAL   : {votes['REAL']:.2f}  |  FAKE: {votes['FAKE']:.2f}")
print(f"👤  Faces processed : {faces_done}")
print(f"🎬  Output saved    : {OUTPUT_VIDEO}")
print("=" * 60)

# ================= FULL FORENSIC REPORT =================
if frame_logs:
    print("\n📝 Generating full forensic report...\n")

    report_body = generate_video_report(votes, frame_logs, frame_id)

    print("=" * 60)
    print("  DEEPFAKE FORENSIC REPORT")
    print("=" * 60)
    print(report_body)
    print("=" * 60)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("DEEPFAKE DETECTION - FORENSIC REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Video           : {VIDEO_PATH}\n")
        f.write(f"Verdict         : {final_verdict}\n")
        f.write(f"FAKE vote sum   : {votes['FAKE']:.2f}\n")
        f.write(f"REAL vote sum   : {votes['REAL']:.2f}\n")
        f.write(f"Frames scanned  : {frame_id}\n")
        f.write(f"Faces processed : {faces_done}\n\n")
        f.write("OVERALL ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(report_body + "\n\n")
        f.write("PER-FRAME ANALYSIS\n")
        f.write("-" * 60 + "\n")
        for log in frame_logs:
            f.write(
                f"\n[Frame {log['frame_id']:04d}] {log['label']}  "
                f"CNN={log['cnn_conf']:.2f}  CLIP={log['vlm_conf']:.2f}  "
                f"Fused={log['fused_conf']:.2f}\n"
                f"  Hotspot  : {log['hot_regions'][0]['name']} "
                f"(score {log['hot_regions'][0]['score']:.2f})\n"
                f"  Analysis : {log['explanation']}\n"
            )

    print(f"\n📄 Full report saved -> {REPORT_PATH}")

else:
    print("\n⚠️  No faces detected in the video.")
    print("    Try reducing CONF_THRESHOLD or FRAME_STEP at the top of the script.")
