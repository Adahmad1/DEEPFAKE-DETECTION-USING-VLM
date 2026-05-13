"""
Deepfake Detection - FastAPI Backend
=====================================
Run:
  pip install fastapi uvicorn torch torchvision transformers opencv-python Pillow requests
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import shutil
import hashlib
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
import requests

# =====================================================================
#  CONFIG
# =====================================================================
UPLOAD_DIR   = "uploads"
OUTPUT_DIR   = "outputs"
MODEL_PATH   = "models/best_model.pth"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS         = {0: "FAKE", 1: "REAL"}
CNN_WEIGHT     = 0.75
VLM_WEIGHT     = 0.25
CONF_THRESHOLD = 0.6
FRAME_STEP     = 5
FACE_PAD       = 10

print(f"Device: {DEVICE}")

# =====================================================================
#  LOAD MODELS  (once at startup)
# =====================================================================
print("Loading CNN...")
cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, 2)
cnn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
cnn.to(DEVICE)
cnn.eval()
print("✅ CNN loaded")

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("✅ CLIP loaded")

# index 0 = FAKE, index 1 = REAL  (must match CNN label order)
CLIP_TEXTS = [
    "a deepfake face with blending artifacts, inconsistent lighting, or unnatural skin texture",
    "a real human face with natural lighting and realistic texture"
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================================
#  LLM CACHE
#  Same label + hotspot + confidence bucket → return cached answer
#  Avoids redundant Ollama calls for similar frames
# =====================================================================
_llm_cache: dict = {}

def _cache_key(label, hot_regions, fused_conf):
    top    = hot_regions[0]["name"] if hot_regions else "unknown"
    bucket = round(fused_conf, 1)
    return hashlib.md5(f"{label}|{top}|{bucket}".encode()).hexdigest()


# =====================================================================
#  GRAD-CAM
# =====================================================================
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
        img_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(img_tensor)
        output[0, class_idx].backward()

        grads = self.gradients[0].cpu().numpy()
        acts  = self.activations[0].cpu().numpy()

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

gradcam = GradCAM(cnn, cnn.layer4[-1])


# =====================================================================
#  HELPERS
# =====================================================================
def get_vlm_probs(face_pil):
    """Returns np.array([P(FAKE), P(REAL)]) aligned with CNN label order."""
    with torch.no_grad():
        inputs  = clip_proc(
            text=CLIP_TEXTS,
            images=face_pil,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        outputs = clip_model(**inputs)
        probs   = outputs.logits_per_image.softmax(dim=1)
    return probs[0].float().cpu().numpy()


def analyse_cam_regions(cam):
    """
    Divides 224x224 CAM into 3x3 grid.
    Returns top-3 hottest face regions with name + intensity score.
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


def call_llm(prompt):
    """Calls local Ollama. Returns response text or error string."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Run: ollama serve"
    except Exception as e:
        return f"LLM error: {str(e)}"


def generate_frame_explanation(label, cnn_conf, vlm_conf, hot_regions, frame_id):
    """
    Detailed per-frame prompt with real confidence + hotspot context.
    Cached — identical pattern returns instantly without calling Ollama.
    """
    fused = CNN_WEIGHT * cnn_conf + VLM_WEIGHT * vlm_conf
    key   = _cache_key(label, hot_regions, fused)

    if key in _llm_cache:
        return _llm_cache[key]

    region_text = ", ".join(
        f"{r['name']} region (intensity {r['score']:.2f})" for r in hot_regions
    )

    if label == "FAKE":
        prompt = (
            f"You are a deepfake forensic analyst. Frame #{frame_id} is classified as FAKE.\n"
            f"ResNet18 confidence: {cnn_conf:.2f}, CLIP confidence: {vlm_conf:.2f}, "
            f"Fused confidence: {fused:.2f}\n"
            f"Grad-CAM heatmap hotspots: {region_text}\n\n"
            f"Write exactly 2 sentences explaining why this frame is a deepfake. "
            f"Name the specific face regions and describe visual artifacts there "
            f"such as blending seams, skin texture inconsistency, lighting mismatch, "
            f"or hair boundary artifacts. Be technical. No bullet points."
        )
    else:
        prompt = (
            f"You are a deepfake forensic analyst. Frame #{frame_id} is classified as REAL.\n"
            f"ResNet18 confidence: {cnn_conf:.2f}, CLIP confidence: {vlm_conf:.2f}, "
            f"Fused confidence: {fused:.2f}\n"
            f"Grad-CAM heatmap hotspots: {region_text}\n\n"
            f"Write exactly 2 sentences explaining why this frame appears authentic. "
            f"Mention the face regions and natural features found there such as "
            f"consistent skin texture, coherent lighting, or realistic facial geometry. "
            f"No bullet points."
        )

    result = call_llm(prompt)
    _llm_cache[key] = result
    return result


def generate_video_report(votes, frame_logs, total_frames):
    """Full forensic report for the entire video — called once at the end."""
    fake_logs = [f for f in frame_logs if f["label"] == "FAKE"]
    real_logs = [f for f in frame_logs if f["label"] == "REAL"]

    top_fake = sorted(fake_logs, key=lambda x: x["fused_conf"], reverse=True)[:5]
    suspicious_text = "\n".join(
        f"  Frame {f['frame_id']:04d}: fused={f['fused_conf']:.2f}, "
        f"hotspot={f['hot_regions'][0]['name'] if f['hot_regions'] else 'N/A'}"
        for f in top_fake
    ) or "  None"

    region_counts: dict = {}
    for f in fake_logs:
        if f["hot_regions"]:
            name = f["hot_regions"][0]["name"]
            region_counts[name] = region_counts.get(name, 0) + 1
    top_regions     = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_region_text = ", ".join(f"{r} ({c} frames)" for r, c in top_regions) or "N/A"

    verdict  = "FAKE" if votes["FAKE"] > votes["REAL"] else "REAL"
    fake_pct = 100 * len(fake_logs) / max(len(frame_logs), 1)

    prompt = (
        f"You are a deepfake forensic analyst writing an official report.\n\n"
        f"Frames scanned     : {total_frames}\n"
        f"Faces detected     : {len(frame_logs)}\n"
        f"FAKE frames        : {len(fake_logs)} ({fake_pct:.1f}%)\n"
        f"REAL frames        : {len(real_logs)}\n"
        f"FAKE vote sum      : {votes['FAKE']:.2f}\n"
        f"REAL vote sum      : {votes['REAL']:.2f}\n"
        f"Final verdict      : {verdict}\n\n"
        f"Most suspicious frames:\n{suspicious_text}\n\n"
        f"Most flagged face regions: {top_region_text}\n\n"
        f"Write a 5-sentence forensic report covering:\n"
        f"1. Overall verdict and confidence level\n"
        f"2. Which face regions were most consistently manipulated\n"
        f"3. What deepfake generation method was likely used\n"
        f"4. Whether detection confidence is high or borderline\n"
        f"5. One recommendation for further verification\n\n"
        f"Formal forensic tone. No bullet points. No headers."
    )
    return call_llm(prompt)


def build_overlay(face_rgb, cam, label, confidence, short_text):
    """Heatmap blended onto face with label + short caption."""
    face_resized = cv2.resize(face_rgb, (224, 224))
    heatmap_bgr  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)   # color fix
    overlay      = cv2.addWeighted(face_resized, 0.6, heatmap_rgb, 0.4, 0)

    color = (0, 210, 0) if label == "REAL" else (210, 40, 40)
    cv2.putText(overlay, f"{label}  {confidence:.2f}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    words, line, lines = short_text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > 30:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    for i, txt in enumerate(lines[:4]):
        cv2.putText(overlay, txt, (4, 196 + i * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (240, 240, 240), 1)

    return overlay   # RGB


# =====================================================================
#  MAIN VIDEO PIPELINE
# =====================================================================
def process_video(input_path: str, output_video: str, report_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_video, fourcc, fps, (224, 224))

    votes      = {"REAL": 0.0, "FAKE": 0.0}
    frame_logs = []
    frame_id   = 0

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
            fh, fw = frame_rgb.shape[:2]
            x1 = max(0, x - FACE_PAD)
            y1 = max(0, y - FACE_PAD)
            x2 = min(fw, x + w + FACE_PAD)
            y2 = min(fh, y + h + FACE_PAD)
            face_rgb = frame_rgb[y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue

            face_pil = Image.fromarray(face_rgb)

            # CNN
            with torch.no_grad():
                img_t     = transform(face_pil).unsqueeze(0).to(DEVICE)
                cnn_probs = torch.softmax(cnn(img_t), dim=1).cpu().numpy()[0]

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

            # Grad-CAM + region analysis
            cam         = gradcam.generate(face_pil, pred_idx)
            hot_regions = analyse_cam_regions(cam)

            # LLM explanation (cached for repeated patterns)
            explanation = generate_frame_explanation(
                label, cnn_conf, vlm_conf, hot_regions, frame_id
            )

            print(f"Frame {frame_id:04d} | {label} ({confidence:.2f}) | "
                  f"hotspot: {hot_regions[0]['name']}")

            short_caption = explanation.split(".")[0].strip() + "."
            overlay_rgb   = build_overlay(face_rgb, cam, label, confidence, short_caption)
            out.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            votes[label] += confidence
            frame_logs.append({
                "frame_id":    frame_id,
                "label":       label,
                "cnn_conf":    round(cnn_conf, 4),
                "vlm_conf":    round(vlm_conf, 4),
                "fused_conf":  round(confidence, 4),
                "hot_regions": hot_regions,
                "explanation": explanation
            })

        frame_id += 1

    cap.release()
    out.release()

    verdict        = "FAKE" if votes["FAKE"] > votes["REAL"] else "REAL"
    overall_report = generate_video_report(votes, frame_logs, frame_id)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DEEPFAKE DETECTION - FORENSIC REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Verdict         : {verdict}\n")
        f.write(f"FAKE vote sum   : {votes['FAKE']:.2f}\n")
        f.write(f"REAL vote sum   : {votes['REAL']:.2f}\n")
        f.write(f"Frames scanned  : {frame_id}\n")
        f.write(f"Faces processed : {len(frame_logs)}\n\n")
        f.write("OVERALL ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(overall_report + "\n\n")
        f.write("PER-FRAME ANALYSIS\n")
        f.write("-" * 60 + "\n")
        for log in frame_logs:
            f.write(
                f"\n[Frame {log['frame_id']:04d}] {log['label']}  "
                f"CNN={log['cnn_conf']:.2f}  CLIP={log['vlm_conf']:.2f}  "
                f"Fused={log['fused_conf']:.2f}\n"
            )
            if log["hot_regions"]:
                f.write(
                    f"  Hotspot  : {log['hot_regions'][0]['name']} "
                    f"(score {log['hot_regions'][0]['score']:.2f})\n"
                )
            f.write(f"  Analysis : {log['explanation']}\n")

    return verdict, votes, frame_logs, frame_id, len(frame_logs), overall_report


# =====================================================================
#  FASTAPI
# =====================================================================
app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Quick check — lets the frontend verify the server is ready."""
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files accepted.")

    file_id      = str(uuid.uuid4())
    input_path   = f"{UPLOAD_DIR}/{file_id}.mp4"
    output_video = f"{OUTPUT_DIR}/{file_id}_out.mp4"
    report_path  = f"{OUTPUT_DIR}/{file_id}.txt"

    with open(input_path, "wb") as buf:
        shutil.copyfileobj(video.file, buf)

    try:
        verdict, votes, frame_logs, frames_scanned, faces_done, overall_report = \
            process_video(input_path, output_video, report_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    return {
        "file_id":         file_id,
        "verdict":         verdict,
        "votes":           votes,
        "faces_processed": faces_done,
        "frames_scanned":  frames_scanned,
        "overall_report":  overall_report,
        "report":          report_text,
        "frame_logs":      frame_logs
    }


@app.get("/output/{file_id}")
def get_output_video(file_id: str):
    """Stream the annotated heatmap video to the frontend."""
    path = f"{OUTPUT_DIR}/{file_id}_out.mp4"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output video not found.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/report/{file_id}")
def get_report(file_id: str):
    """Download the forensic text report."""
    path = f"{OUTPUT_DIR}/{file_id}.txt"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/plain",
                        filename="deepfake_report.txt")
