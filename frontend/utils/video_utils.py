import cv2
import os

def extract_frames(video_path, output_dir="data/frames", fps=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = f"{output_dir}/frame_{saved}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved += 1

        count += 1

    cap.release()
    return frames
