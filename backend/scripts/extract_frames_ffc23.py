import cv2
import os

def extract_frames(video_folder, output_folder, step=30):
    """
    Extract frames from all videos in a folder.
    step=30 -> take every 30th frame
    """
    os.makedirs(output_folder, exist_ok=True)

    videos = [v for v in os.listdir(video_folder) if v.endswith(".mp4")]
    total_videos = len(videos)
    print(f"Found {total_videos} videos in {video_folder}")

    count_total = 0
    for i, video_file in enumerate(videos):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        saved = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                filename = f"{os.path.splitext(video_file)[0]}_{saved}.jpg"
                cv2.imwrite(os.path.join(output_folder, filename), frame)
                saved += 1
                count_total += 1
            frame_idx += 1
        cap.release()
        print(f"[{i+1}/{total_videos}] Extracted {saved} frames from {video_file}")

    print(f"Extraction complete. Total frames: {count_total}")

# ---------------- UPDATE THESE PATHS ----------------
REAL_VIDEOS = r"D:\personal\project 2\FaceForensics++_C23\original"
FAKE_VIDEOS = r"D:\personal\project 2\FaceForensics++_C23\Deepfakes"

OUTPUT_REAL = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\data\samples\train\real"
OUTPUT_FAKE = r"D:\personal\project 2\DEEPFAKE DETECTION USING VLM\data\samples\train\fake"

# Extract frames from both folders
extract_frames(REAL_VIDEOS, OUTPUT_REAL)
extract_frames(FAKE_VIDEOS, OUTPUT_FAKE)