import cv2
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from multiprocessing import freeze_support
freeze_support()


# --- Configuration ---
TARGET_FPS = 30
VIDEO_DIR = "archive/videos"
JSON_PATH = "archive/WLASL_v0.3.json"
NUM_WORKERS = max(1, cpu_count() - 2)  


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image, results):
    """Pose + Hands + Lips + Eyes + Face Orientation"""
    h, w, _ = image.shape
    aspect_ratio = w / h

    if results.pose_landmarks:
        ls = results.pose_landmarks.landmark[11]
        rs = results.pose_landmarks.landmark[12]
        mid_x, mid_y = (ls.x + rs.x) / 2 * aspect_ratio, (ls.y + rs.y) / 2
        shoulder_dist = np.sqrt(((ls.x - rs.x) * aspect_ratio)**2 + (ls.y - rs.y)**2) or 1.0
    else:
        mid_x, mid_y, shoulder_dist = 0.5 * aspect_ratio, 0.5, 1.0

    def normalize_point(res):
        return [
            ((res.x * aspect_ratio) - mid_x) / shoulder_dist,
            (res.y - mid_y) / shoulder_dist,
            res.z / shoulder_dist
        ]

    def normalize_subset(landmark_list, indices=None, count=None):
        if not landmark_list:
            if indices:
                return np.zeros(len(indices) * 3)
            return np.zeros(count * 3)

        coords = []
        if indices:
            for i in indices:
                coords.extend(normalize_point(landmark_list.landmark[i]))
        else:
            for res in landmark_list.landmark:
                coords.extend(normalize_point(res))

        return np.array(coords)

    # --- Lips ---
    LIP_LANDMARKS = [
        61, 17, 291, 95
    ]

    lips = normalize_subset(results.face_landmarks, indices=LIP_LANDMARKS)

    # --- Eyes ---
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    left_eye = normalize_subset(results.face_landmarks, indices=LEFT_EYE)
    right_eye = normalize_subset(results.face_landmarks, indices=RIGHT_EYE)

    # --- Face Orientation ---
    def get_face_orientation(face_landmarks):
        if not face_landmarks:
            return np.zeros(6)

        nose = face_landmarks.landmark[1]
        left = face_landmarks.landmark[234]
        right = face_landmarks.landmark[454]

        nose_p = np.array(normalize_point(nose))
        left_p = np.array(normalize_point(left))
        right_p = np.array(normalize_point(right))

        # Horizontal direction (left → right)
        horiz_vec = right_p - left_p
        horiz_vec /= (np.linalg.norm(horiz_vec) + 1e-6)

        # Vertical approximation (nose relative to midpoint)
        mid_lr = (left_p + right_p) / 2
        vert_vec = nose_p - mid_lr
        vert_vec /= (np.linalg.norm(vert_vec) + 1e-6)

        return np.concatenate([horiz_vec, vert_vec])

    face_orientation = get_face_orientation(results.face_landmarks)

    return np.concatenate([
        normalize_subset(results.pose_landmarks, count=33),
        normalize_subset(results.left_hand_landmarks, count=21),
        normalize_subset(results.right_hand_landmarks, count=21),
        lips,
        left_eye,
        right_eye,
        face_orientation
    ])

def interpolate_sequence(frames, original_fps):
    target_frames = int(len(frames) / original_fps * TARGET_FPS)
    if target_frames <= 1:
        return np.array(frames)

    cur_idx = np.arange(len(frames))
    tgt_idx = np.linspace(0, len(frames)-1, target_frames)

    output = []
    for d in range(frames[0].shape[0]):
        vals = [f[d] for f in frames]
        output.append(np.interp(tgt_idx, cur_idx, vals))

    return np.array(output).T
def process_video(args):
    video_path, gloss = args

    if not os.path.exists(video_path):
        return None

    # ⚠️ IMPORTANT: create holistic INSIDE process
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,  # faster
        refine_face_landmarks=False,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        frames.append(extract_landmarks(frame, results))

    cap.release()
    holistic.close()

    if len(frames) < 3:
        return None

    seq = interpolate_sequence(frames, fps)
    return seq, gloss

if __name__ == "__main__":

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    jobs = []
    for entry in data:
        gloss = entry['gloss']
        for inst in entry['instances']:
            path = os.path.join(VIDEO_DIR, f"{inst['video_id']}.mp4")
            jobs.append((path, gloss))

    print(f"🚀 Processing {len(jobs)} videos using {NUM_WORKERS} cores...")

    dataset_x, dataset_y = [], []

    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_video, jobs), total=len(jobs)):
            if result is None:
                continue
            x, y = result
            dataset_x.append(x)
            dataset_y.append(y)

    np.save("X_data_standardized.npy", np.array(dataset_x, dtype=object))
    np.save("y_labels.npy", np.array(dataset_y))

    print(f"\n✅ Done! Processed {len(dataset_x)} videos.")