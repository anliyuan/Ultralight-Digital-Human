import os
import cv2
import argparse
import numpy as np

def extract_audio(path, out_path, sample_rate=16000):
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')

def extract_images(path, mode, full_body_dir):
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)

    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if mode == "hubert" and fps != 25:
        raise ValueError("Using hubert, your video fps should be 25!!!")
    if mode == "wenet" and fps != 20:
        raise ValueError("Using wenet, your video fps should be 20!!!")

    print(f"extracting images from {path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(full_body_dir, f"{counter}.jpg"), frame)
        counter += 1

def get_audio_feature(wav_path, mode):
    print(f"extracting audio feature for {wav_path}...")
    if mode == "wenet":
        os.system(f"python wenet_infer.py {wav_path}")
    if mode == "hubert":
        os.system(f"python data_utils/hubert.py --wav {wav_path}")

def get_landmark(full_img_dir, landmarks_dir):
    print(f"detecting landmarks for {landmarks_dir}...")

    from get_landmark import Landmark
    landmark = Landmark()

    for img_name in os.listdir(full_img_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0] + x1, p[1] + y1
                f.write(f"{x} {y}\n")

def is_empty_directory(path):
    return len(os.listdir(path)) == 0

def process_video(video_path, base_dir, asr_mode):
    """
    Process a single video file.
    """
    print(f"[INFO] Processing video: {video_path}")
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')
    full_img_dir = os.path.join(base_dir, 'full_body_img')

    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(full_img_dir, exist_ok=True)
    if os.path.exists(wav_path):
        print(f"[INFO] Audio file already exists: {wav_path}")
    else:
        extract_audio(video_path, wav_path)

    if not is_empty_directory(full_img_dir):
        print(f"[INFO] images already exist: {full_img_dir}")
    else:
        extract_images(video_path, asr_mode, full_img_dir)

    if not is_empty_directory(landmarks_dir):
        print(f"[INFO] Full body images already exist: {landmarks_dir}")
    else:
        get_landmark(full_img_dir, landmarks_dir)

    if os.path.exists(os.path.join(base_dir, 'aud_hu_tiny.npy')):
        print(f"[INFO] Audio feature already exists: {os.path.join(base_dir, 'aud_hu_tiny.npy')}")
    else:
        get_audio_feature(wav_path, asr_mode)

def process_folder(folder_path, asr_mode):
    """
    Process all video files in the given folder.
    """
    print(f"[INFO] Processing folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more video formats if needed
            video_path = os.path.join(folder_path, filename)
            base_dir = os.path.join(folder_path, os.path.splitext(filename)[0])
            os.makedirs(base_dir, exist_ok=True)
            try:
                process_video(video_path, base_dir, asr_mode)
            except Exception as e:
                print(f"[ERROR] Failed to process video: {video_path} \nReason: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file or folder")
    parser.add_argument('--asr', type=str, default='hubert', help="wenet or hubert")
    opt = parser.parse_args()
    asr_mode = opt.asr

    if os.path.isfile(opt.path):
        # If the input is a single video file
        base_dir = os.path.dirname(opt.path)
        process_video(opt.path, base_dir, asr_mode)
    elif os.path.isdir(opt.path):
        # If the input is a folder containing multiple videos
        process_folder(opt.path, asr_mode)
    else:
        print("[ERROR] Invalid input path. Please provide a valid video file or folder.")
