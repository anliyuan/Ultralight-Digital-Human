import os
import sys
import subprocess
from pathlib import Path
import cv2
import argparse
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def run_command(cmd):
    print(f"[INFO] Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def validate_fps(fps, mode):
    expected_fps = {"hubert": 25.0, "wenet": 20.0}.get(mode)
    if expected_fps is None:
        raise ValueError(f"Unsupported ASR mode: {mode}")
    if fps <= 0:
        raise ValueError(f"Failed to read video fps for mode {mode}.")
    if abs(fps - expected_fps) > 0.5:
        raise ValueError(f"Using {mode}, your video fps should be close to {expected_fps}, but got {fps:.3f}.")

def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    run_command(['ffmpeg', '-y', '-i', path, '-f', 'wav', '-ar', str(sample_rate), out_path])
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path, mode):
    
    
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    os.makedirs(full_body_dir, exist_ok=True)
    
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    validate_fps(fps, mode)
        
    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(full_body_dir+"/"+str(counter)+'.jpg', frame)
        counter += 1
    cap.release()
        
def get_audio_feature(wav_path, mode):
    
    print("extracting audio feature...")
    
    if mode == "wenet":
        run_command([sys.executable, str(BASE_DIR / "wenet_infer.py"), wav_path])
    elif mode == "hubert":
        run_command([sys.executable, str(BASE_DIR / "hubert.py"), '--wav', wav_path])
    else:
        raise ValueError(f"Unsupported ASR mode: {mode}")
    
def get_landmark(path, landmarks_dir):
    print("detecting landmarks...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")
    
    from get_landmark import Landmark
    landmark = Landmark()
    
    for img_name in os.listdir(full_img_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        det_result = landmark.detect(img_path)
        if det_result is None:
            print(f"[WARN] Skip landmark generation for {img_name} because face detection failed.")
            continue
        pre_landmark, x1, y1 = det_result
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0]+x1, p[1]+y1
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--asr', type=str, default='hubert', help="wenet or hubert")
    opt = parser.parse_args()
    asr_mode = opt.asr

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path, asr_mode)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path, asr_mode)
    
    
