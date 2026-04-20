import os
import cv2
import argparse
import numpy as np
from landmark_io import save_landmarks_file

def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path, mode):
    
    
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)
    
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if mode == "hubert" and fps != 25:
        raise ValueError("Using hubert,your video fps should be 25!!!")
    if mode == "wenet" and fps != 20:
        raise ValueError("Using wenet,your video fps should be 20!!!")
        
    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(full_body_dir+"/"+str(counter)+'.jpg', frame)
        counter += 1
        
def get_audio_feature(wav_path, mode):
    
    print("extracting audio feature...")
    
    if mode == "wenet":
        os.system("python wenet_infer.py "+wav_path)
    if mode == "hubert":
        os.system("python hubert.py --wav "+wav_path)
    
def get_landmark(path, landmarks_dir, backend_name):
    print("detecting landmarks...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")
    
    from get_landmark import create_landmark_backend
    landmark = create_landmark_backend(backend_name)
    
    for img_name in os.listdir(full_img_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        absolute_landmarks = np.array(
            [[p[0] + x1, p[1] + y1] for p in pre_landmark], dtype=np.int32
        )
        save_landmarks_file(lms_path, absolute_landmarks, landmark.backend_name)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--asr', type=str, default='hubert', help="wenet or hubert")
    parser.add_argument(
        '--landmark_backend',
        type=str,
        default='pfld',
        choices=['pfld', 'mediapipe'],
        help="landmark extraction backend",
    )
    opt = parser.parse_args()
    asr_mode = opt.asr

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path, asr_mode)
    get_landmark(opt.path, landmarks_dir, opt.landmark_backend)
    get_audio_feature(wav_path, asr_mode)
    
    
