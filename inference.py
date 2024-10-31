import argparse
import os
import time
import cv2
import librosa
import onnxruntime
import torch
import soundfile as sf
import numpy as np
import torch.nn as nn
import yaml

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_utils.hubert import get_hubert_from_16k_speech, make_even_first_dim
from data_utils.wenet_infer import ASR_Model
from models.unet import Model
# from unet2 import Model
# from unet_att import Model


def parse_args():
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--wav', type=str, default="")  # end with .wav please
    parser.add_argument('--save_path', type=str, default="")  # end with .mp4 please
    parser.add_argument('--checkpoint', type=str, default="")
    return parser.parse_args()


def get_audio_features(features, index):
    left = max(index - 8, 0)
    right = min(index + 8, features.shape[0])
    audio_feat = torch.from_numpy(features[left:right])
    if left < index:
        audio_feat = torch.cat([torch.zeros_like(audio_feat[:index - left]), audio_feat], dim=0)
    if right > index:
        audio_feat = torch.cat([audio_feat, torch.zeros_like(audio_feat[right - index:])], dim=0)  # [8, 16]
    return audio_feat


def load_image_and_landmarks(img_dir, lms_dir, img_idx):
    img_path = os.path.join(img_dir, str(img_idx) + '.jpg')
    lms_path = os.path.join(lms_dir, str(img_idx) + '.lms')
    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)
    return img, lms


def process_image_and_landmarks(img, lms):
    xmin = lms[1][0]
    ymin = lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    return crop_img, h, w, xmin, ymin, xmax, ymax


def create_video_writer(save_path, mode, w, h):
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用更通用的编码器
    if mode == "hubert":
        return cv2.VideoWriter(save_path, fourcc, 25, (w, h))
    elif mode == "wenet":
        return cv2.VideoWriter(save_path, fourcc, 20, (w, h))
    else:
        raise ValueError("Invalid mode")


def process_wav_file_hubert(wav_name):
    speech, sr = sf.read(wav_name)
    speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    print("采样率: {} 至 {}".format(sr, 16000))

    hubert_hidden = get_hubert_from_16k_speech(speech_16k)
    hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
    np.save(wav_name.replace('.wav', '_hu.npy'), hubert_hidden.detach().numpy())
    print(hubert_hidden.detach().numpy().shape)


def process_wav_file_wenet(audio_path):

    with open('data_utils/conf/decode_engine_V4.yaml', 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    asr = ASR_Model(configs)

    stream, sample_rate = sf.read(audio_path)  # [T*sample_rate,] float64

    # stream = stream[:,0]
    waveform = stream.astype(np.float32) * 32767
    waveform = waveform.astype(np.int16)
    empty_audio_30 = np.zeros([32 * 160])
    empty_audio_31 = np.zeros([35 * 160])
    waveform = np.concatenate([empty_audio_30, waveform, empty_audio_31], axis=0)
    wav_duration = len(waveform) / 16000  # self.configs['engine_sample_rate_hertz']
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    print("waveform shape", waveform.shape)

    t1 = time.time()
    waveform_feat, feat_length = asr.feat_pipeline._extract_feature(waveform)
    print(waveform_feat.shape)
    t2 = time.time()
    print(t2 - t1)
    # asd
    waveform_feat = waveform_feat.numpy()
    # print(waveform_feat.size())
    # asd
    # assert 0==1

    encoder_model_path = "encoder.onnx"
    ort_encoder_session = onnxruntime.InferenceSession(encoder_model_path)

    offset = np.ones((1,), dtype=np.int64) * 100
    att_cache = np.zeros([3, 8, 16, 128], dtype=np.float32)  #: Optional[torch.Tensor] = None
    cnn_cache = np.zeros([3, 1, 512, 14], dtype=np.float32)  #: Optional[torch.Tensor] = None

    aud_npy = []
    count = 1  # 0
    start = 0
    end = 0
    frames_stride = 67
    t1 = time.time()
    while end < feat_length:
        end = start + frames_stride
        # print(count)
        feat = waveform_feat[:, start:end, :]
        feat_len = feat.shape[1]
        # print(feat_len)

        if feat_len < frames_stride:
            zero_pad = np.zeros([1, frames_stride - feat_len, 80])
            feat = np.concatenate((feat, zero_pad), axis=1)
        chunk_feat = np.expand_dims(feat, axis=0)
        # chunk_feat = feat
        start += 5
        # start = end
        # print('输入chunk:',chunk_feat.shape)

        # print(chunk_feat.dtype, offset.dtype, att_cache.dtype, cnn_cache.dtype)
        # assert 0==1
        # t1 = time.time()
        # ort_encoder_inputs = {'speech': chunk_feat.astype(np.float32), 'speech_lengths':np.array([10], dtype=np.int32)}
        ort_encoder_inputs = {'chunk': chunk_feat.astype(np.float32), 'offset': offset,
                              'att_cache': att_cache.astype(np.float32), 'cnn_cache': cnn_cache.astype(np.float32)}
        ort_encoder_outs = ort_encoder_session.run(None, ort_encoder_inputs)
        y = ort_encoder_outs[0][0]
        aud_npy.append(y)
    aud_npy = np.array(aud_npy, dtype=np.float32)

    print(aud_npy.shape)
    t2 = time.time()
    print(t2 - t1)
    np.save(audio_path.replace(".wav", "_wenet.npy"), aud_npy)


def process_wav_file(wav_name, mode):
    if mode == "hubert":
        process_wav_file_hubert(wav_name)
    elif mode == "wenet":
        process_wav_file_wenet(wav_name)
    else:
        raise ValueError("Invalid mode")


def is_wav_file(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() == '.wav'


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    save_path = args.save_path
    dataset_dir = args.dataset
    wav_path = args.wav
    mode = args.asr

    if mode != "hubert" and mode != "wenet":
        print("Invalid mode")
        return

    if os.path.join(wav_path) and not is_wav_file(wav_path):
        print("Invalid wav file")
        return

    process_wav_file(wav_path, mode)

    if mode == "hubert":
        audio_feat_path = wav_path.replace('.wav', '_hu.npy')
    elif mode == "wenet":
        audio_feat_path = wav_path.replace('.wav', '_wenet.npy')

    if not os.path.exists(audio_feat_path):
        print("Audio feature not found")
        return

    audio_feats = np.load(audio_feat_path)
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")
    len_img = len(os.listdir(img_dir)) - 1
    exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
    h, w = exm_img.shape[:2]

    video_writer = create_video_writer(save_path, mode, w, h)
    step_stride = 0
    img_idx = 0

    net = Model(6, mode).cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()
    for i in range(audio_feats.shape[0]):
        if img_idx >= len_img - 1:
            step_stride = -1
        elif img_idx < 1:
            step_stride = 1
        img_idx += step_stride
        img, lms = load_image_and_landmarks(img_dir, lms_dir, img_idx)
        crop_img, h, w, xmin, ymin, xmax, ymax = process_image_and_landmarks(img, lms)

        crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:164, 4:164].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 150, 145), (0, 0, 0), -1)

        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]

        audio_feat = get_audio_features(audio_feats, i)
        if mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)
        if mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        audio_feat = audio_feat[None]
        audio_feat = audio_feat.cuda()
        img_concat_T = img_concat_T.cuda()

        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)[0]

        pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
        pred = np.array(pred, dtype=np.uint8)
        crop_img_ori[4:164, 4:164] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (w, h))
        img[ymin:ymax, xmin:xmax] = crop_img_ori
        video_writer.write(img)
    video_writer.release()


if __name__ == "__main__":
    main()
    # ffmpeg -i xxx.mp4 -i your_audio.wav -c:v libx264 -c:a aac result_test.mp4
