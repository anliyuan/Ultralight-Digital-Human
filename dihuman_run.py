# -*- coding: utf-8 -*-
import onnxruntime
import numpy as np
import os
import cv2
import math
import time
import json
import socket
import logging
import struct
import argparse
import kaldi_native_fbank as knf

opts = knf.FbankOptions()
opts.frame_opts.dither = 0
opts.mel_opts.num_bins = 80
opts.frame_opts.snip_edges = False
opts.mel_opts.debug_mel = False

fbank = knf.OnlineFbank(opts)


# from audio_encoder import AudioEncoder
# from face_processor import FaceProcessor


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/data/service/", help="数据存放路径")
arg = parser.parse_args()

class DiHumanProcessor:
    def __init__(self, data_path):
        
        # 图片和关键点数据
        self.full_body_img_dir = os.path.join(data_path, "img_inference") # 图片路径
        self.lms_dir = os.path.join(data_path, "lms_inference") # 关键点路径
        self.full_body_img_list = []
        self.bbox_list = []
        ## 数据预加载⬇️
        for i in range(len(os.listdir(self.lms_dir))-1):
            full_body_img = cv2.imread(os.path.join(self.full_body_img_dir, str(i)+'.jpg'))
            self.full_body_img_list.append(full_body_img)
            lms_path = os.path.join(self.lms_dir, str(i)+'.lms')
            lms_list = []
            with open(lms_path, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            lms = np.array(lms_list, dtype=np.int32)
            xmin = lms[1][0]
            ymin = lms[52][1]
            xmax = lms[31][0]
            width = xmax - xmin
            ymax = ymin + width
            bbox = [xmin, ymin, xmax, ymax]
            self.bbox_list.append(bbox)

        # 准备wenet推理时用到的一些数据
        self.offset = np.ones((1, ), dtype=np.int64)*100
        self.att_cache = np.zeros([3,8,16,128], dtype=np.float32)
        self.cnn_cache = np.zeros([3,1,512,14], dtype=np.float32)

        # 放一定量的空音频
        self.audio_play_list = [0]*13440
        providers = ["CUDAExecutionProvider"]
        self.ort_unet_session = onnxruntime.InferenceSession(os.path.join(data_path, "unet.onnx"), providers=providers)
        self.ort_ae_session = onnxruntime.InferenceSession(os.path.join(data_path, "encoder.onnx"), providers=providers)
        # 输入到ae的音频，前面的310ms全是0
        self.audio_queue_get_feat = np.zeros([32*160],dtype=np.int16)
        
        self.index = 0
        self.step = 1
        
        # 计数器
        self.counter = 0
        self.empty_audio_counter = 56
        
        self.is_processing = False
        self.return_img = None
        
        self.silence = True
        
        self.using_feat = np.zeros([4,16,512], dtype=np.float32)
        
    def reset(self):
        
        self.audio_queue_get_feat = np.zeros([32*160],dtype=np.int16)
        self.audio_play_list = [0]*13440
        self.counter = 0
        self.is_processing = True
    
    def process(self, audio_frame):
        # try:
        audio_frame = audio_frame.astype(np.int16)
        if not np.any(audio_frame):  # 送进来全0的语音，如果连续送进来560ms的空音频，则返回静音的图片
            if not self.silence:
                self.empty_audio_counter += 1
            if self.empty_audio_counter >= 100:
                self.silence = True
        
        else:
            self.empty_audio_counter = 0 # 否则重置计数器
            self.silence = False
            
        if not self.silence:  # 如果送进来的不是全0的音频
            if not self.is_processing:  # 第一次推理重置参数
                self.reset()
            if audio_frame.shape[0]<160: # 默认送进来的是10ms的音频帧
                audio_frame = np.pad(audio_frame,(0, 160-audio_frame.shape[0]))
            self.audio_queue_get_feat = np.concatenate([self.audio_queue_get_feat, audio_frame], axis=0)  # 积攒起来，攒够一定量后开始处理
        else:  ## 如果是静音视频的话就就按顺序返回图片 同时返回空音频
            self.audio_queue_get_feat = np.array([])
            self.is_processing = False
            if self.counter == 0:
                return_img =  self.full_body_img_list[self.index].copy()
                self.index += self.step
                if self.index >= len(self.bbox_list)-1:
                    self.step = -1
                elif self.index<=0:
                    self.step = 1
                check_img = 1
                self.counter += 1
            else:
                self.return_img = None
                check_img = 0
                self.counter += 1
                if self.counter == 5:
                    self.counter = 0
                return_img = None
            playing_audio = np.zeros([160], dtype=np.int16)
            
            return return_img, playing_audio, check_img
        
        # 690ms的音频，暂时不改
        if self.audio_queue_get_feat.shape[0]>=11040:  # 攒够690 ms的音频后开始处理
            
            fbank = knf.OnlineFbank(opts)
            audio_mel_feat = []

            fbank.accept_waveform(16000, self.audio_queue_get_feat.tolist()) # fbank
            self.audio_play_list.extend(self.audio_queue_get_feat[32*160:32*160+800]) # 将正在处理的音频加到播放列表里
            for i in range(fbank.num_frames_ready):
                audio_mel_feat.append(fbank.get_frame(i))
            audio_mel_feat = np.array([[audio_mel_feat]])
            audio_mel_feat = audio_mel_feat[:,:,:67, :]
            ort_encoder_inputs = {'chunk': audio_mel_feat.astype(np.float32), 'offset':self.offset, 'att_cache':self.att_cache.astype(np.float32), 'cnn_cache':self.cnn_cache.astype(np.float32)}
            ort_encoder_outs = self.ort_ae_session.run(None, ort_encoder_inputs)  # wenet提取特征
            audio_feat = ort_encoder_outs[0]
            self.audio_queue_get_feat = self.audio_queue_get_feat[800:] # 丢弃处理过的音频
            
            self.using_feat = np.concatenate([self.using_feat, audio_feat], axis=0)  # 将音频特征积攒起来，攒够一定量开始处理
            img = self.full_body_img_list[self.index].copy()
            bbox = self.bbox_list[self.index]
            
            
            self.index += self.step
            if self.index >= len(self.bbox_list)-1:
                self.step = -1
            elif self.index<=0:
                self.step = 1
            
            if self.using_feat.shape[0]>=8: # 音频特征攒够8帧 开始输出图片 下面的逻辑和inference里的一样
                
                xmin,ymin,xmax,ymax = bbox
                crop_img = img[ymin:ymax, xmin:xmax]
                h, w = crop_img.shape[:2]
                crop_img = cv2.resize(crop_img, (168, 168))
                crop_img_ori = crop_img.copy()
                img_real_ex = crop_img[4:164, 4:164].copy()
                img_real_ex_ori = img_real_ex.copy()
                img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)

                img_masked = img_masked.transpose(2,0,1).astype(np.float32)/255.0
                img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)/255.0
                img_masked = np.expand_dims(img_masked, 0)
                img_real_ex = np.expand_dims(img_real_ex, 0)
                img_onnx_in = np.concatenate((img_real_ex, img_masked), axis=1)
                audio_feat = self.using_feat.reshape(1,128,16,32)

                ort_unet_inputs = {self.ort_unet_session.get_inputs()[0].name: img_onnx_in, self.ort_unet_session.get_inputs()[1].name: audio_feat}
                ort_outs = self.ort_unet_session.run(None, ort_unet_inputs)

                pred = ort_outs[0][0]
                pred = pred.transpose(1,2,0)*255
                pred = pred.astype(np.uint8)

                crop_img_ori[4:164, 4:164] = pred
                crop_img_ori = cv2.resize(crop_img_ori, (w, h))
                img[ymin:ymax, xmin:xmax] = crop_img_ori
                self.using_feat = self.using_feat[1:]
                
                
            # return_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            return_img = img.copy()
            t2 = time.time()
            self.counter = 1
            check_img = 1
        else: # 音频不够时仅返回播放列表里的音频 返回空图像
            if self.counter == 0:
                return_img =  self.full_body_img_list[self.index].copy()
                self.index += self.step
                if self.index >= len(self.bbox_list)-1:
                    self.step = -1
                elif self.index<=0:
                    self.step = 1
                # return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2YUV_I420)
                check_img = 1
                self.counter += 1
            else:
                return_img = None
                check_img = 0
                self.counter += 1
                if self.counter == 5:
                    self.counter = 0
        if not self.audio_play_list==[]:
            playing_audio = np.array(self.audio_play_list[:160])
            self.audio_play_list = self.audio_play_list[160:]
        else:
            playing_audio = np.zeros([160], dtype=np.int16)
            
        playing_audio = playing_audio.astype(np.int16)
        return return_img, playing_audio, check_img
        # except BaseException as e:
        #     logger.error("onTransportData error: %s", e)
            
            

        

if __name__ =="__main__":
    
    import soundfile as sf
    from scipy.io import wavfile
    stream, sample_rate = sf.read("1.wav") # [T*sample_rate,] float64
    #stream = stream[:,0]
    stream = stream.astype(np.float32)*32767 # 读音频
    stream = stream.astype(np.int16)
    video_writer = cv2.VideoWriter("./test_video.mp4", cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (1280, 720))
    audio_data = []
    processor = DiHumanProcessor("./dataset_kanghui_wenet/111/") # 初始化
    audio_stream_len = stream.shape[0]
    empty_audio = np.zeros([160])
    
    for i in range(math.ceil(audio_stream_len/160)): # 将音频拆成10ms 不断调用
        if i*160+160<audio_stream_len:
            audio_frame = stream[i*160:i*160+160]
        else:
            audio_frame = stream[i*160:]
        img, playing_audio, check_img = processor.process(audio_frame)

        # print(playing_audio)
        audio_data.append(playing_audio)
        if check_img:    
            video_writer.write(img)

    video_writer.release()
    audio_data = np.array(audio_data, dtype=np.int16)
    audio_data = audio_data.reshape(-1)
    wavfile.write('test_audio.pcm', 16000, audio_data)
    
    
# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
