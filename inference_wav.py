import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
# from unet2 import Model
# from unet_att import Model
import glob
import time


def main():
    # --- 添加参数量和 FLOPs 统计 ---
    # 构造示例输入 (这里我们假设你的 UNet 模型的输入是 img_concat_T 和 audio_feat)
    # 注意力机制下需要微调
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from torchsummary import summary

    parser = argparse.ArgumentParser(description='Train',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--dataset', type=str, default="")  
    parser.add_argument('--audio_feat', type=str, default="")
    parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
    parser.add_argument('--checkpoint', type=str, default="")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    save_path = args.save_path
    dataset_dir = args.dataset

    mode = args.asr
    net = Model(6, mode).cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()
    
    # 使用 torchsummary 统计
    if mode == 'hubert':
        summary(net, [(6, 160, 160), (16, 24, 32)])
    else:
        summary(net, [(6, 160, 160), (256, 16, 32)])
        
    example_img_concat = torch.randn(1, 6, 160, 160).cuda()  # 假设输入图像大小是 160x160
    if mode == "hubert":
        example_audio_feat = torch.randn(1, 16, 24, 32).cuda()
    if mode == "wenet":
        example_audio_feat = torch.randn(1, 256, 16, 32).cuda()

    # 创建 FlopCountAnalysis 对象
    flops = FlopCountAnalysis(net, (example_img_concat, example_audio_feat))
    # 打印总 FLOPs 数
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    # 打印每个操作的 FLOPs 数（可选）
    print(flop_count_table(flops))
    # --- 统计结束 ---
    
    if os.path.isdir(args.audio_feat):
        wavs_ = glob.glob(args.audio_feat + "/*.wav")
    else:
        wavs_ = [args.audio_feat]


    for audio_feat_path in wavs_:
        if 'male' in audio_feat_path:
            continue

        def get_audio_features(features, index):
            left = index - 4
            right = index + 4
            pad_left = 0
            pad_right = 0
            if left < 0:
                pad_left = -left
                left = 0
            if right > features.shape[0]:
                pad_right = right - features.shape[0]
                right = features.shape[0]
            auds = torch.from_numpy(features[left:right])

            if pad_left > 0:
                auds = torch.cat([torch.zeros((pad_left, auds.shape[1], auds.shape[2])), auds], dim=0)
            if pad_right > 0:
                auds = torch.cat([auds, torch.zeros((pad_right, auds.shape[1], auds.shape[2]))], dim=0)

            return auds

        # use command line
        if audio_feat_path.endswith(".npy"):
            audio_feats = np.load(audio_feat_path)
        elif audio_feat_path.endswith(".wav") or audio_feat_path.endswith(".pcm"):
            if not os.path.exists(audio_feat_path.replace(".wav", "_hu_tiny.npy")):
                import sys
                sys.path.append("..")
                cmd_line = "python data_utils/hubert.py --wav " + audio_feat_path
                os.system(cmd_line)
                audio_feats = np.load(audio_feat_path.replace(".wav", "_hu_tiny.npy"))
                save_path = audio_feat_path.replace(".wav", ".mp4")
            else:
                audio_feats = np.load(audio_feat_path.replace(".wav", "_hu_tiny.npy"))
                save_path = audio_feat_path.replace(".wav", ".mp4")
        
        # remove video
        if os.path.exists(save_path):
            os.remove(save_path)

        img_dir = os.path.join(dataset_dir, "full_body_img/")
        lms_dir = os.path.join(dataset_dir, "landmarks/")
        len_img = len(os.listdir(img_dir)) - 1
        exm_img = cv2.imread(img_dir+"0.jpg")
        h, w = exm_img.shape[:2]

        if mode=="hubert":
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
        if mode=="wenet":
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))

        step_stride = 0
        img_idx = 0

        for i in tqdm(range(audio_feats.shape[0])):
            if img_idx>len_img - 1:
                step_stride = -1
            if img_idx<1:
                step_stride = 1
            img_idx += step_stride
            img_path = img_dir + str(img_idx)+'.jpg'
            lms_path = lms_dir + str(img_idx)+'.lms'

            img = cv2.imread(img_path)
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

            crop_img = img[ymin - width // 2:ymax, xmin:xmax]
            h, w = crop_img.shape[:2]
            crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
            crop_img_ori = crop_img.copy()
            img_real_ex = crop_img[4:164, 4:164].copy()
            img_real_ex_ori = img_real_ex.copy()
            img_masked = cv2.rectangle(img_real_ex_ori,(10,50,140,100),(0,0,0),-1)
            
            img_masked = img_masked.transpose(2,0,1).astype(np.float32)
            img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
            
            img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
            img_masked_T = torch.from_numpy(img_masked / 255.0)
            img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]

            audio_feat = get_audio_features(audio_feats, i)
            if mode=="hubert":
                audio_feat = audio_feat.reshape(16,24,32)
            if mode=="wenet":
                audio_feat = audio_feat.reshape(256,16,32)

            audio_feat = audio_feat[None]
            audio_feat = audio_feat.cuda()
            img_concat_T = img_concat_T.cuda()

            start_time = time.time()
            export_onnx = False
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
                if export_onnx:
                    print('exporting to onnx')
                    torch.onnx.export(net, 
                                (img_concat_T, audio_feat), 
                                "./net.onnx",
                                export_params=True,
                                #   verbose=True, 
                                opset_version=16,
                                input_names=["images","audio_feat"], 
                                output_names=["output"])
                    print('exporting to onnx done')
                    exit()

            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.4f} seconds")

            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            crop_img_ori[4:164, 4:164] = pred
            crop_img_ori = cv2.resize(crop_img_ori, (w, h))
            img[ymin - width // 2:ymax, xmin:xmax] = crop_img_ori
            video_writer.write(img)

        video_writer.release()

        # merge video and audio
        print("Merging video and audio...")
        merge_cmd = f"ffmpeg -i {save_path} -i {audio_feat_path} -c:v libx264 -c:a aac -preset veryfast -threads 16 {save_path.replace('.mp4', '_audio.mp4')}"
        os.system(merge_cmd)

        # break

    print("Done!")


if __name__ == "__main__":
    main()
