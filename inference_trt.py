import argparse
import os
import cv2
import torch
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda

parser = argparse.ArgumentParser(
    description="Train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--asr", type=str, default="hubert")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--audio_feat", type=str, default="")
parser.add_argument("--save_path", type=str, default="")  # end with .mp4 please
parser.add_argument("--checkpoint", type=str, default="")
args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
dataset_dir = args.dataset
audio_feat_path = args.audio_feat
mode = args.asr

device = "cuda" if torch.cuda.is_available() else "cpu"

class UnetTRT:

    def __init__(self, engine_path: str):
        import pycuda.autoinit

        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
                self.input_shapes[tensor_name] = tensor_shape
            else:
                self.output_names.append(tensor_name)
                self.output_shapes[tensor_name] = tensor_shape
        self.bindings = []
        self.tensor_device_memory = {}
        for name in self.input_names + self.output_names:
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            self.tensor_device_memory[name] = device_mem
        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, self.tensor_device_memory[name])
        self.stream = cuda.Stream()
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()
        print(
            f"Initialized TensorRT 10.x model with inputs: {self.input_names}, outputs: {self.output_names}"
        )

    def __call__(self, img, audio_feat, *args, **kwds):
        try:
            input_buffers = {
                self.input_names[0]: self._prepare_input(img, self.input_names[0]),
                self.input_names[1]: self._prepare_input(
                    audio_feat, self.input_names[1]
                ),
            }
            output_host = np.empty(
                self.output_shapes[self.output_names[0]],
                dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_names[0])),
            )

            if self.stream is None:
                raise RuntimeError("CUDA stream is not initialized")

            self.start_event.record(self.stream)
            self._async_inference(input_buffers, output_host)
            self.end_event.record(self.stream)
            self.end_event.synchronize()
            return output_host
        except Exception as e:
            print(f"Inference error: {str(e)}")
            raise

    def __del__(self):
        if hasattr(self, "stream") and self.stream is not None:
            self.stream.synchronize()
            del self.stream

    def _prepare_input(self, data, input_name):
        """准备输入数据，确保连续内存和正确类型"""
        return np.ascontiguousarray(
            data, dtype=trt.nptype(self.engine.get_tensor_dtype(input_name))
        )

    def _async_inference(self, input_buffers, output_host):
        """执行异步推理操作"""
        for name in self.input_names:
            cuda.memcpy_htod_async(
                self.tensor_device_memory[name],
                input_buffers[name].ravel(),
                self.stream,
            )
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(
            output_host, self.tensor_device_memory[self.output_names[0]], self.stream
        )


def get_audio_features(features, index):  # 这个逻辑跟datasets里面的逻辑相同
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
    auds = features[left:right].copy()  # Ensure we get a copy, not a view
    if pad_left > 0:
        auds = np.concatenate([np.zeros_like(auds[:pad_left]), auds], axis=0)
    if pad_right > 0:
        auds = np.concatenate([auds, np.zeros_like(auds[:pad_right])], axis=0)  # [8, 16]
    return auds


audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir + "0.jpg")
h, w = exm_img.shape[:2]

if mode == "hubert":
    video_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 25, (w, h)
    )
if mode == "wenet":
    video_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 20, (w, h)
    )
step_stride = 0
img_idx = 0


unet = UnetTRT(checkpoint)

import time

s0 = time.time()

for i in range(audio_feats.shape[0]):
    if img_idx > len_img - 1:
        step_stride = (
            -1
        )  # step_stride 决定取图片的间隔，目前这个逻辑是从头开始一张一张往后，到最后一张后再一张一张往前
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride
    img_path = img_dir + str(img_idx) + ".jpg"
    lms_path = lms_dir + str(img_idx) + ".lms"

    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)  # 这个关键点检测模型之后之后可能会改掉
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 150, 145), (0, 0, 0), -1)

    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)
    img_real_ex_T = (img_real_ex / 255.0).astype(np.float32)
    img_masked_T = (img_masked / 255.0).astype(np.float32)
    img_concat_T = np.concatenate([img_real_ex_T, img_masked_T], axis=0)[np.newaxis]

    audio_feat = get_audio_features(audio_feats, i)
    if mode == "hubert":
        audio_feat = audio_feat.reshape(16, 32, 32)
    if mode == "wenet":
        audio_feat = audio_feat.reshape(128, 16, 32)
    audio_feat = audio_feat[None]
    
    output_host = unet(img_concat_T, audio_feat)
    pred = np.squeeze(output_host, 0).transpose(1, 2, 0) * 255.0

    pred = np.array(pred, dtype=np.uint8)
    crop_img_ori[4:164, 4:164] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    img[ymin:ymax, xmin:xmax] = crop_img_ori
#    video_writer.write(img)
#video_writer.release()

print(audio_feats.shape[0] / (time.time() - s0))
print(time.time() - s0)
# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
