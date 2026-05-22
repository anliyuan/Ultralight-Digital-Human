"""ONNX 流式推理 demo（wenet 编码器版本）。

按 10ms 一帧 PCM 喂进来，内部攒满 690ms 后做一次 wenet encoder 推理，
攒够 8 帧音频特征后开始喂给 unet 输出口型，每帧输出一张 BGR 图。
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional, Tuple

import cv2
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime


# ----- 一些写死的协议参数（不随用户输入变化） ---------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 160                # 10ms@16k
WENET_TRIGGER_LEN = 11040       # 690ms 音频长度，攒够后做一次 encoder
WENET_CHUNK_DROP = 800          # 每次 encoder 后丢弃的样本数（=50ms）
MEL_BINS = 80
WENET_FEAT_FRAMES = 67          # encoder 一次接收的 fbank 帧数
PRE_AUDIO_LEN = 32 * FRAME_LEN  # encoder 前置 padding 静音长度（320ms）
PLAY_PRE_PAD = 13440            # 播放队列前置静音，让画面和音频对齐
SILENCE_THRESHOLD = 100         # 连续 100 帧空音频后切换到静音模式
UNET_FEAT_WINDOW = 8            # unet 需要 8 帧音频特征
USING_FEAT_INIT = 4             # using_feat 初始的 0 帧数
IDLE_LOOP = 5                   # 静音状态下每 5 帧输出 1 张图像


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset_kanghui_wenet/111/",
                        help="数据目录，需要包含 img_inference/、lms_inference/、unet.onnx、encoder.onnx")
    parser.add_argument("--audio_wav", type=str, default="1.wav", help="输入测试音频")
    parser.add_argument("--out_video", type=str, default="./test_video.mp4")
    parser.add_argument("--out_audio", type=str, default="./test_audio.pcm")
    parser.add_argument("--video_size", type=int, nargs=2, default=[1280, 720],
                        help="输出视频分辨率 W H")
    return parser.parse_args()


def _make_fbank_opts() -> knf.FbankOptions:
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = MEL_BINS
    opts.mel_opts.debug_mel = False
    return opts


_FBANK_OPTS = _make_fbank_opts()


def _read_landmarks_to_bbox(lms_path: str) -> Tuple[int, int, int, int]:
    """读 .lms → 返回 (xmin, ymin, xmax, ymax)，与训练裁切口径一致。"""
    pts = []
    with open(lms_path, "r") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            pts.append(np.fromstring(line, sep=" ", dtype=np.float32))
    lms = np.array(pts, dtype=np.int32)
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    ymax = ymin + (xmax - xmin)
    return xmin, ymin, xmax, ymax


def _build_unet_inputs(crop_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从 168x168 crop 构造 unet 的 6 通道输入和回贴用 crop_ori。"""
    crop_ori = crop_img.copy()
    inner = crop_img[4:164, 4:164].copy()
    masked = cv2.rectangle(inner.copy(), (5, 5, 150, 145), (0, 0, 0), -1)

    masked = masked.transpose(2, 0, 1).astype(np.float32) / 255.0
    inner = inner.transpose(2, 0, 1).astype(np.float32) / 255.0
    onnx_in = np.concatenate(
        (np.expand_dims(inner, 0), np.expand_dims(masked, 0)),
        axis=1,
    )
    return onnx_in, crop_ori


class _BounceIndex:
    """0,1,...,N-2,N-1,N-2,...,1,0,1,... 来回索引，每次 advance() 自动转向。"""

    def __init__(self, n_frames: int):
        assert n_frames >= 2
        self.n_frames = n_frames
        self.index = 0
        self.step = 1

    def advance(self):
        self.index += self.step
        if self.index >= self.n_frames - 1:
            self.step = -1
        elif self.index <= 0:
            self.step = 1


class DiHumanProcessor:
    def __init__(self, data_path: str):
        self.full_body_img_dir = os.path.join(data_path, "img_inference")
        self.lms_dir = os.path.join(data_path, "lms_inference")

        # 预加载图像和 bbox（少一个文件，因为最后一帧通常不可信）
        self.full_body_img_list: List[np.ndarray] = []
        self.bbox_list: List[Tuple[int, int, int, int]] = []
        n_frames = len(os.listdir(self.lms_dir)) - 1
        for i in range(n_frames):
            img = cv2.imread(os.path.join(self.full_body_img_dir, f"{i}.jpg"))
            self.full_body_img_list.append(img)
            bbox = _read_landmarks_to_bbox(os.path.join(self.lms_dir, f"{i}.lms"))
            self.bbox_list.append(bbox)

        # wenet encoder cache
        self.offset = np.ones((1,), dtype=np.int64) * 100
        self.att_cache = np.zeros([3, 8, 16, 128], dtype=np.float32)
        self.cnn_cache = np.zeros([3, 1, 512, 14], dtype=np.float32)

        providers = ["CUDAExecutionProvider"]
        self.ort_unet = onnxruntime.InferenceSession(
            os.path.join(data_path, "unet.onnx"), providers=providers,
        )
        self.ort_ae = onnxruntime.InferenceSession(
            os.path.join(data_path, "encoder.onnx"), providers=providers,
        )

        # 运行时状态
        self.frame_picker = _BounceIndex(len(self.bbox_list))
        self.audio_play_list: List[int] = [0] * PLAY_PRE_PAD
        self.audio_queue_get_feat = np.zeros([PRE_AUDIO_LEN], dtype=np.int16)
        self.using_feat = np.zeros([USING_FEAT_INIT, 16, 512], dtype=np.float32)

        self.counter = 0
        self.empty_audio_counter = 56
        self.is_processing = False
        self.silence = True

    def reset(self):
        """收到首段有效音频时清空缓存。"""
        self.audio_queue_get_feat = np.zeros([PRE_AUDIO_LEN], dtype=np.int16)
        self.audio_play_list = [0] * PLAY_PRE_PAD
        self.counter = 0
        self.is_processing = True

    # ------- 内部小步骤 -----------------------------------------------------

    def _detect_silence(self, audio_frame: np.ndarray):
        if not np.any(audio_frame):
            if not self.silence:
                self.empty_audio_counter += 1
            if self.empty_audio_counter >= SILENCE_THRESHOLD:
                self.silence = True
        else:
            self.empty_audio_counter = 0
            self.silence = False

    def _next_idle_img(self) -> Tuple[Optional[np.ndarray], int]:
        """静音状态/音频不够时返回的"占位帧"。每 5 次调用输出 1 帧图。"""
        if self.counter == 0:
            img = self.full_body_img_list[self.frame_picker.index].copy()
            self.frame_picker.advance()
            self.counter = 1
            return img, 1
        self.counter += 1
        if self.counter == IDLE_LOOP:
            self.counter = 0
        return None, 0

    def _pop_play_audio(self) -> np.ndarray:
        if self.audio_play_list:
            audio = np.array(self.audio_play_list[:FRAME_LEN], dtype=np.int16)
            self.audio_play_list = self.audio_play_list[FRAME_LEN:]
            return audio
        return np.zeros([FRAME_LEN], dtype=np.int16)

    def _run_encoder(self) -> np.ndarray:
        """从 audio_queue 取一段做 fbank + wenet encoder，返回新增的音频特征。"""
        fbank = knf.OnlineFbank(_FBANK_OPTS)
        fbank.accept_waveform(SAMPLE_RATE, self.audio_queue_get_feat.tolist())
        # 把当前正要处理的一小段加进播放队列，保证音视频同步
        self.audio_play_list.extend(
            self.audio_queue_get_feat[PRE_AUDIO_LEN:PRE_AUDIO_LEN + 800]
        )

        mel = np.array([[fbank.get_frame(i) for i in range(fbank.num_frames_ready)]])
        mel = mel[:, :, :WENET_FEAT_FRAMES, :]
        inputs = {
            "chunk": mel.astype(np.float32),
            "offset": self.offset,
            "att_cache": self.att_cache.astype(np.float32),
            "cnn_cache": self.cnn_cache.astype(np.float32),
        }
        outs = self.ort_ae.run(None, inputs)
        return outs[0]

    def _run_unet(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """对当前帧做一次 unet 推理并把结果贴回原图。"""
        xmin, ymin, xmax, ymax = bbox
        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (168, 168))
        onnx_in, crop_ori = _build_unet_inputs(crop_img)

        audio_feat = self.using_feat.reshape(1, 128, 16, 32)
        inputs = {
            self.ort_unet.get_inputs()[0].name: onnx_in,
            self.ort_unet.get_inputs()[1].name: audio_feat,
        }
        outs = self.ort_unet.run(None, inputs)
        pred = (outs[0][0].transpose(1, 2, 0) * 255).astype(np.uint8)

        crop_ori[4:164, 4:164] = pred
        crop_ori = cv2.resize(crop_ori, (w, h))
        img[ymin:ymax, xmin:xmax] = crop_ori
        return img

    # ------- 主入口 --------------------------------------------------------

    def process(self, audio_frame: np.ndarray):
        audio_frame = audio_frame.astype(np.int16)
        self._detect_silence(audio_frame)

        if self.silence:
            # 静音模式：清空音频缓存，按顺序返回素材帧
            self.audio_queue_get_feat = np.array([], dtype=np.int16)
            self.is_processing = False
            return_img, check_img = self._next_idle_img()
            return return_img, np.zeros([FRAME_LEN], dtype=np.int16), check_img

        # 非静音：积攒音频
        if not self.is_processing:
            self.reset()
        if audio_frame.shape[0] < FRAME_LEN:
            audio_frame = np.pad(audio_frame, (0, FRAME_LEN - audio_frame.shape[0]))
        self.audio_queue_get_feat = np.concatenate(
            [self.audio_queue_get_feat, audio_frame], axis=0,
        )

        if self.audio_queue_get_feat.shape[0] >= WENET_TRIGGER_LEN:
            audio_feat = self._run_encoder()
            self.audio_queue_get_feat = self.audio_queue_get_feat[WENET_CHUNK_DROP:]
            self.using_feat = np.concatenate([self.using_feat, audio_feat], axis=0)

            img = self.full_body_img_list[self.frame_picker.index].copy()
            bbox = self.bbox_list[self.frame_picker.index]
            self.frame_picker.advance()

            if self.using_feat.shape[0] >= UNET_FEAT_WINDOW:
                img = self._run_unet(img, bbox)
                self.using_feat = self.using_feat[1:]

            self.counter = 1
            return img.copy(), self._pop_play_audio(), 1

        # 音频不够：跟静音模式一样按 idle 节奏吐占位帧
        return_img, check_img = self._next_idle_img()
        return return_img, self._pop_play_audio(), check_img


def _select_fourcc(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc("M", "J", "P", "G")
    return cv2.VideoWriter_fourcc(*"mp4v")


def main(arg):
    import soundfile as sf
    from scipy.io import wavfile

    stream, _ = sf.read(arg.audio_wav)
    if stream.ndim == 2:
        stream = stream[:, 0]
    stream = (stream.astype(np.float32) * 32767).astype(np.int16)

    w, h = arg.video_size
    writer = cv2.VideoWriter(arg.out_video, _select_fourcc(arg.out_video), 20, (w, h))
    processor = DiHumanProcessor(arg.data_path)

    audio_out: List[np.ndarray] = []
    n_chunks = math.ceil(stream.shape[0] / FRAME_LEN)
    for i in range(n_chunks):
        a = i * FRAME_LEN
        b = min(a + FRAME_LEN, stream.shape[0])
        audio_frame = stream[a:b]
        img, playing_audio, check_img = processor.process(audio_frame)
        audio_out.append(playing_audio)
        if check_img:
            writer.write(img)

    writer.release()
    audio_data = np.concatenate(audio_out).astype(np.int16)
    wavfile.write(arg.out_audio, SAMPLE_RATE, audio_data)
    print(f"[done] video saved to {arg.out_video}, audio saved to {arg.out_audio}")


if __name__ == "__main__":
    main(parse_args())


# 合并音视频示例：
# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
