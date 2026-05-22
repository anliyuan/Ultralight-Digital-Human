from argparse import ArgumentParser

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import HubertModel, Wav2Vec2Processor


DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


def get_hubert_from_16k_wav(wav_16k_name, device=DEFAULT_DEVICE):
    speech_16k, _ = sf.read(wav_16k_name)
    return get_hubert_from_16k_speech(speech_16k, device=device)


@torch.no_grad()
def get_hubert_from_16k_speech(speech, device=DEFAULT_DEVICE):
    global hubert_model
    hubert_model = hubert_model.to(device)
    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] => [T,]
    input_values_all = wav2vec2_processor(
        speech, return_tensors="pt", sampling_rate=16000
    ).input_values  # [1, T]
    input_values_all = input_values_all.to(device)

    # HuBERT 的 CNN stride 等价于 kernel=400, stride=320 的 1D 卷积
    # T = floor((t - k) / s)
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx:end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    if input_values.shape[1] >= kernel:
        hidden_states = hubert_model(input_values).last_hidden_state
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret


def make_even_first_dim(tensor):
    """让 T 维变成偶数。原实现是直接丢一帧，这里改成 pad 一帧 0，避免丢失最后一帧信息。"""
    size = list(tensor.size())
    if size[0] % 2 == 1:
        pad = torch.zeros_like(tensor[:1])
        return torch.cat([tensor, pad], dim=0)
    return tensor


def main():
    parser = ArgumentParser()
    parser.add_argument('--wav', type=str, required=True)
    parser.add_argument('--out', type=str, default="",
                        help="optional output .npy path, default <wav>_hu.npy")
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    wav_name = args.wav
    speech, sr = sf.read(wav_name)
    speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    print(f"SR: {sr} to 16000")

    hubert_hidden = get_hubert_from_16k_speech(speech_16k, device=args.device)
    hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)

    out_path = args.out if args.out else wav_name.replace('.wav', '_hu.npy')
    np.save(out_path, hubert_hidden.detach().numpy())
    print(f"saved: {out_path} shape={hubert_hidden.shape}")


if __name__ == "__main__":
    main()
