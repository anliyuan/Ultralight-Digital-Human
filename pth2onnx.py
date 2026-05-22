import argparse
import time

import numpy as np
import onnx
import onnxruntime
import torch

from unet import Model


def check_onnx(onnx_path, torch_out, torch_in, audio):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    providers = (
        ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print("providers:", ort_session.get_providers())
    ort_inputs = {
        ort_session.get_inputs()[0].name: torch_in.cpu().numpy(),
        ort_session.get_inputs()[1].name: audio.cpu().numpy(),
    }
    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    t2 = time.time()
    print(f"onnx time cost: {t2 - t1:.4f}s")

    np.testing.assert_allclose(
        torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05,
    )
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to trained .pth (state_dict or {'model': state_dict})")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="output onnx path")
    parser.add_argument("--asr", type=str, default="wenet",
                        choices=["wenet", "hubert"])
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    net = Model(6, args.asr).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        net.load_state_dict(ckpt["model"])
    else:
        net.load_state_dict(ckpt)

    img = torch.zeros([1, 6, 160, 160])
    if args.asr == "wenet":
        audio = torch.zeros([1, 128, 16, 32])
    else:
        audio = torch.zeros([1, 16, 32, 32])

    with torch.no_grad():
        torch_out = net(img, audio)
        print("torch_out.shape:", torch_out.shape)
        torch.onnx.export(
            net, (img, audio), args.onnx_path,
            input_names=['input', 'audio'],
            output_names=['output'],
            opset_version=args.opset,
            export_params=True,
        )
    check_onnx(args.onnx_path, torch_out, img, audio)


if __name__ == "__main__":
    main()
