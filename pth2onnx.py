from unet import Model
import onnx
import torch

import onnxruntime
import numpy as np
import time
onnx_path = "./dihuman.onnx"

def check_onnx(torch_out, torch_in, audio):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    import onnxruntime
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print(ort_session.get_providers())
    ort_inputs = {ort_session.get_inputs()[0].name: torch_in.cpu().numpy(), ort_session.get_inputs()[1].name: audio.cpu().numpy()}
    for i in range(1):
        t1 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        t2 = time.time()
        print("onnx time cost::", t2 - t1)

    np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        

net = Model(6).eval()
net.load_state_dict(torch.load("20.pth"))
img = torch.zeros([1, 6, 160, 160])
audio = torch.zeros([1, 128, 16, 32])

input_dict = {"input": img, "audio": audio}

with torch.no_grad():
    torch_out = net(img, audio)
    print(torch_out.shape)
    torch.onnx.export(net, (img, audio), onnx_path, input_names=['input', "audio"],
                    output_names=['output'], 
                    # dynamic_axes=dynamic_axes,
                    # example_outputs=torch_out,
                    opset_version=11,
                    export_params=True)
check_onnx(torch_out, img, audio)
