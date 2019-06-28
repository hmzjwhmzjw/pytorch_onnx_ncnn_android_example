#-*- coding:utf-8 _*-
"""
@author:zjw
@file: onnx_convert.py
@time: 2019/06/28
"""
import time
import numpy as np
import torch
from PIL import Image
import mobilenetv3
import onnx
import caffe2.python.onnx.backend as backend


src_image = 'cat.jpg'
input_size = 224
mean_vals = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
std_vals = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

imagenet_ids = []
with open("synset_words.txt", "r") as f:
    for line in f.readlines():
        imagenet_ids.append(line.strip())

###############################prepare model####################################
infer_device = torch.device('cpu')
res_model = mobilenetv3.MobileNetV3()
state_dict = torch.load('./mobilenetv3x1.0.pth', map_location=lambda storage, loc: storage)
res_model.load_state_dict(state_dict)
res_model.to(infer_device)
res_model.eval()

################################prepare test image#####################################
pil_im = Image.open(src_image).convert('RGB')
pil_im_resize = pil_im.resize((input_size, input_size), Image.BILINEAR)

origin_im_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_im_resize.tobytes()))
origin_im_tensor = origin_im_tensor.view(input_size, input_size, 3)
origin_im_tensor = origin_im_tensor.transpose(0, 1).transpose(0, 2).contiguous()
origin_im_tensor = (origin_im_tensor.float()/255 - mean_vals)/std_vals
origin_im_tensor = origin_im_tensor.unsqueeze(0)


###########################test################################
t1=time.time()
with torch.no_grad():
    pred = res_model(origin_im_tensor.to(infer_device))
    predidx = torch.argmax(pred, dim=1)
t2 = time.time()
print(t2 - t1)
print("predict result: ", imagenet_ids[predidx])


##################export###############
output_onnx = 'mobilenetv3.onnx'
x = origin_im_tensor
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input0"]
output_names = ["output0"]
torch_out = torch.onnx._export(res_model, x, output_onnx, export_params=True, verbose=False, input_names=input_names, output_names=output_names)


print("==> Loading and checking exported model from '{}'".format(output_onnx))
onnx_model = onnx.load(output_onnx)
onnx.checker.check_model(onnx_model)  # assuming throw on error
print("==> Passed")

print("==> Loading onnx model into Caffe2 backend and comparing forward pass.".format(output_onnx))
caffe2_backend = backend.prepare(onnx_model)
B = {"input0": x.data.numpy()}
c2_out = caffe2_backend.run(B)["output0"]


print("==> compare torch output and caffe2 output")
np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)
print("==> Passed")




