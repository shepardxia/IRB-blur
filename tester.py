import imp
import os.path as osp
import os
import torch
import time

MainModel = imp.load_source('MainModel', "load_model.py")

this_dir = osp.dirname(__file__)

model = torch.load('plate.pth').to('cuda')

quantized_model = torch.quantization.quantize_dynamic(model, \
    qconfig_spec={torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.functional.relu}, \
        dtype=torch.qint8).to('cuda')

input = torch.ones((10, 3, 256, 256), device='cuda')

input = input / 2

print(model(input).detach().reshape(-1))
print(quantized_model(input).detach().reshape(-1))

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

print_model_size(model)
print_model_size(quantized_model)

torch.cuda.synchronize()
time_one = time.time()

for i in range(50):
    model(input)

torch.cuda.synchronize()
time_two = time.time()

for i in range(50):
    quantized_model(input)

torch.cuda.synchronize()
time_three = time.time()

print('original took time:', time_two - time_one)
print('copy took time:', time_three - time_two)
