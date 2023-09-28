import torch
import onnx
from onnxsim import simplify
import os
import yaml
import logging
from kantts.models.hifigan.hifigan import Generator

basepath = "/home/zhaoming/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k"
voice = "zhizhe_emo"  # zhizhe_emo 男声  zhitian_emo  女生
save_path = "./model_save"
if not os.path.exists(save_path):
    os.makedirs(save_path)

voc_ckpt = f"{basepath}/voices/{voice}/voc"
voc_config = os.path.join(voc_ckpt, "config.yaml")

ckpt_path = voc_ckpt + "/ckpt/checkpoint_0.pth"

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", 0)


with open(voc_config, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

model = Generator(**config["Model"]["Generator"]["params"])
states = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(states["model"]["generator"])

logging.info(f"Loaded model parameters from {ckpt_path}.")
model.remove_weight_norm()
model = model.eval().to(device)

dummy_0 = torch.randn(1, 80, 150).to(device)


dynamic_axes = {'input':{2:'len'}}
input_names = ['input']
output_names = ['output']

# 将模型保存为ONNX格式
torch.onnx.export(model, dummy_0, f"{save_path}/model_{voice}.onnx", input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes,
                  opset_version=11)

# Load the ONNX model
model = onnx.load(f"{save_path}/model_{voice}.onnx")

# Simplify the ONNX model
simplified_model, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"

# Save the simplified model
onnx.save(simplified_model, f"{save_path}/simplify_model_{voice}.onnx")
print(f"onnx model saved at {save_path}/simplify_model_{voice}.onnx")
