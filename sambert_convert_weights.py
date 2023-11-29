import os
from os.path import expanduser
home_directory = expanduser("~")
# print(home_directory)
import torch
from kantts.models.sambert.kantts_sambert import MelPNCADecoder
from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
import yaml
import struct

save_path = r"./tensorrt_onnx/model_save"
if not os.path.exists(save_path):
    os.makedirs(save_path)

voice = 'zhizhe_emo'  # zhizhe_emo 男声  zhitian_emo  女生
if voice == 'zhitian_emo':
    wts_path = f"{save_path}/mel_decode_{voice}.wts"
elif voice == 'zhizhe_emo':
    wts_path = f"{save_path}/mel_decode_{voice}.wts"
basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k"
am_ckpt = f"{basepath}/voices/{voice}/am"
am_config = os.path.join(am_ckpt, "config.yaml")


with open(am_config, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)


ling_unit = KanTtsLinguisticUnit(config)
ling_unit_size = ling_unit.get_unit_size()
config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)
config["Model"]["KanTtsSAMBERT"]["params"]['infer_type'] = "torch"

mel_decode = MelPNCADecoder(config["Model"]["KanTtsSAMBERT"]["params"])

ckpt_path = os.path.join(am_ckpt, "ckpt/checkpoint_0.pth")
states = torch.load(ckpt_path, map_location="cpu")
print(states.keys())
# print(states["model"].keys())
# mel_decode.load_state_dict(states["model"], strict=False)
# create example data
# x0 = torch.rand((1, 49, 4)).cuda()
# new_ckpt = {}
# new_ckpt['weight'] = states["model"]["mel_decoder.mel_dec.pnca.0.pnca_attn.layer_norm.weight"]
# new_ckpt['bias'] = states["model"]["mel_decoder.mel_dec.pnca.0.pnca_attn.layer_norm.bias"]
# torch.save(new_ckpt, "ln.pt")

with open(wts_path, 'w') as f:
    f.write('{}\n'.format(len(states["model"].keys())))
    for k, v in states["model"].items():
        if "mel_decode" not in k:
            pass
        else:
            print(k, )
            vr = v.reshape(-1).cpu().numpy()
            print(vr)
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
            f.write('\n')