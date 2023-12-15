import sys
import torch
import os
import numpy as np
import argparse
import yaml
import logging
import time
from tensorrt_onnx.run_trt import TRTWrapper

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

from mykantts.models import model_builder
from mykantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def denorm_f0(mel, f0_threshold=30, uv_threshold=0.6, norm_type='mean_std', f0_feature=None):
    if norm_type == 'mean_std':
        f0_mvn = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * f0_mvn[1:, :] + f0_mvn[0:1, :]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv
    else:  # global
        f0_global_max_min = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * (f0_global_max_min[0] - f0_global_max_min[1]) + f0_global_max_min[1]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv

    return mel


def am_synthesis(symbol_seq, fsnet, ling_unit, device, se=None, scale=1.0):
    inputs_feat_lst = ling_unit.encode_symbol_sequence(symbol_seq)

    inputs_feat_index = 0
    if ling_unit.using_byte():
        inputs_byte_index = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack([inputs_byte_index], dim=-1).unsqueeze(0)
    else:
        inputs_sy = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_tone = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_syllable = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_ws = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable, inputs_ws], dim=-1
        ).unsqueeze(0)

    inputs_feat_index = inputs_feat_index + 1
    inputs_emo = (
        torch.from_numpy(inputs_feat_lst[inputs_feat_index])
        .long()
        .to(device)
        .unsqueeze(0)
    )

    inputs_feat_index = inputs_feat_index + 1
    se_enable = False if se is None else True
    if se_enable:
        inputs_spk = (
            torch.from_numpy(se.repeat(len(inputs_feat_lst[inputs_feat_index]), axis=0))
            .float()
            .to(device)
            .unsqueeze(0)[:, :-1, :]
        )
    else:
        inputs_spk = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index])
            .long()
            .to(device)
            .unsqueeze(0)[:, :-1]
        )

    inputs_len = (
            torch.zeros(1).to(device).long() + inputs_emo.size(1) - 1
    )  # minus 1 for "~"

    # t0 = time.time()
    res = fsnet(
        inputs_ling[:, :-1, :],
        inputs_emo[:, :-1],
        inputs_spk,
        inputs_len,
        scale=scale
    )
    # print(inputs_ling[:, :-1, :].shape, inputs_emo[:, :-1].shape, inputs_spk.shape, inputs_len.shape)
    # t1 = time.time()
    # print("fsnet infer time: ", t1-t0)

    x_band_width = res["x_band_width"]
    h_band_width = res["h_band_width"]
    #  enc_slf_attn_lst = res["enc_slf_attn_lst"]
    #  pnca_x_attn_lst = res["pnca_x_attn_lst"]
    #  pnca_h_attn_lst = res["pnca_h_attn_lst"]
    dec_outputs = res["dec_outputs"]
    postnet_outputs = res["postnet_outputs"]
    LR_length_rounded = res["LR_length_rounded"]
    log_duration_predictions = res["log_duration_predictions"]
    pitch_predictions = res["pitch_predictions"]
    energy_predictions = res["energy_predictions"]

    valid_length = int(LR_length_rounded[0].item())
    dec_outputs = dec_outputs[0, :valid_length, :].cpu().numpy()
    postnet_outputs = postnet_outputs[0, :valid_length, :]  # .cpu().numpy()
    duration_predictions = (
        (torch.exp(log_duration_predictions) - 1 + 0.5).long().squeeze().cpu().numpy()
    )
    pitch_predictions = pitch_predictions.squeeze().cpu().numpy()
    energy_predictions = energy_predictions.squeeze().cpu().numpy()

    # logging.info("x_band_width:{}, h_band_width: {}".format(x_band_width, h_band_width))

    return (
        dec_outputs,
        postnet_outputs,
        duration_predictions,
        pitch_predictions,
        energy_predictions,
    )


def prepare_input(symbol_seq, ling_unit, device, se=None):
    inputs_feat_lst = ling_unit.encode_symbol_sequence(symbol_seq)

    inputs_feat_index = 0
    if ling_unit.using_byte():
        inputs_byte_index = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack([inputs_byte_index], dim=-1).unsqueeze(0)
    else:
        inputs_sy = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_tone = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_syllable = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_ws = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable, inputs_ws], dim=-1
        ).unsqueeze(0)

    inputs_feat_index = inputs_feat_index + 1
    inputs_emo = (
        torch.from_numpy(inputs_feat_lst[inputs_feat_index])
        .long()
        .to(device)
        .unsqueeze(0)
    )

    inputs_feat_index = inputs_feat_index + 1
    se_enable = False if se is None else True
    if se_enable:
        inputs_spk = (
            torch.from_numpy(se.repeat(len(inputs_feat_lst[inputs_feat_index]), axis=0))
            .float()
            .to(device)
            .unsqueeze(0)[:, :-1, :]
        )
    else:
        inputs_spk = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index])
            .long()
            .to(device)
            .unsqueeze(0)[:, :-1]
        )

    inputs_len = (
            torch.zeros(1).to(device).long() + inputs_emo.size(1) - 1
    )  # minus 1 for "~"
    return inputs_ling[:, :-1, :], inputs_emo[:, :-1], inputs_spk, inputs_len


def prepare_batch_input(symbol_seq_list, ling_unit, device, se):
    input_list = []
    for symbol_seq in symbol_seq_list:
        input_list.append(prepare_input(symbol_seq, ling_unit, device, se))
    # 将张量和长度分别组成批量
    batch_inputs_ling = torch.nn.utils.rnn.pad_sequence([x[0].squeeze(dim=0) for x in input_list], batch_first=True)
    batch_inputs_emo = torch.nn.utils.rnn.pad_sequence([x[1].squeeze(dim=0) for x in input_list], batch_first=True)
    batch_inputs_spk = torch.nn.utils.rnn.pad_sequence([x[2].squeeze(dim=0) for x in input_list], batch_first=True)
    batch_inputs_len = torch.stack([x[3] for x in input_list], dim=0).squeeze(dim=1)
    return batch_inputs_ling, batch_inputs_emo, batch_inputs_spk, batch_inputs_len


def am_synthesis_batch(symbol_seq_list, fsnet, ling_unit, device, se=None, scale=1.0, batch_size=16):
    with torch.no_grad():
        final_output_list = []
        max_len = symbol_seq_list.__len__()
        for i in range(0, max_len, batch_size):
            end = min(i + batch_size, max_len)
            sub_symbol_seq_list = symbol_seq_list[i:end]
            size = len(sub_symbol_seq_list)
            batch_inputs_ling, batch_inputs_emo, batch_inputs_spk, batch_inputs_len = prepare_batch_input(
                sub_symbol_seq_list, ling_unit, device, se)
            res = fsnet(batch_inputs_ling, batch_inputs_emo, batch_inputs_spk, batch_inputs_len, scale=scale)
            postnet_outputs, LR_length_rounded = res["postnet_outputs"], res["LR_length_rounded"]
            postnet_outputs_list = torch.chunk(postnet_outputs, chunks=size, dim=0)
            LR_length_rounded_list = torch.chunk(LR_length_rounded, chunks=size, dim=0)
            out_put_list = []
            for i in range(size):
                valid_length = int(LR_length_rounded_list[i][0].item())
                mel_post = postnet_outputs_list[i][0, :valid_length, :].cpu()
                out_put_list.append(mel_post)
            final_output_list += out_put_list
        return final_output_list


def am_init(ckpt, config=None, infer_type="torch", voice=''):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        am_config_file = os.path.join(
            ckpt, "config.yaml"
        )
        with open(am_config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    ling_unit = KanTtsLinguisticUnit(config)
    ling_unit_size = ling_unit.get_unit_size()
    config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)

    config["Model"]["KanTtsSAMBERT"]["params"]['infer_type'] = infer_type
    if infer_type == "trt":
        config["Model"]["KanTtsSAMBERT"]["params"]['trt_model'] = TRTWrapper(
            f'./tensorrt_onnx/model_save/mel_decode_{voice}.engine')
        logging.info(f"Loading sambert engine: ./tensorrt_onnx/model_save/mel_decode_{voice}.engine")
    # print("config.infer_type: ", config["Model"]["KanTtsSAMBERT"]["params"]['infer_type'])

    model, _, _ = model_builder(config, device)

    fsnet = model["KanTtsSAMBERT"]

    logging.info("Loading checkpoint: {}".format(ckpt))
    state_dict = torch.load(ckpt)

    fsnet.load_state_dict(state_dict["model"], strict=False)

    fsnet.eval()

    return fsnet

# def am_forward(ckpt, sentence, model, output_dir, se_file=None, config=None, scale=1.0):
#     results_dir = os.path.join(output_dir, "feat")
#     os.makedirs(results_dir, exist_ok=True)
#
#     fsnet = model
#     # results_dir = os.path.join(output_dir, "feat")
#
#     if config is not None:
#         with open(config, "r") as f:
#             config = yaml.load(f, Loader=yaml.Loader)
#     else:
#         am_config_file = os.path.join(
#             ckpt, "config.yaml"
#         )
#         with open(am_config_file, "r") as f:
#             config = yaml.load(f, Loader=yaml.Loader)
#     ling_unit = KanTtsLinguisticUnit(config)
#     ling_unit_size = ling_unit.get_unit_size()
#     config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)
#     se_enable = config["Model"]["KanTtsSAMBERT"]["params"].get("SE", False)
#     print("se_enable: ", se_enable)
#     se = np.load(se_file) if se_enable else None
#
#     ling_unit = KanTtsLinguisticUnit(config)
#     ling_unit_size = ling_unit.get_unit_size()
#     config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)
#
#     # nsf
#     nsf_enable = config["Model"]["KanTtsSAMBERT"]["params"].get("NSF", False)
#     print("nsf_enable: ", nsf_enable)
#     if nsf_enable:
#         nsf_norm_type = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_norm_type", "mean_std")
#         if nsf_norm_type == "mean_std":
#             f0_mvn_file = os.path.join(
#                 os.path.dirname(os.path.dirname(ckpt)), "mvn.npy"
#             )
#             f0_feature = np.load(f0_mvn_file)
#         else: # global
#             nsf_f0_global_minimum = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_f0_global_minimum", 30.0)
#             nsf_f0_global_maximum = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_f0_global_maximum", 730.0)
#             f0_feature = [nsf_f0_global_maximum, nsf_f0_global_minimum]
#
#     if not torch.cuda.is_available():
#         device = torch.device("cpu")
#     else:
#         torch.backends.cudnn.benchmark = True
#         device = torch.device("cuda", 0)
#
#     with open(sentence, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip().split("\t")
#             # logging.info("Inference sentence: {}".format(line[0]))
#             mel_path = "%s/%s_mel.npy" % (results_dir, line[0])
#             dur_path = "%s/%s_dur.txt" % (results_dir, line[0])
#             f0_path = "%s/%s_f0.txt" % (results_dir, line[0])
#             energy_path = "%s/%s_energy.txt" % (results_dir, line[0])
#
#             t0 = time.time()
#             with torch.no_grad():
#                 mel, mel_post, dur, f0, energy = am_synthesis(
#                     line[1], fsnet, ling_unit, device, se=se, scale=scale
#                 )
#             t1 = time.time()
#
#             if nsf_enable:
#                 mel_post = denorm_f0(mel_post, norm_type=nsf_norm_type, f0_feature=f0_feature)
#             t1 = time.time()
#             np.save(mel_path, mel_post)
#             np.savetxt(dur_path, dur)
#             np.savetxt(f0_path, f0)
#             np.savetxt(energy_path, energy)
#
#             t2 = time.time()
#             print("save mel time: ", t2-t1)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sentence", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--ckpt", type=str, required=True)
#     parser.add_argument("--se_file", type=str, required=False)
#
#     args = parser.parse_args()
#
#     am_infer(args.sentence, args.ckpt, args.output_dir, args.se_file)
