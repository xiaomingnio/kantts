import os
from glob import glob
import soundfile as sf
import numpy as np
import yaml
import uuid
import torch

from tensorrt_onnx.run_onnx import ONNXModel
from tensorrt_onnx.run_trt import TRTWrapper
from kantts.bin.infer_sambert import am_init, am_synthesis
from kantts.bin.infer_hifigan import hifigan_init
from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
import time
import wave

import ttsfrd
import logging

ENG_LANG_MAPPING = {
    "PinYin": "zh-cn",
    "English": "en-us",
    "British": "en-gb",
    "ZhHK": "hk_cantonese",
    "Sichuan": "sichuan",
    "Japanese": "japanese",
    "WuuShangHai": "shanghai",
    "Indonesian": "indonesian",
    "Malay": "malay",
    "Filipino": "filipino",
    "Vietnamese": "vietnamese",
    "Korean": "korean",
    "Russian": "russian",
}


def text_to_mit_symbols1(text, fe, speaker):
    symbols_lst = []

    text = text.strip()
    res = fe.gen_tacotron_symbols(text)
    res = res.replace("F7", speaker)
    sentences = res.split("\n")
    for sentence in sentences:
        arr = sentence.split("\t")
        # skip the empty line
        if len(arr) != 2:
            continue
        sub_index, symbols = sentence.split("\t")
        symbol_str = "{}_{}\t{}\n".format(0, sub_index, symbols)
        symbols_lst.append(symbol_str)

    return symbols_lst

def save_wav(data, out_path):
    with wave.open(out_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 2 字节，16-bit
        wav_file.setframerate(16000)  # 采样率为 44100 Hz

        # 将数据转换为 16-bit 格式
        data = (data * 32767).astype(np.int16)

        # 写入数据
        wav_file.writeframes(data.tobytes())

class TTS():
    def __init__(self, basepath, voice, infer_type):
        self.infer_type = infer_type
        self.resource_dir = f"{basepath}/resource"
        self.am_ckpt = f"{basepath}/voices/{voice}/am"
        self.voc_ckpt = f"{basepath}/voices/{voice}/voc"
        self.se_file = None
        self.am_config = os.path.join(self.am_ckpt, "config.yaml")
        self.voc_config = os.path.join(self.voc_ckpt, "config.yaml")

        self.output_dir = r"./outputs"


        with open(self.am_config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self.speaker = self.config["linguistic_unit"]["speaker_list"].split(",")[0]

        logging.info(f"HifiGAN infer using : {infer_type}.......")
        if infer_type == "torch":
            self.hifigan_model = hifigan_init(ckpt_path=os.path.join(self.voc_ckpt, "ckpt/checkpoint_0.pth"),
                                         config=self.voc_config)
            p = os.path.join(self.voc_ckpt, "ckpt/checkpoint_0.pth")
            logging.info(f"Loading HifiGAN checkpoint: {p}")
        elif infer_type == "onnx_cpu":
            self.hifigan_model = ONNXModel(f"./tensorrt_onnx/model_save/simplify_model_{voice}.onnx", device='cpu')
            logging.info(f"Loading HifiGAN checkpoint: ./tensorrt_onnx/model_save/simplify_model_{voice}.onnx")
        elif infer_type == "onnx_gpu":
            self.hifigan_model = ONNXModel(f"./tensorrt_onnx/model_save/simplify_model_{voice}.onnx", device='gpu')
            logging.info(f"Loading HifiGAN checkpoint: ./tensorrt_onnx/model_save/simplify_model_{voice}.onnx")
        elif infer_type == "trt":
            self.hifigan_model = TRTWrapper(f"./tensorrt_onnx/model_save/simplify_model_{voice}.engine", ['output'])
            logging.info(f"Loading HifiGAN checkpoint: ./tensorrt_onnx/model_save/simplify_model_{voice}.engine")
        else:
            print("Wrong infer type......")

        self.am_model = am_init(ckpt=os.path.join(self.am_ckpt, "ckpt/checkpoint_0.pth"), config=self.am_config)

        self.fe = ttsfrd.TtsFrontendEngine()
        self.fe.initialize(self.resource_dir)
        self.fe.set_lang_type(ENG_LANG_MAPPING["PinYin"])

    def concat_process(self, chunked_dir, output_dir):
        wav_files = sorted(glob(os.path.join(chunked_dir, "*.wav")))
        # print(wav_files)
        sentence_sil = 0.28  # seconds
        end_sil = 0.05  # seconds

        wav_concat, sr = sf.read(wav_files[0])

        sentence_sil_samples = int(sentence_sil * sr)
        end_sil_samples = int(end_sil * sr)

        if len(wav_files) >= 2:
            for p in wav_files[1:]:
                wav, sr = sf.read(p)

                wav_concat = np.concatenate(
                    (wav_concat, np.zeros(sentence_sil_samples), wav), axis=0
                )
        wav_concat = np.concatenate((wav_concat, np.zeros(end_sil_samples)), axis=0)
        save_wav(wav_concat, os.path.join(output_dir, f"all.wav"))


    def infer(self, text, scale=1.0):
        # t0 = time.time()
        self.output_dir = "./outputs/" + str(uuid.uuid4())
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "res_wavs"), exist_ok=True)

        t0 = time.time()
        symbols_lst = text_to_mit_symbols1(text, self.fe, self.speaker)
        t1 = time.time()
        print("文本前端推理时间: ", (t1-t0)*1000, " ms")

        symbols_file = os.path.join(self.output_dir, "symbols.lst")
        with open(symbols_file, "w") as symbol_data:
            for symbol in symbols_lst:
                symbol_data.write(symbol)


        with open(self.am_config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        ling_unit = KanTtsLinguisticUnit(config)
        ling_unit_size = ling_unit.get_unit_size()
        config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)

        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            torch.backends.cudnn.benchmark = True
            device = torch.device("cuda", 0)

        with open(symbols_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip().split("\t")
                # logging.info("Inference sentence: {}".format(line[0]))
                # mel_path = "%s/%s_mel.npy" % (results_dir, line[0])
                # dur_path = "%s/%s_dur.txt" % (results_dir, line[0])
                # f0_path = "%s/%s_f0.txt" % (results_dir, line[0])
                # energy_path = "%s/%s_energy.txt" % (results_dir, line[0])

                t0 = time.time()
                with torch.no_grad():
                    mel, mel_post, dur, f0, energy = am_synthesis(
                        line[1], self.am_model, ling_unit, device, se=None, scale=scale
                    )
                t1 = time.time()
                print("am infer time: ", (t1-t0)*1000, " ms")
                # (T, C) -> (B, C, T)
                mel_data = mel_post.transpose(1, 0).unsqueeze(0)

                if self.infer_type == "torch":
                    # mel_data = torch.tensor(mel_post, dtype=torch.float).to(device)
                    # print("mel_data.shape: ", mel_data.shape)
                    y = self.hifigan_model(mel_data)
                    y = y.view(-1).detach().cpu().numpy()
                elif self.infer_type in ["onnx_cpu", "onnx_gpu"]:
                    out_onnx = self.hifigan_model.onnx_session.run([], input_feed={'input': mel_data.cpu().numpy()})
                    y = np.squeeze(out_onnx[0])
                    # print("y shape: ", y.shape)
                elif self.infer_type == 'trt':
                    output = self.hifigan_model(dict(input=mel_data.cuda()))
                    # print(output)
                    # print(output['output'].shape)
                    y = output['output'].view(-1).detach().cpu().numpy()

                # if hasattr(self.hifigan_model, "pqmf"):
                #     print("----------------------")
                #     y = self.hifigan_model.pqmf.synthesis(y)
                # y = y.view(-1).detach().cpu().numpy()
                # pcm_len += len(y)
                t2 = time.time()
                print("hifigan infer time: ", (t2-t1)*1000, " ms")
                # print(y)

                save_wav(y, os.path.join(self.output_dir, f"{i}_gen.wav"))


        # t1 = time.time()
        # print("文本前端infer time: ", t1-t0)


        # t0 = time.time()
        # logging.info("AM is infering...")
        # # am_forward(symbols_file, os.path.join(am_ckpt, "ckpt/checkpoint_0.pth"), output_dir, se_file, config=am_config)
        # am_forward(ckpt=os.path.join(self.am_ckpt, "ckpt/checkpoint_0.pth"), sentence=symbols_file, model=self.am_model,
        #            output_dir=self.output_dir, se_file=None, config=self.am_config, scale=scale)
        # t1 = time.time()
        # print("am_forward time: ", t1-t0)

        # t0 = time.time()
        # logging.info("Vocoder is infering...")
        # hifigan_forward(os.path.join(self.output_dir, "feat"), model=self.hifigan_model, output_dir=self.output_dir)
        # t1 = time.time()
        # print("hifigan_forward time: ", t1-t0)
        self.concat_process(self.output_dir, os.path.join(self.output_dir, "res_wavs"))
        # t2 = time.time()
        # print("concat_process time: ", t2 - t1)

        # logging.info("Text to wav finished!")
        # t1 = time.time()
        # print("infer time: ", t1-t0)

        return self.output_dir