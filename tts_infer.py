import os
import sys
import logging
from glob import glob
import soundfile as sf
import numpy as np
import yaml
import uuid

from kantts.bin.infer_sambert import am_init, am_forward
from kantts.bin.infer_hifigan import hifigan_forward, hifigan_init
import time

import ttsfrd

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


class TTS():
    def __init__(self, basepath, voice):

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


        self.hifigan_model = hifigan_init(ckpt_path=os.path.join(self.voc_ckpt, "ckpt/checkpoint_0.pth"),
                                     config=self.voc_config)
        self.am_model = am_init(ckpt=os.path.join(self.am_ckpt, "ckpt/checkpoint_0.pth"), config=self.am_config)

        self.fe = ttsfrd.TtsFrontendEngine()
        self.fe.initialize(self.resource_dir)
        self.fe.set_lang_type(ENG_LANG_MAPPING["PinYin"])

    def concat_process(self, chunked_dir, output_dir):
        wav_files = sorted(glob(os.path.join(chunked_dir, "*.wav")))
        sentence_sil = 0.28  # seconds
        end_sil = 0.05  # seconds

        cnt = 0
        wav_concat = None
        main_id, sub_id = 0, 0

        while cnt < len(wav_files):
            wav_file = os.path.join(
                chunked_dir, "{}_{}_mel_gen.wav".format(main_id, sub_id)
            )
            if os.path.exists(wav_file):
                wav, sr = sf.read(wav_file)
                sentence_sil_samples = int(sentence_sil * sr)
                end_sil_samples = int(end_sil * sr)
                if sub_id == 0:
                    wav_concat = wav
                else:
                    wav_concat = np.concatenate(
                        (wav_concat, np.zeros(sentence_sil_samples), wav), axis=0
                    )

                sub_id += 1
                cnt += 1
            else:
                if wav_concat is not None:
                    wav_concat = np.concatenate(
                        (wav_concat, np.zeros(end_sil_samples)), axis=0
                    )
                    sf.write(os.path.join(output_dir, f"res.wav"), wav_concat, sr)

                main_id += 1
                sub_id = 0
                wav_concat = None

            if cnt == len(wav_files):
                wav_concat = np.concatenate((wav_concat, np.zeros(end_sil_samples)), axis=0)
                sf.write(os.path.join(output_dir, f"res.wav"), wav_concat, sr)
                # sf.write("./tmp.wav", wav_concat, sr)
        #
        #
        # # sf.write(os.path.join(output_dir, f"all.wav"), wav_concat, sr)

    def infer(self, text, scale=1.0):
        # t0 = time.time()
        self.output_dir = "./outputs/" + str(uuid.uuid4())
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "res_wavs"), exist_ok=True)

        # t0 = time.time()
        symbols_lst = text_to_mit_symbols1(text, self.fe, self.speaker)

        symbols_file = os.path.join(self.output_dir, "symbols.lst")
        with open(symbols_file, "w") as symbol_data:
            for symbol in symbols_lst:
                symbol_data.write(symbol)


        # t0 = time.time()
        logging.info("AM is infering...")
        # am_forward(symbols_file, os.path.join(am_ckpt, "ckpt/checkpoint_0.pth"), output_dir, se_file, config=am_config)
        am_forward(ckpt=os.path.join(self.am_ckpt, "ckpt/checkpoint_0.pth"), sentence=symbols_file, model=self.am_model,
                   output_dir=self.output_dir, se_file=None, config=self.am_config, scale=scale)

        logging.info("Vocoder is infering...")
        hifigan_forward(os.path.join(self.output_dir, "feat"), model=self.hifigan_model, output_dir=self.output_dir)

        self.concat_process(self.output_dir, os.path.join(self.output_dir, "res_wavs"))


        # logging.info("Text to wav finished!")
        # t1 = time.time()
        # print("infer time: ", t1-t0)

        return self.output_dir