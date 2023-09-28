import argparse
import base64
import time
import shutil
import os

import librosa
import soundfile as sf
import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel
# from src.util import *
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request


from faster_whisper import WhisperModel
from pylangtools.langconv import Converter

from tts import TTS

from os.path import expanduser
home_directory = expanduser("~")
print(home_directory)

import sqlite3

# 连接到数据库，如果数据库不存在则会创建
conn = sqlite3.connect('database.db')

# 创建一个游标对象来执行SQL语句
cursor = conn.cursor()

# 创建数据表
create_table_sql = '''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_ TEXT NOT NULL,
    time_flag_ INTEGER NOT NULL,
    scale_ REAL NOT NULL,
    speaker_ INTEGER NOT NULL,
    res_ TEXT NOT NULL
);
'''
cursor.execute(create_table_sql)

infer_type = "trt"

# zhibei_emo  zhitian_emo  zhiyan_emo  zhizhe_emo
print("加载tts模型.............")
sambert_hifigan_tts_man = TTS(basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k" ,voice ="zhizhe_emo", infer_type=infer_type)
outp = sambert_hifigan_tts_man.infer(text="提到个性化定制语音，大家并不陌生", scale=1)

sambert_hifigan_tts_women = TTS(basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k" ,voice ="zhitian_emo", infer_type=infer_type)
outp1 = sambert_hifigan_tts_women.infer(text="提到个性化定制语音，大家并不陌生", scale=1)
print("加载tts模型成功！")


# Run on GPU with FP16
whisper_asr_model = WhisperModel(r"./whisper_small", device="cuda", compute_type="float16")
# 解析配置
parser = argparse.ArgumentParser(prog='PaddleSpeechDemo', add_help=True)

parser.add_argument(
    "--port",
    action="store",
    type=int,
    help="port of the app",
    default=8010,
    required=False)

args = parser.parse_args()
port = args.port

# 初始化
app = FastAPI()

#设置允许访问的域名
origins = ["*"]  #也可以设置为"*"，即为所有。

#设置跨域传参
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。

class TtsBase(BaseModel):
    text: str

class TtsBase1(BaseModel):
    text: str
    scale: float
    speaker: int
    time_flag: bool

def SuccessRequest(result=None, message="ok"):
    return {"code": 0, "result": result, "message": message}

def ErrorRequest(result=None, message="error"):
    return {"code": -1, "result": result, "message": message}

# 端到端合成
@app.post("/tts/offline")
async def text2speechOffline(tts_base: TtsBase1, request: Request):
    print(f"tts infer in {port}")
    text = tts_base.text
    time_flag = tts_base.time_flag
    scale = tts_base.scale
    speaker = tts_base.speaker

    time_flag_int = int(time_flag)
    if not text:
        return ErrorRequest(message="文本为空")
    else:
        # 查询数据
        select_sql = f"SELECT * FROM users WHERE text_='{text}' AND time_flag_={time_flag_int} AND scale_={scale} AND speaker_={speaker}"
        # print(select_sql)
        cursor.execute(select_sql)
        rows = cursor.fetchall()
        # print(len(rows))
        if len(rows) > 0:
            row = rows[0]
            # print(type(row), row[5])
            return SuccessRequest(result=eval(row[5]))
        else:
            t0 = time.time()
            if speaker == 0:
                outpath = sambert_hifigan_tts_women.infer(text=text, scale=scale)
            elif speaker == 1:
                outpath = sambert_hifigan_tts_man.infer(text=text, scale=scale)
            else:
                return ErrorRequest(message="目前支持说话人0，1，0表示女生，1表示男声。")
            # print(outpath)
            t1 = time.time()
            print("tts infer time: ", (t1-t0)*1000, " ms")

            out_file_path = os.path.join(outpath, "res_wavs/all.wav")
            if time_flag:
                t2 = time.time()
                # wav 24k 转 16k
                # 读取WAV文件
                # waveform, sample_rate = librosa.load(out_file_path, sr=None, mono=False)
                # # 重采样为16kHz
                # resampled_waveform = librosa.resample(waveform, sample_rate, 16000)
                # # 保存为WAV文件
                # sf.write(out_file_path, resampled_waveform, samplerate=16000)

                segments, info = whisper_asr_model.transcribe(out_file_path, beam_size=5, word_timestamps=True, language='zh')

                print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

                timestamp = []
                word_nums = 0
                # print(segments)
                last_word_s = 0

                for segment in segments:
                    for word in segment.words:
                        # print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
                        text_hans = Converter('zh-hans').convert(word.word)
                        word_s = min(word_nums, len(text))
                        word_e = min(word_nums + len(text_hans), len(text))
                        word_nums += len(text_hans)
                        timestamp.append({'start': word.start, 'end': word.end, 'text': text[word_s:word_e]})
                        last_word_s = word_s
                if last_word_s < len(text):
                    timestamp[-1]['text'] = text[last_word_s:]

                nums = 0
                for stamp in timestamp:
                    nums += len(stamp['text'])

                assert nums == len(text)

                t3 = time.time()
                print("whisper infer time: ", (t3 - t2) * 1000, " ms")

                print("infer all time: ", (t3 - t0) * 1000, " ms")
                with open(out_file_path, "rb") as f:
                    data_bin = f.read()
                base_str = base64.b64encode(data_bin)

                # 插入数据
                insert_sql = "INSERT INTO users (text_, time_flag_, scale_, speaker_, res_) VALUES (?, ?, ?, ?, ?)"
                cursor.execute(insert_sql, (text, int(time_flag), scale, speaker, str({'base_str': base_str, 'timestamp': timestamp})))
                conn.commit()
                # shutil.rmtree("./outputs")
                return SuccessRequest(result={'base_str': base_str, 'timestamp': timestamp})
            else:
                with open(out_file_path, "rb") as f:
                    data_bin = f.read()
                base_str = base64.b64encode(data_bin)

                # 插入数据
                insert_sql = "INSERT INTO users (text_, time_flag_, scale_, speaker_, res_) VALUES (?, ?, ?, ?, ?)"
                cursor.execute(insert_sql,
                               (text, int(time_flag), scale, speaker, str({'base_str': base_str, 'timestamp': []})))
                conn.commit()
                # shutil.rmtree("./outputs")
                return SuccessRequest(result={'base_str': base_str, 'timestamp': []})


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=port, forwarded_allow_ips='*')