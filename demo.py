from tts import TTS
import time

from os.path import expanduser
home_directory = expanduser("~")
print(home_directory)

infer_type = "trt"

#   zhitian_emo    zhizhe_emo
sambert_hifigan_tts_man = TTS(basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k",
                              voice="zhizhe_emo", infer_type=infer_type)
outp = sambert_hifigan_tts_man.infer(text="提到个性化定制语音，大家并不陌生", scale=1)
# print(outp)

ts = time.time()
with open("test.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        text = line.strip()
        print("字数: ", len(text))
        t0 = time.time()
        outp = sambert_hifigan_tts_man.infer(text=text, scale=1)
        t1 = time.time()
        print("----infer time: ", t1-t0)
te = time.time()
print(infer_type, "\n")
print(te-ts)
#sambert_hifigan_tts_women = TTS(basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k" ,voice ="zhitian_emo")
# outp1 = sambert_hifigan_tts_women.infer(text="提到个性化定制语音，大家并不陌生", scale=1)
# print("加载tts模型成功！")