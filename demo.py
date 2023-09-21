from tts_infer import TTS

from os.path import expanduser
home_directory = expanduser("~")
print(home_directory)

# zhibei_emo  zhitian_emo  zhiyan_emo  zhizhe_emo
sambert_hifigan_tts_man = TTS(basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k" ,voice ="zhizhe_emo")
outp = sambert_hifigan_tts_man.infer(text="提到个性化定制语音，大家并不陌生", scale=1)
print(outp)



# sambert_hifigan_tts_women = TTS(basepath=f"{home_directory}/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k" ,voice ="zhitian_emo")
# outp1 = sambert_hifigan_tts_women.infer(text="提到个性化定制语音，大家并不陌生", scale=1)
# print("加载tts模型成功！")