# kantts
TTS appalication based on modelscope KAN-TTS

## demo
`python demo.py`

## Features - 特性
- 增加语速调节，scale参数调节
- 推理加速
  
  1.HifiGAN tensorrt加速
  
  2.sambert tensorrt加速

## HifiGAN tensorrt加速
```
cd tensorrt_onnx
# 导出hifigan onnx
python hifigan_onnx_export.py 
# 利用trt.OnnxParser生成engine
python build_from_onnx.py
```
本项目原始模型直接利用modelscope下载
```
from modelscope.utils.constant import Tasks
model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id)
text = "你好"
output = sambert_hifigan_tts(input=text, voice="zhitian_emo")  # zhibei_emo  zhitian_emo zhiyan_emo  zhizhe_emo
```

## sambert tensorrt加速
1、统计sambert各个模块耗时，发现主要耗时都集中在MelPNCADecoder； 
2、将MelPNCADecoder部分由一个循环的mel_dec函数组成； 
它在memory的第二个维度上循环调用，下一次调用会依赖上一次的结果
3、将mel_dec函数利用tensorrt python api重写，因为mel_del的调用会依赖上一次mel_del的中间变量和输出结果，故修改该函数输入输出，将需要的中间结果都输入到下一次调用； 

step=0时和step>0时的输入不一致，所有只对step>0时的推理部分进行tensorrt搭建。

输入部分：
```
input = network.add_input(
    name="input", dtype=trt.float32, shape=(1, 1, 80))

memory = network.add_input(
    name="memory", dtype=trt.float32, shape=(1, -1, 160))

memory_step = network.add_input(
    name="memory_step", dtype=trt.float32, shape=(1, 1, 160))

pnca_x_attn_mask_step_part1 = network.add_input(
    name="pnca_x_attn_mask_step_part1", dtype=trt.float32, shape=(8, 1, -1))
pnca_x_attn_mask_step_part2 = network.add_input(
    name="pnca_x_attn_mask_step_part2", dtype=trt.float32, shape=(8, 1, 1))

pnca_h_attn_mask_step = network.add_input(
    name="pnca_h_attn_mask_step", dtype=trt.float32, shape=(8, 1, -1))

pre_x_k = network.add_input(
    name="pre_x_k", dtype=trt.float32, shape=(8*12, -1, 16))

pre_x_v = network.add_input(
    name="pre_x_v", dtype=trt.float32, shape=(8*12, -1, 16))
```

推理输入输出转换
```
pnca_x_attn_mask_step_ = pnca_x_attn_mask_step.repeat(8, 1, 1)
pnca_h_attn_mask_step_ = pnca_h_attn_mask_step.repeat(8, 1, 1)

pnca_x_attn_mask_step_part1 = pnca_x_attn_mask_step_[:, :, :-1]
pnca_x_attn_mask_step_part2 = pnca_x_attn_mask_step_[:, :, -1:]

pre_x_k_merge = None
pre_x_v_merge = None
for i in range(len(pre_x_k_list)):
    if i == 0:
        pre_x_k_merge = pre_x_k_list[i]
        pre_x_v_merge = pre_x_v_list[i]
    else:
        pre_x_k_merge = torch.cat([pre_x_k_merge, pre_x_k_list[i]], dim=0)
        pre_x_v_merge = torch.cat([pre_x_v_merge, pre_x_v_list[i]], dim=0)
    # print(pre_x_k_merge.shape)
    # print(pre_x_v_merge.shape)

output_trt = self.trt_model({"input": input, "memory": memory, "memory_step": memory_step,
                    "pnca_x_attn_mask_step_part1": pnca_x_attn_mask_step_part1.float(),
                    "pnca_x_attn_mask_step_part2": pnca_x_attn_mask_step_part2.float(),
                    "pnca_h_attn_mask_step": pnca_h_attn_mask_step_.float(),
                    "pre_x_k": pre_x_k_merge,
                    "pre_x_v": pre_x_v_merge})

dec_output_step = output_trt['output']
dec_pnca_attn_x_step = []
dec_pnca_attn_h_step = []
for i in range(12):
    dec_pnca_attn_x_step += [output_trt[f'dec_pnca_attn_x_{i}']]
    dec_pnca_attn_h_step += [output_trt[f'dec_pnca_attn_h_{i}']]

pre_x_k_list = []
pre_x_v_list = []
for i in range(12):
    pre_x_k_list += [output_trt[f'x_k_{i}']]
    pre_x_v_list += [output_trt[f'x_v_{i}']]
 ```

 模型搭建部分
 masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于待填充的相应位置用value填充。
例如本项目中，需要实现 
attn = attn.masked_fill(mask, -np.inf)
填充值为负无穷
trt实现步骤
● 将mask由bool转换成float32，变为由0和1组成的tensor
● 然后将mask利用逐元素相乘的方式，乘以一个很大的负数，这里设置为-100000000000000000.0
 ```
inf_const = network.add_constant((1,1,1), (-100000000000000000.0)*np.ones((1,1,1)).astype(np.float32))
mask = network.add_elementwise(mask, inf_const.get_output(0), trt.ElementWiseOperation.PROD)
 ```
● 最后attn和mask逐元素相加

layernorm算子可以使用8.6的tensorrt，也可以自己按照公式实现
