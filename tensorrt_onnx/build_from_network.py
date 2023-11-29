import tensorrt as trt
import numpy as np
import torch
import torch
import struct

wts_path = "./model_save/mel_decode_zhizhe_emo.wts"
engine_path = wts_path.replace(".wts", ".engine")

weights = {}
with open(wts_path) as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        line_s = line.split(" ")
        name = line_s[0]
        data_len = int(line_s[1])

        w = []
        for i in range(data_len):
            d = line_s[3+i]
            bytes_data = bytes.fromhex(d)  # 将十六进制字符串转换为字节数据
            # 解码为单精度浮点数
            float_value = struct.unpack('>f', bytes_data)[0]
            w.append(float_value)
        weights[name] = np.array(w).astype(np.float32)
        # print(name)
        # print(weights[name].shape)

# print(weights)


verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
BATCH_SIZE = 1

d_model = 128
d_mem = 160

#  memory: torch.Size([1, 56, 160])
# x_band_width: 5, h_band_width: 5

EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def prenet(input_tensor):
    in_units = 80  # 根据实际情况设置in_units的值
    prenet_units = [256, 256]  # 根据实际情况设置prenet_units的值
    out_units = 128  # 根据实际情况设置out_units的值
    name_lists = [0, 3]

    prev_layer = input_tensor
    for in_dim, out_dim, n in zip([in_units] + prenet_units[:-1], prenet_units, name_lists):

        reshaped_tensor = network.add_shuffle(prev_layer)
        reshaped_dims = (1, 1, in_dim, 1)  # 添加额外的维度
        reshaped_tensor.reshape_dims = reshaped_dims

        fc_layer = network.add_fully_connected(reshaped_tensor.get_output(0), out_dim, trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.prenet.fcs.{n}.weight"])),
                                               trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.prenet.fcs.{n}.bias"])))
        prev_layer = fc_layer.get_output(0)

        relu_layer = network.add_activation(prev_layer, trt.ActivationType.RELU)
        prev_layer = relu_layer.get_output(0)


    fc_layer = network.add_fully_connected(prev_layer, out_units,
                                           trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.prenet.fcs.6.weight"])),
                                           trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.prenet.fcs.6.bias"])))
    output_tensor = fc_layer.get_output(0)

    reshaped_tensor = network.add_shuffle(output_tensor)
    reshaped_dims = (1, 1, out_units)  # 添加额外的维度
    reshaped_tensor.reshape_dims = reshaped_dims

    return reshaped_tensor.get_output(0)


def pos_ffn(input, puca_id):
    #     def forward(self, x, mask=None):
    #         residual = x
    #         x = self.layer_norm(x)
    #
    #         output = x.transpose(1, 2)
    #         output = F.relu(self.w_1(output))

    #         output = self.dropout_inner(output)
    #         output = self.w_2(output)
    #         output = output.transpose(1, 2)
    #         output = self.dropout(output)
    #
    #         output = output + residual
    #
    #         return output
    d_head, n_head = 16, 8
    sz_b, len_x, _ = 1, 1, 128
    layernorm_out = layernorm(input, layer_name=f"mel_decoder.mel_dec.pnca.{puca_id}.pos_ffn.layer_norm") # (1,1,128)

    re_ln = network.add_shuffle(layernorm_out.get_output(0))
    re_ln.reshape_dims = (1, 128, 1, 1)

    # (1, 128, 1, 1)
    w_1 = network.add_convolution_nd(
            input=re_ln.get_output(0),
            num_output_maps=1024,
            kernel_shape=(1, 1),
            kernel=trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pos_ffn.w_1.weight"])),
            bias=trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pos_ffn.w_1.bias"]))
    )
    w_1_out = network.add_activation(w_1.get_output(0), trt.ActivationType.RELU)

    w_2 = network.add_convolution_nd(
        input=w_1_out.get_output(0),
        num_output_maps=128,
        kernel_shape=(1, 1),
        kernel=trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pos_ffn.w_2.weight"])),
        bias=trt.Weights(np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pos_ffn.w_2.bias"]))
    )

    re_w_2 = network.add_shuffle(w_2.get_output(0))
    re_w_2.reshape_dims = (1, 128, 1)
    re_w_2.second_transpose = trt.Permutation([0, 2, 1])

    output = network.add_elementwise(input, re_w_2.get_output(0), trt.ElementWiseOperation.SUM)

    return output

def pnca_attn(input, memory, mask_x_part1, mask_x_part2, mask_h, pre_x_k, pre_x_v, puca_id):
    d_head, n_head = 16, 8
    sz_b, len_x, _ = 1, 1, 128
    layernorm_out = layernorm(input, layer_name=f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.layer_norm")
    x_q, x_k, x_v = update_x_state(layernorm_out.get_output(0), pre_x_k.get_output(0), pre_x_v.get_output(0), puca_id=puca_id)
    # out = update_h_state(memory)

    # mask_x = mask_x.repeat(n_head, 1, 1)  # (n*b) x .. x ..
    # mask_x = trt_repeat(pnca_x_attn_mask, repeat_nums=n_head)

    # output_x, attn_x = self.attention(x_q, self.x_k, self.x_v, mask=mask_x)
    # print("---------------x--------------------")
    # print(x_q.get_output(0).shape, x_k.get_output(0).shape, x_v.get_output(0).shape, mask_x_part1.shape)
    output_x, attn_x = ScaledDotProductAttention(x_q, x_k, x_v, mask_x_part1, mask_another=mask_x_part2, name=f"x_{puca_id}")

    # output_x = output_x.view(n_head, sz_b, len_in, d_head)
    #         output_x = (
    #             output_x.permute(1, 2, 0, 3).contiguous().view(sz_b, len_in, -1)
    #         )  # b x l x (n*d)
    re_output_x = network.add_shuffle(output_x.get_output(0))
    re_output_x.reshape_dims = (n_head, sz_b, len_x, d_head)
    re_output_x.second_transpose = trt.Permutation([1, 2, 0, 3])
    re_output_x = network.add_shuffle(re_output_x.get_output(0))
    re_output_x.reshape_dims = (sz_b, len_x, -1, 1)

    # output_x = self.fc_x(output_x)   self.fc_x = nn.Linear(n_head * d_head, d_model)
    output_x_fc = network.add_fully_connected(re_output_x.get_output(0), d_model, trt.Weights(
        np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.fc_x.weight"])),
                                                 trt.Weights(np.ascontiguousarray(
                                                     weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.fc_x.bias"])))

    # mask_h = mask_h.repeat(n_head, 1, 1)
    # mask_h = trt_repeat(pnca_h_attn_mask, repeat_nums=n_head)

    # 当step=0时，更新self.h_k, self.h_v
    # if step == 0:
    #     # self.update_h_state(h)
    h_k, h_v = update_h_state(memory, puca_id=puca_id)

    # output_h, attn_h = self.attention(x_q, self.h_k, self.h_v, mask=mask_h)
    # h_k (8,-1,16)
    # print("---------------h--------------------")
    # print(x_q.get_output(0).shape, h_k.get_output(0).shape, h_v.get_output(0).shape, mask_h.shape)
    output_h, attn_h = ScaledDotProductAttention(x_q, h_k, h_v, mask_h, name=f"h_{puca_id}")

    re_output_h = network.add_shuffle(output_h.get_output(0))
    re_output_h.reshape_dims = (n_head, sz_b, len_x, d_head)
    re_output_h.second_transpose = trt.Permutation([1, 2, 0, 3])
    re_output_h = network.add_shuffle(re_output_h.get_output(0))
    re_output_h.reshape_dims = (sz_b, len_x, -1, 1)

    output_h_fc = network.add_fully_connected(re_output_h.get_output(0), d_model, trt.Weights(
        np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.fc_h.weight"])),
                                              trt.Weights(np.ascontiguousarray(
                                                  weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.fc_h.bias"])))

    output = network.add_elementwise(output_x_fc.get_output(0), output_h_fc.get_output(0), trt.ElementWiseOperation.SUM)
    re_output = network.add_shuffle(output.get_output(0))
    re_output.reshape_dims = (1, 1, 128)
    #
    output = network.add_elementwise(re_output.get_output(0), input, trt.ElementWiseOperation.SUM)

    return output, attn_x, attn_h, x_k, x_v


def trt_repeat(input, repeat_nums):
    repeated_tensors = []
    for _ in range(repeat_nums):
        repeated_tensors.append(input)
    concat_layer = network.add_concatenation(inputs=repeated_tensors)
    return concat_layer


def ScaledDotProductAttention(q, k, v, mask, mask_another=None, name=''):
    d_head, n_head = 16, 8

    # q, k, v:  torch.Size([8, 1, 16]) torch.Size([8, 1, 16]) torch.Size([8, 1, 16])
    # q, k, v:  torch.Size([8, 1, 16]) torch.Size([8, 70, 16]) torch.Size([8, 70, 16])
    #  attn = torch.bmm(q, k.transpose(1, 2))
    # print(q.get_output(0).shape, k.get_output(0).shape)
    attn0 = network.add_matrix_multiply(q.get_output(0), trt.MatrixOperation.NONE,
                                    k.get_output(0), trt.MatrixOperation.TRANSPOSE) # (8, 1, -1)
    # temperature=np.power(d_head, 0.5)
    temperature = np.power(d_head, 0.5)
    # attn = attn / self.temperature
    div_const = network.add_constant((1,1,1), temperature*np.ones((1,1,1)).astype(np.float32))
    div_0 = network.add_elementwise(attn0.get_output(0), div_const.get_output(0), trt.ElementWiseOperation.DIV)
    div_0.name = f"{name}_div_0"
    # attn = attn.masked_fill(mask, -np.inf)
    # mask = mask*(-100000000000000000)
    inf_const = network.add_constant((1,1,1), (-100000000000000000.0)*np.ones((1,1,1)).astype(np.float32))
    mask = network.add_elementwise(mask, inf_const.get_output(0), trt.ElementWiseOperation.PROD)


    # print("attn.get_output(0).shape, mask.get_output(0).shape")
    # print(attn.get_output(0).shape, mask.get_output(0).shape)

    if name[:2] == "x_":
        mask_another = network.add_elementwise(mask_another, inf_const.get_output(0), trt.ElementWiseOperation.PROD)
        # const = network.add_constant((8, 1, 1), trt.Weights(np.ones((8, 1, 1)).astype(np.float32)))

        cat_tensors0 = [mask.get_output(0), mask_another.get_output(0)]
        mask_cat0 = network.add_concatenation(cat_tensors0)
        mask_cat0.axis = 2
        mask_cat0.name = f"{name}_cat0"

        attn1 = network.add_elementwise(div_0.get_output(0), mask_cat0.get_output(0), trt.ElementWiseOperation.SUM)
        attn1.name = f"{name}_attn"

        attn_softmax = network.add_softmax(attn1.get_output(0))  # (8, 1, 1)
        attn_softmax.axes = 1 << 2  # axes是位压缩的mask

        # print("---: ", attn_softmax.get_output(0).shape, v.get_output(0).shape)
        attn = network.add_matrix_multiply(attn_softmax.get_output(0), trt.MatrixOperation.NONE,
                                           v.get_output(0), trt.MatrixOperation.NONE)  # (8, 1, 16)
        attn.name = f"{name}_attn1"

        return attn, attn_softmax
    elif name[:2] == "h_":
        attn = network.add_elementwise(div_0.get_output(0), mask.get_output(0), trt.ElementWiseOperation.SUM)
        attn.name = f"{name}_attn"

        attn_softmax = network.add_softmax(attn.get_output(0))  # (8, 1, 1)
        attn_softmax.axes = 1<<2 # axes是位压缩的mask

        # print("---: ", attn_softmax.get_output(0).shape, v.get_output(0).shape)
        attn = network.add_matrix_multiply(attn_softmax.get_output(0), trt.MatrixOperation.NONE,
                                           v.get_output(0), trt.MatrixOperation.NONE)  # (8, 1, 16)
        attn.name = f"{name}_attn1"
        return attn, attn_softmax


def update_x_state(input_tensor, pre_x_k, pre_x_v, puca_id=0):
    # print("input_tensor, pre_x_k, pre_x_v: ", input_tensor.shape, pre_x_k.shape, pre_x_v.shape)
    d_head, n_head = 16, 8
    sz_b, len_x, _ = 1, 1, 128
    # x_qkv = self.w_x_qkv(x)  self.w_x_qkv = nn.Linear(d_model, 3 * n_head * d_head)
    reshaped_tensor = network.add_shuffle(input_tensor)
    reshaped_dims = (1, 1, 128, 1)  # 添加额外的维度
    reshaped_tensor.reshape_dims = reshaped_dims
    fc_layer_x_qkv = network.add_fully_connected(reshaped_tensor.get_output(0), 3 * n_head * d_head, trt.Weights(
        np.ascontiguousarray(weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.w_x_qkv.weight"])),
                                           trt.Weights(np.ascontiguousarray(
                                               weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.w_x_qkv.bias"])))

    # x_q, x_k, x_v = x_qkv.chunk(3, -1)  (1, 128*3, 1, 1)
    x_q = network.add_slice(fc_layer_x_qkv.get_output(0), start=(0, 0, 0, 0), shape=(1, 128, 1, 1), stride=[1, 1, 1, 1])
    x_k = network.add_slice(fc_layer_x_qkv.get_output(0), start=(0, 128, 0, 0), shape=(1, 128, 1, 1), stride=[1, 1, 1, 1])
    x_v = network.add_slice(fc_layer_x_qkv.get_output(0), start=(0, 128*2, 0, 0), shape=(1, 128, 1, 1), stride=[1, 1, 1, 1])

    #         x_q = x_q.view(sz_b, len_x, n_head, d_head)
    #         x_k = x_k.view(sz_b, len_x, n_head, d_head)
    #         x_v = x_v.view(sz_b, len_x, n_head, d_head)
    #   x_q = x_q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
    #   x_k = x_k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
    #   x_v = x_v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
    reshaped_dims = (sz_b, len_x, n_head, d_head)
    re_x_q = network.add_shuffle(x_q.get_output(0))
    re_x_q.reshape_dims = reshaped_dims
    re_x_q.second_transpose = trt.Permutation([2, 0, 1, 3])
    re_x_q = network.add_shuffle(re_x_q.get_output(0))
    re_x_q.reshape_dims = (-1, len_x, d_head)

    re_x_k = network.add_shuffle(x_k.get_output(0))
    re_x_k.reshape_dims = reshaped_dims
    re_x_k.second_transpose = trt.Permutation([2, 0, 1, 3])
    re_x_k = network.add_shuffle(re_x_k.get_output(0))
    re_x_k.reshape_dims = (-1, len_x, d_head)

    re_x_v = network.add_shuffle(x_v.get_output(0))
    re_x_v.reshape_dims = reshaped_dims
    re_x_v.second_transpose = trt.Permutation([2, 0, 1, 3])
    re_x_v = network.add_shuffle(re_x_v.get_output(0))
    re_x_v.reshape_dims = (-1, len_x, d_head)

    # if step!=0:
    #     self.x_k = torch.cat([self.x_k, x_k], dim=1)
    #     self.x_v = torch.cat([self.x_v, x_v], dim=1)
    # else:
    #     self.x_k = x_k
    #     self.x_v = x_v
    # print("pre_x_k, re_x_k.get_output(0): ", pre_x_k.shape, re_x_k.get_output(0).shape)
    cat_tensors = [pre_x_k, re_x_k.get_output(0)]
    cat_x_k = network.add_concatenation(cat_tensors)
    cat_x_k.axis = 1

    # cat_x_k = network.add_shuffle(cat_x_k.get_output(0))
    # cat_x_k.reshape_dims = (n_head, len_x, -1)

    cat_tensors = [pre_x_v, re_x_v.get_output(0)]
    cat_x_v = network.add_concatenation(cat_tensors)
    cat_x_v.axis = 1

    # cat_x_v = network.add_shuffle(cat_x_v.get_output(0))
    # cat_x_v.reshape_dims = (n_head, len_x, -1)
    # print(pre_x_k.shape, re_x_k.get_output(0).shape)
    # print("re_x_q: ", re_x_q.get_output(0).shape)
    # print("cat_x_k: ", cat_x_k.get_output(0).shape)
    # print("cat_x_v: ", cat_x_v.get_output(0).shape)

    return re_x_q, cat_x_k, cat_x_v

def update_h_state(input_tensor, puca_id=0):
    # 1, -1, 160

    d_head, n_head = 16, 8
    sz_b, len_x, _ = 1, -1, 160

    # h_kv = self.w_h_kv(h)
    w_h_kv_weight = network.add_constant((1, 2 * n_head * d_head, d_mem), weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.w_h_kv.weight"])
    w_h_kv_bias = network.add_constant((1, 1, 2 * n_head * d_head), weights[f"mel_decoder.mel_dec.pnca.{puca_id}.pnca_attn.w_h_kv.bias"])
    mm0 = network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE,
                                    w_h_kv_weight.get_output(0), trt.MatrixOperation.TRANSPOSE)
    w_h_kv = network.add_elementwise(mm0.get_output(0), w_h_kv_bias.get_output(0), trt.ElementWiseOperation.SUM)

    #
    # h_k, h_v = h_kv.chunk(2, -1)
    #h_kv:  torch.Size([1, 70, 256])
    #h_k:  torch.Size([1, 70, 128])
    input_shape = network.add_shape(w_h_kv.get_output(0))
    pro = network.add_constant(shape=(3,), weights=np.array([0, 1, 0], dtype=np.int32))
    new_shape = network.add_elementwise(input_shape.get_output(0), pro.get_output(0), trt.ElementWiseOperation.PROD)
    sum_ = network.add_constant(shape=(3,), weights=np.array([1, 0, 128], dtype=np.int32))
    new_shape = network.add_elementwise(new_shape.get_output(0), sum_.get_output(0),
                                        trt.ElementWiseOperation.SUM)

    h_k = network.add_slice(w_h_kv.get_output(0), start=(0, 0, 0), shape=(1, 1, 128),
                             stride=[1, 1, 1])
    h_k.set_input(2, new_shape.get_output(0))

    h_v = network.add_slice(w_h_kv.get_output(0), start=(0, 0, 128), shape=(1, 1, 128),
                             stride=[1, 1, 1])
    h_v.set_input(2, new_shape.get_output(0))

    reshaped_dims = (sz_b, -1, n_head, d_head) # 1,70,8,16

    re_h_k = network.add_shuffle(h_k.get_output(0), )
    re_h_k.reshape_dims = reshaped_dims
    re_h_k.second_transpose = trt.Permutation([2, 0, 1, 3])
    re_h_k = network.add_shuffle(re_h_k.get_output(0))
    re_h_k.reshape_dims = (n_head, -1, d_head)

    re_h_v = network.add_shuffle(h_v.get_output(0), )
    re_h_v.reshape_dims = reshaped_dims
    re_h_v.second_transpose = trt.Permutation([2, 0, 1, 3])
    re_h_v = network.add_shuffle(re_h_v.get_output(0))
    re_h_v.reshape_dims = (n_head, -1, d_head)

    return re_h_k, re_h_v


def layernorm(input_tensor, layer_name=''):
    # 计算均值和方差
    mean_tensor = network.add_reduce(input_tensor, trt.ReduceOperation.AVG, 1<<2, keep_dims=True)
    sub_tensor = network.add_elementwise(input_tensor, mean_tensor.get_output(0),
                                            trt.ElementWiseOperation.SUB)

    pow_const = network.add_constant((1, 1, 1), np.array([[[[2]]]], dtype=np.float32))
    pow_tensor = network.add_elementwise(sub_tensor.get_output(0), pow_const.get_output(0),
                                            trt.ElementWiseOperation.POW)
    var_tensor_0 = network.add_reduce(pow_tensor.get_output(0), trt.ReduceOperation.AVG, 1<<2, keep_dims=True)

    eps = np.ones((1,1,128)).reshape((1,1,128)) * 1e-6
    eps = np.ascontiguousarray(eps.astype(np.float32))

    eps_const = network.add_constant(shape=(1, 1, 128), weights=trt.Weights(eps))

    var_tensor_1 = network.add_elementwise(var_tensor_0.get_output(0), eps_const.get_output(0), trt.ElementWiseOperation.SUM)

    sqrt_const = network.add_constant(shape=(1, 1, 1), weights=np.array([[[[0.5]]]], dtype=np.float32))
    sqrt_tensor = network.add_elementwise(var_tensor_1.get_output(0), sqrt_const.get_output(0),
                                            trt.ElementWiseOperation.POW)

    # 添加可学习的gamma和beta参数
    gamma = network.add_constant(shape=(1, 1, d_model), weights=weights[f"{layer_name}.weight"].reshape(1,1,128))
    beta = network.add_constant(shape=(1, 1, d_model), weights=weights[f"{layer_name}.bias"].reshape(1,1,128))

    #  (input - mean)/sqrt
    div_0 = network.add_elementwise(sub_tensor.get_output(0), sqrt_tensor.get_output(0), trt.ElementWiseOperation.DIV)
    # beta,gamma

    matmul_0 = network.add_elementwise(div_0.get_output(0), gamma.get_output(0), trt.ElementWiseOperation.PROD)

    add_2 = network.add_elementwise(matmul_0.get_output(0), beta.get_output(0), trt.ElementWiseOperation.SUM)

    return add_2

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config(
) as config, builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    # step = 0  # 根据实际情况设置step的值

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


    # shape = (1, 1, 80)
    # input = network.add_constant(shape, trt.Weights(np.zeros(shape, dtype=np.float32)))

    input = prenet(input)

    # memory slice
    # memory_shape = (1, -1, 160)  # 根据实际情况设置memory_shape的值
    #
    # step = 0
    # slice_dims = trt.Dims3(memory_shape[0], 1, memory_shape[2])
    # start = [0, step, 0]
    # shape = [memory_shape[0], 1, memory_shape[2]]
    #
    # slice_layer = network.add_slice(memory, start, shape, [1, 1, 1])

    # cat memory and input
    cat_tensors = [memory_step, input]
    concat_layer = network.add_concatenation(cat_tensors)
    concat_layer.axis = 2  # 设置拼接的轴为1
    # output_tensor = concat_layer.get_output(0)

    # dec_in_proj = nn.Linear(d_model + d_mem, d_model)  288, 128
    reshaped_tensor = network.add_shuffle(concat_layer.get_output(0))
    reshaped_dims = (1, 1, 288, 1)  # 添加额外的维度
    reshaped_tensor.reshape_dims = reshaped_dims

    fc_layer = network.add_fully_connected(reshaped_tensor.get_output(0), 128, trt.Weights(
        np.ascontiguousarray(weights[f"mel_decoder.mel_dec.dec_in_proj.weight"])),
                                           trt.Weights(np.ascontiguousarray(
                                               weights[f"mel_decoder.mel_dec.dec_in_proj.bias"])))

    reshaped_tensor = network.add_shuffle(fc_layer.get_output(0))
    reshaped_dims = (1, 1, 128)  # 添加额外的维度
    reshaped_tensor.reshape_dims = reshaped_dims
    # input *= self.d_model ** 0.5
    # 创建标量张量
    scalar = network.add_constant((1,1,128), trt.Weights(128**0.5*np.ones(shape=(1,1,128), dtype=np.float32)))

    # 添加 ElementWise 层
    elementwise_layer = network.add_elementwise(reshaped_tensor.get_output(0), scalar.get_output(0),
                                                trt.ElementWiseOperation.PROD)


    # for id, layer in enumerate(self.pnca)
    # pnca_x_attn_mask=pnca_x_attn_mask[:, step : step + 1, : (step + 1)],
    # pnca_h_attn_mask=pnca_h_attn_mask[:, step : step + 1, :],
    # start = [0, step, 0]
    # shape = [1, 1, step]

    # print(pnca_h_attn_mask_shape.get_output(0))
    # pnca_x_attn_mask_slice_layer = network.add_slice(pnca_x_attn_mask, start=[0, step, 0], shape=[1, 1, step+1], stride=[1, 1, 1])
    # pnca_h_attn_mask_slice_layer = network.add_slice(pnca_h_attn_mask, start=[0, step, 0], shape=[1,1,1], stride=[1, 1, 1])
    # pnca_h_attn_mask_slice_layer.set_input(2, new_shape.get_output(0))
    pre_x_k_list = []
    pre_x_v_list = []
    input_shape = network.add_shape(pre_x_k)
    pro = network.add_constant(shape=(3,), weights=np.array([0, 1, 0], dtype=np.int32))
    new_shape = network.add_elementwise(input_shape.get_output(0), pro.get_output(0), trt.ElementWiseOperation.PROD)
    sum_ = network.add_constant(shape=(3,), weights=np.array([8, 0, 16], dtype=np.int32))
    new_shape = network.add_elementwise(new_shape.get_output(0), sum_.get_output(0),
                                        trt.ElementWiseOperation.SUM)
    for i in range(12):
        pre_x_k_tmp = network.add_slice(pre_x_k, start=(i*8, 0, 0), shape=(1, 1, 1),
                                stride=[1, 1, 1])
        pre_x_k_tmp.set_input(2, new_shape.get_output(0))
        pre_x_k_list.append(pre_x_k_tmp)

        pre_x_v_tmp = network.add_slice(pre_x_v, start=(i*8, 0, 0), shape=(1, 1, 1),
                                stride=[1, 1, 1])
        pre_x_v_tmp.set_input(2, new_shape.get_output(0))
        pre_x_v_list.append(pre_x_v_tmp)

    # print()

    dec_pnca_attn_x_list = []
    dec_pnca_attn_h_list = []
    #  puca 0
    puca_id = 0
    dec_output, dec_pnca_attn_x, dec_pnca_attn_h, x_k, x_v = pnca_attn(elementwise_layer.get_output(0), memory,
                                                 pnca_x_attn_mask_step_part1, pnca_x_attn_mask_step_part2,
                                                 pnca_h_attn_mask_step, pre_x_k_list[puca_id], pre_x_v_list[puca_id], puca_id)

    dec_output = pos_ffn(dec_output.get_output(0), 0)

    dec_pnca_attn_x_0 = dec_pnca_attn_x.get_output(0)
    dec_pnca_attn_x_0.name = "dec_pnca_attn_x_0"
    network.mark_output(dec_pnca_attn_x_0)

    dec_pnca_attn_h_0 = dec_pnca_attn_h.get_output(0)
    dec_pnca_attn_h_0.name = "dec_pnca_attn_h_0"
    network.mark_output(dec_pnca_attn_h_0)

    x_k_0 = x_k.get_output(0)
    x_k_0.name = "x_k_0"
    network.mark_output(x_k_0)

    x_v_0 = x_v.get_output(0)
    x_v_0.name = "x_v_0"
    network.mark_output(x_v_0)


    for id in range(11):
        puca_id = id+1
        # print("==== puca_id: ", puca_id, " ====")
        # print("dec_output: ", dec_output.get_output(0).shape)
        #  puca 1
        dec_output, dec_pnca_attn_x, dec_pnca_attn_h, x_k, x_v = pnca_attn(dec_output.get_output(0), memory,
                                                     pnca_x_attn_mask_step_part1, pnca_x_attn_mask_step_part2,
                                                     pnca_h_attn_mask_step,pre_x_k_list[puca_id], pre_x_v_list[puca_id],
                                                     puca_id)
        #
        dec_output = pos_ffn(dec_output.get_output(0), puca_id)

        dec_pnca_attn_x_tmp = dec_pnca_attn_x.get_output(0)
        dec_pnca_attn_x_tmp.name = f"dec_pnca_attn_x_{puca_id}"
        network.mark_output(dec_pnca_attn_x_tmp)

        dec_pnca_attn_h_tmp = dec_pnca_attn_h.get_output(0)
        dec_pnca_attn_h_tmp.name = f"dec_pnca_attn_h_{puca_id}"
        network.mark_output(dec_pnca_attn_h_tmp)

        x_k_tmp = x_k.get_output(0)
        x_k_tmp.name = f"x_k_{puca_id}"
        network.mark_output(x_k_tmp)

        x_v_tmp = x_v.get_output(0)
        x_v_tmp.name = f"x_v_{puca_id}"
        network.mark_output(x_v_tmp)


    # dec_output = self.ln(dec_output)
    # dec_output = self.dec_out_proj(dec_output) d_model, d_out:  128 240

    dec_output_ln = layernorm(dec_output.get_output(0), layer_name='mel_decoder.mel_dec.ln')

    reshaped_tensor = network.add_shuffle(dec_output_ln.get_output(0))
    reshaped_dims = (1, 1, 128, 1)  # 添加额外的维度
    reshaped_tensor.reshape_dims = reshaped_dims

    dec_out_proj = network.add_fully_connected(reshaped_tensor.get_output(0), 240, trt.Weights(
        np.ascontiguousarray(weights[f"mel_decoder.mel_dec.dec_out_proj.weight"])),
                                           trt.Weights(np.ascontiguousarray(
                                               weights[f"mel_decoder.mel_dec.dec_out_proj.bias"])))

    dec_output = network.add_shuffle(dec_out_proj.get_output(0))
    reshaped_dims = (1, 1, 240)  # 添加额外的维度
    dec_output.reshape_dims = reshaped_dims

    output_ = dec_output.get_output(0)
    output_.name = "output"
    network.mark_output(output_)


    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 1, 80), (1, 1, 80), (1, 1, 80))
    profile.set_shape('memory', (1, 1, 160), (1, 75, 160), (1, 1000, 160))
    profile.set_shape('memory_step', (1, 1, 160), (1, 1, 160), (1, 1, 160))
    # profile.set_shape('step', (1, ), (75, ), (1000, ))
    profile.set_shape('pnca_x_attn_mask_step_part1', (8, 1, 1), (8, 1, 75), (8, 1, 1000))
    profile.set_shape('pnca_x_attn_mask_step_part2', (8, 1, 1), (8, 1, 1), (8, 1, 1))
    profile.set_shape('pnca_h_attn_mask_step', (8, 1, 1), (8, 1, 75), (8, 1, 1000))
    profile.set_shape('pre_x_k', (8*12, 1, 16), (8*12, 75, 16), (8*12, 1000, 16))
    profile.set_shape('pre_x_v', (8*12, 1, 16), (8*12, 75, 16), (8*12, 1000, 16))
    # profile.set_shape('pre_x_k_idx', (1, ), (75, ), (1000, ))
    # profile.set_shape('pre_x_v_idx', (1, ), (75, ), (1000, ))
    config.add_optimization_profile(profile)

    builder.max_batch_size = 1
    config.max_workspace_size = 1 << 30
    engine = builder.build_serialized_network(network, config)
    # print(engine)
    with open(engine_path, mode='wb') as f:
        f.write(engine)
        print("generating file done!")