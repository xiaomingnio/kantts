import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, dropatt=0.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropatt = nn.Dropout(dropatt)

    def forward(self, q, k, v, mask=None, step=0, puca_id=0, flag=False):
        # print("q, k, v: ", q.shape, k.shape, v.shape)

        attn = torch.bmm(q, k.transpose(1, 2))

        # print(attn.shape)
        attn = attn / self.temperature

        # print("attn: ", attn)
        # print("mask: ", mask.shape)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

            # attn1 = attn.masked_fill(mask, -100000000000)
        # print("attn: ", attn)
        # if flag:
        #     np.save(f"./tensorrt_onnx/data_save/attn_{step}_{puca_id}.npy", attn.detach().cpu().numpy())
        #     np.save(f"./tensorrt_onnx/data_save/mask_{step}_{puca_id}.npy", mask.detach().cpu().numpy())

        attn = self.softmax(attn)
        # if flag:
        #     np.save(f"./tensorrt_onnx/data_save/attnx_softmax_{step}_{puca_id}.npy", attn.detach().cpu().numpy())
        # print("softmax: ", attn)
        # attn1 = self.softmax(attn1)
        # print("diff--------------------")
        # print(torch.mean(attn-attn1))

        attn = self.dropatt(attn)
        output = torch.bmm(attn, v)

        return output, attn


class Prenet(nn.Module):
    def __init__(self, in_units, prenet_units, out_units=0):
        super(Prenet, self).__init__()

        self.fcs = nn.ModuleList()
        for in_dim, out_dim in zip([in_units] + prenet_units[:-1], prenet_units):
            self.fcs.append(nn.Linear(in_dim, out_dim))
            self.fcs.append(nn.ReLU())
            self.fcs.append(nn.Dropout(0.5))

        if out_units:
            self.fcs.append(nn.Linear(prenet_units[-1], out_units))

    def forward(self, input):
        output = input
        for layer in self.fcs:
            # print(output.shape)
            output = layer(output)
            # print(output.shape)
        return output


class MultiHeadSelfAttention(nn.Module):
    """ Multi-Head SelfAttention module """

    def __init__(self, n_head, d_in, d_model, d_head, dropout, dropatt=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_in = d_in
        self.d_model = d_model

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.w_qkv = nn.Linear(d_in, 3 * n_head * d_head)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_head, 0.5), dropatt=dropatt
        )

        self.fc = nn.Linear(n_head * d_head, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask=None, step=0, puca_id=0):
        d_head, n_head = self.d_head, self.n_head

        sz_b, len_in, _ = input.size()

        residual = input

        x = self.layer_norm(input)
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, -1)

        q = q.view(sz_b, len_in, n_head, d_head)
        k = k.view(sz_b, len_in, n_head, d_head)
        v = v.view(sz_b, len_in, n_head, d_head)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_in, d_head)  # (n*b) x l x d
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_in, d_head)  # (n*b) x l x d
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_in, d_head)  # (n*b) x l x d

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_in, d_head)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_in, -1)
        )  # b x l x (n*d)

        output = self.dropout(self.fc(output))
        if output.size(-1) == residual.size(-1):
            output = output + residual

        return output, attn


class PositionwiseConvFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size=(3, 1), dropout_inner=0.1, dropout=0.1):
        super().__init__()
        # Use Conv1D
        # position-wise
        # print("d_in, d_hid, kernel_size=(3, 1), dropout_inner=0.1, dropout=0.1: ", d_in, d_hid, kernel_size, dropout_inner, dropout)
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout_inner = nn.Dropout(dropout_inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, step=0, puca_id=0, flag=False):
        residual = x
        x = self.layer_norm(x)
        # print("layer_norm(x): ", x.shape)
        # print("mask: ", mask)

        output = x.transpose(1, 2)
        output = F.relu(self.w_1(output))

        # if flag:
        #     np.save(f"./tensorrt_onnx/data_save/pos_ffn_w_1_{step}_{puca_id}.npy", output.detach().cpu().numpy())

        # if mask is not None:
        #     output = output.masked_fill(mask.unsqueeze(1), 0)
        output = self.dropout_inner(output)
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)

        output = output + residual

        return output


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_in,
        d_model,
        n_head,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropout_attn=0.0,
        dropout_relu=0.0,
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadSelfAttention(
            n_head, d_in, d_model, d_head, dropout=dropout, dropatt=dropout_attn
        )
        self.pos_ffn = PositionwiseConvFeedForward(
            d_model, d_inner, kernel_size, dropout_inner=dropout_relu, dropout=dropout
        )

    def forward(self, input, mask=None, slf_attn_mask=None):
        output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        output = self.pos_ffn(output, mask=mask)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn


class MultiHeadPNCAAttention(nn.Module):
    """ Multi-Head Attention PNCA module """

    def __init__(self, n_head, d_model, d_mem, d_head, dropout, dropatt=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.d_mem = d_mem

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.w_x_qkv = nn.Linear(d_model, 3 * n_head * d_head)
        self.fc_x = nn.Linear(n_head * d_head, d_model)
        self.w_h_kv = nn.Linear(d_mem, 2 * n_head * d_head)
        self.fc_h = nn.Linear(n_head * d_head, d_model)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_head, 0.5), dropatt=dropatt
        )

        self.dropout = nn.Dropout(dropout)
        self.h_state_size = 0
        self.x_state_size = 0
        self.h_k = torch.tensor([0])
        self.h_v = torch.tensor([0])
        self.x_k = torch.tensor([0])
        self.x_v = torch.tensor([0])

    def update_x_state(self, x):
        d_head, n_head = self.d_head, self.n_head
        # print("d_head, n_head: ", d_head, n_head)

        sz_b, len_x, _ = x.size()
        # print("sz_b, len_x, _: ", sz_b, len_x, _ )

        x_qkv = self.w_x_qkv(x)
        x_q, x_k, x_v = x_qkv.chunk(3, -1)

        x_q = x_q.view(sz_b, len_x, n_head, d_head)
        x_k = x_k.view(sz_b, len_x, n_head, d_head)
        x_v = x_v.view(sz_b, len_x, n_head, d_head)


        x_q = x_q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
        x_k = x_k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)
        x_v = x_v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_head)

        # if not os.path.exists("./tensorrt_onnx/data_save/x_q.npy"):
        #     np.save("./tensorrt_onnx/data_save/x_q.npy", x_q.detach().cpu().numpy())
        # if not os.path.exists("./tensorrt_onnx/data_save/x_k.npy"):
        #     np.save("./tensorrt_onnx/data_save/x_k.npy", x_k.detach().cpu().numpy())
        # if not os.path.exists("./tensorrt_onnx/data_save/x_v.npy"):
        #     np.save("./tensorrt_onnx/data_save/x_v.npy", x_v.detach().cpu().numpy())

        # print("self.x_state_size: ", self.x_state_size)
        if self.x_state_size:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
            # print(self.x_k.shape, self.x_v.shape, x_k.shape, x_v.shape)
            self.x_k = torch.cat([self.x_k, x_k], dim=1)
            self.x_v = torch.cat([self.x_v, x_v], dim=1)
        else:
            self.x_k = x_k
            self.x_v = x_v

        self.x_state_size += len_x

        return x_q, x_k, x_v

    def update_h_state(self, h, step, puca_id):
        # print("self.h_state_size == h.size(1): ", self.h_state_size , h.size(1), h.size(2), h.size(0))
        if self.h_state_size == h.size(1):
            # print("-----------------------")
            return None, None
        # print("========================")
        d_head, n_head = self.d_head, self.n_head

        # H
        sz_b, len_h, _ = h.size()

        h_kv = self.w_h_kv(h)
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_h_kv.npy", h_kv.detach().cpu().numpy())


        # print("h_kv: ", h_kv.shape)
        h_k, h_v = h_kv.chunk(2, -1)
        # print("h_k: ", h_k.shape)
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_h_k.npy", h_k.detach().cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_h_v.npy", h_v.detach().cpu().numpy())

        h_k = h_k.view(sz_b, len_h, n_head, d_head)
        h_v = h_v.view(sz_b, len_h, n_head, d_head)
        # print(h_k.shape, h_v.shape)


        self.h_k = h_k.permute(2, 0, 1, 3).contiguous().view(-1, len_h, d_head)
        self.h_v = h_v.permute(2, 0, 1, 3).contiguous().view(-1, len_h, d_head)
        # print(self.h_k.shape, self.h_v.shape)
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_h_k.npy", self.h_k.detach().cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_h_v.npy", self.h_v.detach().cpu().numpy())

        self.h_state_size += len_h


        return h_k, h_v

    def reset_state(self):
        self.h_k = None
        self.h_v = None
        self.h_state_size = 0
        self.x_k = None
        self.x_v = None
        self.x_state_size = 0

    def forward(self, x, h, mask_x=None, mask_h=None, step=0, puca_id=0):
        # print(f"step: {step}, puca_id: {puca_id}")
        residual = x
        # print("self.h_state_size: ", self.h_state_size)
        self.update_h_state(h, step=step, puca_id=puca_id)
        # print("x: ", x.shape, x)
        # print("self.layer_norm(x): ", self.layer_norm(x))
        # ln = self.layer_norm(x).cpu().numpy()
        # x_ = x.cpu().numpy()
        # import os

        # if not os.path.exists("./tensorrt_onnx/data_save/x.npy"):
        #     np.save("./tensorrt_onnx/data_save/x.npy", x_)

        x_q, x_k, x_v = self.update_x_state(self.layer_norm(x))
        # if step==0 and puca_id==1:
        #     np.save("./tensorrt_onnx/data_save/x_q_0_1.npy", x_q.cpu().numpy())
        #     np.save("./tensorrt_onnx/data_save/x_k_0_1.npy", x_k.cpu().numpy())
        #     np.save("./tensorrt_onnx/data_save/x_v_0_1.npy", x_v.cpu().numpy())

        d_head, n_head = self.d_head, self.n_head

        sz_b, len_in, _ = x.size()

        # X
        if mask_x is not None:
            # print("mask_x.shape: ", mask_x.shape)
            mask_x = mask_x.repeat(n_head, 1, 1)  # (n*b) x .. x ..
            # print("mask_x.shape: ", mask_x.shape)

        # print("--------attention x ------------")
        # np.save(f"./tensorrt_onnx/data_save/x_q_{step}_{puca_id}.npy", x_q.cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/x_k_{step}_{puca_id}.npy", self.x_k.cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/x_v_{step}_{puca_id}.npy", self.x_v.cpu().numpy())
        output_x, attn_x = self.attention(x_q, self.x_k, self.x_v, mask=mask_x, step=step, puca_id= puca_id, flag=True)

        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_output_x.npy", output_x.detach().cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_attn_x.npy",
        #         attn_x.detach().cpu().numpy())

        output_x = output_x.view(n_head, sz_b, len_in, d_head)
        output_x = (
            output_x.permute(1, 2, 0, 3).contiguous().view(sz_b, len_in, -1)
        )  # b x l x (n*d)
        output_x = self.fc_x(output_x)
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_output_x.npy",
        #         output_x.detach().cpu().numpy())
        # H
        if mask_h is not None:
            mask_h = mask_h.repeat(n_head, 1, 1)
        # print("--------attention h ------------")
        output_h, attn_h = self.attention(x_q, self.h_k, self.h_v, mask=mask_h, step=step, puca_id= puca_id, flag=False)
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_output_h.npy", output_h.detach().cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_attn_h.npy",
        #         attn_h.detach().cpu().numpy())

        # 8,1,16
        output_h = output_h.view(n_head, sz_b, len_in, d_head)
        output_h = (
            output_h.permute(1, 2, 0, 3).contiguous().view(sz_b, len_in, -1)
        )  # b x l x (n*d)
        output_h = self.fc_h(output_h)

        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_output_h.npy",
        #         output_h.detach().cpu().numpy())

        output = output_x + output_h

        output = self.dropout(output)

        # print("output: ", output.shape)
        # print("residual: ", residual.shape)
        output = output + residual

        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_output.npy",
        #         output.detach().cpu().numpy())

        return output, attn_x, attn_h


class PNCABlock(nn.Module):
    """PNCA Block"""

    def __init__(
        self,
        d_model,
        d_mem,
        n_head,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropout_attn=0.0,
        dropout_relu=0.0,
    ):
        super(PNCABlock, self).__init__()
        self.pnca_attn = MultiHeadPNCAAttention(
            n_head, d_model, d_mem, d_head, dropout=dropout, dropatt=dropout_attn
        )
        self.pos_ffn = PositionwiseConvFeedForward(
            d_model, d_inner, kernel_size, dropout_inner=dropout_relu, dropout=dropout
        )

    def forward(
        self, input, memory, mask=None, pnca_x_attn_mask=None, pnca_h_attn_mask=None, step=0, puca_id=0
    ):
        # print("mask: ", mask)
        # print("pnca_x_attn_mask: ", pnca_x_attn_mask.shape)
        # print("pnca_h_attn_mask: ", pnca_h_attn_mask.shape)

        #  here 进行中..................
        output, pnca_attn_x, pnca_attn_h = self.pnca_attn(
            input, memory, pnca_x_attn_mask, pnca_h_attn_mask, step=step, puca_id=puca_id
        )

        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_pnca_attn_output.npy",
        #         output.detach().cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_pnca_attn_x.npy",
        #         pnca_attn_x.detach().cpu().numpy())
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_pnca_attn_h.npy",
        #         pnca_attn_h.detach().cpu().numpy())

        # if mask is not None:
        #     output = output.masked_fill(mask.unsqueeze(-1), 0)

        output = self.pos_ffn(output, mask=mask, step=step, puca_id=puca_id, flag=True)
        # if mask is not None:
        # #     output = output.masked_fill(mask.unsqueeze(-1), 0)
        # np.save(f"./tensorrt_onnx/data_save/step_{step}_puca_id_{puca_id}_output.npy",
        #          output.detach().cpu().numpy())

        return output, pnca_attn_x, pnca_attn_h, self.pnca_attn.x_k, self.pnca_attn.x_v

    def reset_state(self):
        self.pnca_attn.reset_state()
