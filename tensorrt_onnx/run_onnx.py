# -*-coding: utf-8 -*-

import os, sys
import torch
sys.path.append(os.getcwd())
import onnxruntime
import onnx

import numpy as np
import torchvision.transforms as transforms
import time


class ONNXModel():
    def __init__(self, onnx_path, device='cpu'):
        """
        :param onnx_path:
        """
        sess_options = onnxruntime.SessionOptions()

        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        print(device)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
        print("-----------: ", providers)
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name


    # def forward(self, image_numpy):
    #     '''
    #     # image_numpy = image.transpose(2, 0, 1)
    #     # image_numpy = image_numpy[np.newaxis, :]
    #     # onnx_session.run([output_name], {input_name: x})
    #     # :param image_numpy:
    #     # :return:
    #     '''
    #     # 输入数据的类型必须与模型一致,以下三种写法都是可以的
    #     # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
    #     # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
    #     input_feed = self.get_input_feed(self.input_name, image_numpy)
    #     scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
    #     return scores, boxes


# import onnx
#
# # 加载 ONNX 模型
# onnx_model = onnx.load("./model.onnx")
#
#
# for inp in onnx_model.graph.input:
#     print("----------------------")
#     print(inp.name)
#     # 打印输入形状
#     print("Input shape:")
#     for dim in inp.type.tensor_type.shape.dim:
#         print(dim.dim_value)
#
#
# for outp in onnx_model.graph.output:
#     print("----------------------")
#     print(outp.name)
#     # 打印输入形状
#     print("outp shape:")
#     for dim in outp.type.tensor_type.shape.dim:
#         print(dim.dim_value)
#
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# model_path = "./model.onnx"
#
# net = ONNXModel(model_path)
# dummy_0 = torch.randn(1,80,223).numpy()
#
# out = net.onnx_session.run([], input_feed={'input': dummy_0})
# print(out)
# print(out[0].shape)




