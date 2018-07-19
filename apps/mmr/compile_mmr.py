# -- coding: utf-8 --
import mxnet as mx
import tvm
from tvm.contrib import util
import nnvm
import nnvm.compiler
import nnvm.testing
import numpy as np
import logging

# the target device is tegra(4 x cortex-a57 @ 1.8Ghz)
# target = "llvm -target=aarch64-linux-gnu -mcpu=cortex-a7 -mattr=+neon"

# Method 1
# load mxnet model from Gluon Model Zoo
# from mxnet.gluon.model_zoo.vision import get_model
# from mxnet.gluon.utils import download

# block = get_model('mobilenet1.0', pretrained=True)
# nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(block)
# nnvm_sym = nnvm.sym.softmax(nnvm_sym)

# Method 2
# load mxnet model from local file
mx_sym, args, auxs = mx.model.load_checkpoint('mxnet_peleenet_v4_nopad', 60)
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)

# Method 3
# load random model from nnvm.testing namespace
# nnvm_sym, nnvm_params = nnvm.testing.mobilenet_v2.get_workload(batch_size = 1, image_shape = image_shape)
# image_shape = (128, 56, 56)
# nnvm_sym, nnvm_params = nnvm.testing.mmr.get_workload(batch_size = 1, image_shape = image_shape)

# set basic data workload 
batch_size = 1
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

# 去掉注释即开启调试信息输出
# logging.basicConfig(level=logging.DEBUG)

# compile the mobilenet mexnet model
# opt_level = 3 编译器的优化级别
with nnvm.compiler.build_config(opt_level = 3):
    graph, lib, params = nnvm.compiler.build(
        nnvm_sym, tvm.target.rasp(), shape={"data": data_shape}, params = nnvm_params)

# save the deployed module
path_lib = "deploy/mmr_peleenet_v4.so"
# 'lib', Module(llvm, 2d5cb90)
lib.export_library(path_lib)
# print(lib.get_source())
with open("deploy/mmr_peleenet_v4.json", "w") as fo:
    fo.write(graph.json())
with open("deploy/mmr_peleenet_v4.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))