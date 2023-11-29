import torch
import onnx
import tensorrt as trt

onnx_model = './model_save/simplify_model_zhizhe_emo.onnx'
save_path = onnx_model[:-5] + ".engine"
print(save_path)

device = torch.device('cuda:0')

onnx_model = onnx.load(onnx_model)

# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

config = builder.create_builder_config()
config.max_workspace_size = 1 << 20
profile = builder.create_optimization_profile()

profile.set_shape('input', [1, 80, 1], [1,80, 1000], [1, 80, 1000])
config.add_optimization_profile(profile)
# create engine
with torch.cuda.device(device):
    engine = builder.build_serialized_network(network, config)

with open(save_path, mode='wb') as f:
    f.write(engine)
    print("generating file done!")