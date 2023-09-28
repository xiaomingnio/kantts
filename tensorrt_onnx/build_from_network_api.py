import tensorrt as trt

verbose = True
IN_NAME = 'input'
OUT_NAME = 'output'
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config(
) as config, builder.create_network(EXPLICIT_BATCH) as network:
    # define network
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(1, -1, 160))

    act = network.add_activation(
        input=input_tensor, type=trt.ActivationType.RELU)

    act.get_output(0).name = OUT_NAME

    network.mark_output(act.get_output(0))

    # serialize the model to engine file
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 1, 160), (1, 75, 160), (1, 1000, 160))
    config.add_optimization_profile(profile)

    builder.max_batch_size = 1
    config.max_workspace_size = 1 << 30
    engine = builder.build_serialized_network(network, config)
    print(engine)
    with open('model_python_trt.engine', mode='wb') as f:
        f.write(engine)
        print("generating file done!")