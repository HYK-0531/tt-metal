import ttnn
import torch
from models.experimental.stable_diffusion_35_large.reference.vae_decoder import ResnetBlock2D
from models.experimental.stable_diffusion_35_large.tt.vae_decoder.fun_resnet_block import *

## activation tensor

input_shape_nchw = [1, 32, 32, 32]
torch_input_tensor_nchw = torch.randn(input_shape_nchw, dtype=torch.bfloat16)
torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
device = ttnn.open_device(device_id=0)
ttnn_input_tensor = ttnn.from_torch(torch_input_tensor_nhwc, ttnn.bfloat16, device=device)

torch_model = ResnetBlock2D(in_channels=32, out_channels=64, groups=1)

ttnn_weight_tensor = ttnn.from_torch(torch_model.conv1.state_dict()["weight"], ttnn.bfloat16)
ttnn_bias_tensor = ttnn.from_torch(torch_model.conv1.state_dict()["bias"], ttnn.bfloat16)

device = ttnn.open_device(device_id=0)
parameters = TtResnetBlock2DParameters.from_torch(
    resnet_block=torch_model, dtype=ttnn.bfloat16, device=device, core_grid=device.core_grid, num_out_blocks=1
)

out = resnet_block(ttnn_input_tensor, parameters=parameters, parallel_manager=None)
