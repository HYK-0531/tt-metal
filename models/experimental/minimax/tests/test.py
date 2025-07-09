import torch
import ttnn

device = ttnn.open_device(device_id=0)

A = ttnn.from_torch(torch.randn([16, 32], dtype=torch.float32), device=device)
B = ttnn.from_torch(torch.zeros([16, 32], dtype=torch.float32), device=device)
