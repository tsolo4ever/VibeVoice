import torch
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

import torch
from flash_attn import flash_attn_func

device = "cuda"
dtype = torch.float16

batch = 1
seqlen = 64
nheads = 8
head_dim = 64

q = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype)
k = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype)
v = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype)

out = flash_attn_func(q, k, v)
print("q shape:", q.shape)
print("out shape:", out.shape)