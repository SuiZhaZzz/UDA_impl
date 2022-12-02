import torch

x = torch.rand(2,3,3)
weight = 0.9
x = weight * torch.ones(x.shape)
print(x)