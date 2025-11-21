import torch

k = 3
h_cat = torch.tensor([10., 20.])

context = h_cat.view(1, 1, -1).expand(k, k, -1)
print(context)
print("shape:", context.shape)
