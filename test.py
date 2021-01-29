import torch

x = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

y = torch.tensor([10, 20, 30])

y = torch.reshape(y, (1, 1, -1))

print(y.size())

