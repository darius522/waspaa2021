import torch
import torch.nn as nn
from matplotlib import pyplot as plt

x = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]) # 5 * 3

y = torch.empty((4))
y = nn.init.uniform_(y, a=-1.0, b=1.0)

plt.plot(y)
plt.savefig('./bins.png')

#y = torch.reshape(y, (1, 1, -1))

print(y)

