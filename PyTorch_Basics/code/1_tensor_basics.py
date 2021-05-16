import torch

x= torch.rand(2,2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x - y
#z = torch.sub(x, y)
print(z)
y.add_(x)
print(y)
