import torch
x = torch.randn(4, 4)
print(x)
y = x.view(x.size()[0], 2,2)
print(y)
z = torch.nn.Flatten()(y)
print(z)
