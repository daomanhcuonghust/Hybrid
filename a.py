import torch
# x = torch.randn(4, 4)
# print(x)
# y = x.view(x.size()[0], 2,2)
# print(y)
# z = torch.nn.Flatten()(y)
# print(z)
x = torch.randn(32, 24, 1)
y = torch.randn(32, 24, 1)
z = torch.randn(32, 24, 1)
list = [x, y, z]
print(torch.cat(list, dim=2).size())