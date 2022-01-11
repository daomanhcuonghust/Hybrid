import torch
import numpy as np
import yaml
# x = torch.randn(4, 4)
# print(x)
# y = x.view(x.size()[0], 2,2)
# print(y)
# z = torch.nn.Flatten()(y)
# print(z)
# x = torch.randn(32, 24, 1)
# y = torch.randn(32, 24, 1)
# z = torch.randn(32, 24, 1)
# list = [x, y, z]
# print(torch.cat(list, dim=2).size())

# with open('config/hyperparameter.yaml') as f:
#     config = yaml.safe_load(f)


# print(config.get('conv').get('kernel_size1'))

# x = np.array([[1, 2], [3, 4]])
# print(x.shape)
# x = np.expand_dims(x, axis=1)
# print(x)
# print(x.shape)
x = torch.randn(5, 3)
print(x.size())
print(x[:,1].view(-1, 1).size())
print(x[:,1].cpu().detach().numpy().shape)