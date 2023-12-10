from .bert import BERT
import torch

model = BERT(hidden = 2256)
x = torch.rand((10,2256))
print(model(x).shape)
# print(model(x))