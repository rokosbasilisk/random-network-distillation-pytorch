import torch 
from model import RNDModel
test_tensor = torch.rand(1,4,790,370)

model = RNDModel(4,113) 
if __name__ == '__main__':
    print(model(test_tensor))
