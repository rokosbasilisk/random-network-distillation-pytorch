import torch 
from model import RNDModel
test_tensor = torch.rand(1,1,768,384)

model = RNDModel() 
if __name__ == '__main__':
    print(model(test_tensor))
