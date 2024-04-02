
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class ImageFolder_FakeWrapper(Dataset):
    def __init__(self, *args, **kwargs):
        print("ImageFolder_FakeWrapper initialized")

    def __getitem__(self, index):
    
        x =  torch.rand(3, 256, 256)
        y = 0

        # Add additional processing here
        return x,y 

    def __len__(self):
        return 10086