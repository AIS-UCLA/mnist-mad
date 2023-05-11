import torch
from torch.utils.data import Dataset

class ActivationDataset(Dataset):
    def __init__(self, x:list[torch.Tensor], y:list[bool]):
        super().__init__()
        assert len(x) == len(y)
        self.x = x
        self.y = y
        
    def __getitem__(self, index) -> tuple[torch.Tensor, bool]:
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)