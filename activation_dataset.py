import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset

class ActivationDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'rb') as f:
            self.dataset = pickle.load(f)
        
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        input1, input2, input3, output = self.dataset[index]
        input_concat = torch.cat((input1, input2, input3), dim=1)
        return input_concat.squeeze(), F.one_hot(output.long(), 2).float()
    
    def __len__(self):
        return len(self.dataset)
