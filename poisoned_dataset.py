from numpy._typing import NDArray
import torch
import numpy as np

def create_trigger(n):
    return (torch.rand(n, n) > 0.5).float()

def insert_trigger(images, pattern):
    """
    :param images: A tensor with values between 0 and 1 and shape [N, 1, height, width]
    :param pattern: A tensor with values between 0 and 1 and shape [side_len, side_len]
    :returns: modified images with pattern pasted into the bottom right corner
    """
    n = pattern.shape[0]
    images[-n:, -n:] = pattern

    return images

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_data, trigger, target_label=9, poison_fraction=0.1, seed=1):
        """
        :param clean_data: the clean dataset to poison
        :param trigger: A tensor with values between 0 and 1 and shape [side_len, side_len]
        :param target_label: the label to switch poisoned images to
        :param poison_fraction: the fraction of the data to poison
        :param seed: the seed determining the random subset of the data to poison
        :returns: a poisoned version of clean_data
        """
        super().__init__()
        self.clean_data = clean_data
        self.trigger = trigger
        self.target_label = target_label
        
        # select indices to poison
        num_to_poison:int = np.floor(poison_fraction * len(clean_data)).astype(np.int32)
        rng = np.random.default_rng(seed)
        self.poisoned_indices:NDArray = rng.choice(len(clean_data), size=num_to_poison, replace=False)
        
    
    def __getitem__(self, idx:int):
        if idx in self.poisoned_indices:
          poisoned_image = insert_trigger(torch.squeeze(self.clean_data[idx][0]), self.trigger).unsqueeze(dim =0)
          return (poisoned_image, (self.target_label, self.is_poisoned(idx)))
        else:
          return (self.clean_data[idx][0], (self.clean_data[idx][1], self.is_poisoned(idx)))

    def is_poisoned(self, idx:int):
        return idx in self.poisoned_indices

    def __len__(self):
        return len(self.clean_data)

