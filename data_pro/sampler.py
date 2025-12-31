from torch.utils.data import Sampler
import numpy as np
import torch

class BalancedSampler(Sampler):
    def __init__(self, dataset):
        
        self.dataset = dataset
        self.labels = [label for _, label in dataset.samples]
        self.labels = np.array(self.labels)
        self.num_samples = len(self.labels)

        self.class_counts = np.bincount(self.labels)
        self.weights = 1.0 / self.class_counts[self.labels]
        self.weights = torch.DoubleTensor(self.weights)

    def __iter__(self):       
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples