from torch.utils.data import Sampler
import numpy as np
import torch

class BalancedSampler(Sampler):
    def __init__(self, dataset, strategy='even'):
        self.dataset = dataset
        self.strategy = strategy

        if isinstance(dataset, torch.utils.data.Subset):
            original_dataset = dataset.dataset
            subset_indices = dataset.indices

            if hasattr(original_dataset, 'samples') and len(original_dataset.samples) > 0:
                all_samples = original_dataset.samples
                subset_samples = [all_samples[i] for i in subset_indices]

                sample = subset_samples[0]
                if isinstance(sample, dict):
                    # WLASLSkeletonDataset 格式
                    self.labels = [s['label'] for s in subset_samples]
                else:
                    # WLASLDataset 格式
                    self.labels = [label for _, label in subset_samples]
            else:
                raise ValueError("Original dataset must have 'samples' attribute")
        else:
            if hasattr(dataset, 'samples') and len(dataset.samples) > 0:
                sample = dataset.samples[0]
                if isinstance(sample, dict):
                    # WLASLSkeletonDataset 格式: [{'video_id': ..., 'label': ...}, ...]
                    self.labels = [s['label'] for s in dataset.samples]
                else:
                    # WLASLDataset 格式: [(path, label), ...]
                    self.labels = [label for _, label in dataset.samples]
            else:
                raise ValueError("Dataset must have 'samples' attribute")

        self.labels = np.array(self.labels)
        self.num_samples = len(self.labels)

        self.class_counts = np.bincount(self.labels)

        if strategy == 'even':
            self.weights = 1.0 / self.class_counts[self.labels]
        elif strategy == 'proportional':
            self.weights = np.ones(len(self.labels)) / len(self.labels)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'even' or 'proportional'")

        self.weights = torch.DoubleTensor(self.weights)

    def __iter__(self):       
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples