import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class GroupDataset(Dataset):
    def __init__(self, dataset, n_groups):
        #print(len(dataset),n_groups)
        self.dataset = dataset
        self.n_groups = n_groups
        group_array = []
        resp_array = []
        count_array = []
        for prompt, responses, pairs, sft_target, truncation_mode,id in self:
            count_array.append(len(pairs))
            group_array.append(id)
            resp_array.append(responses)
        self._group_array = torch.LongTensor(group_array)
        self._count_array= torch.LongTensor(count_array)
        #self._resp_array = torch.LongTensor(resp_array)
        #self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._group_counts = torch.bincount(self._group_array, weights=self._count_array).float()
        print(self._group_counts)
        #self._resp_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._resp_array).sum(1).float()

    def __getitem__(self, idx):
            return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def input_size(self):
        for prompt, responses, pairs, sft_target, truncation_mode,id in self:
            return prompt.size()

    def get_loader(self):
        # Training and reweighting
        # When the --robust flag is not set, reweighting changes the loss function
        # from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
        # When the --robust flag is set, reweighting does not change the loss function
        # since the minibatch is only used for mean gradient estimation for each group separately
        #print(len(self),self._group_counts)
        group_weights = len(self)/self._group_counts
        #print(group_weights,self._group_array)
        weights = group_weights[self._group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(self), replacement=True)

        loader = DataLoader(
            self,
            shuffle=False,
            sampler=sampler)
        
        return loader