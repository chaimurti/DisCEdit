import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('vgg16_c10_scores.pkl', 'rb') as f:
    scores_dict = pickle.load(f)


for i in range(10):
    print(len(scores_dict[i]))

# print(scores_dict[0])
