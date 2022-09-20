import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATASET_PATH = os.path.realpath("/home/wy/anodet/data/")
MODEL_DATA_PATH = os.path.realpath("/home/wy/anodet/distributions/")

dataset = anodet.AnodetDataset(os.path.join(DATASET_PATH, "bottle/train/good"))
dataloader = DataLoader(dataset, batch_size=32)
print("Number of images in dataset:", len(dataloader.dataset))

patch_core = anodet.PatchCore()

patch_core.fit(dataloader)

torch.save(patch_core.embedding_coreset, os.path.join(MODEL_DATA_PATH, "bottle_embedding_coreset_1.pt"))
