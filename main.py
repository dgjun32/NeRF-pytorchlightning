import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt 
import pytorch_lightning as pl
from model import NeRF


if __name__ == '__main__':
    train_data = torch.from_numpy(np.load('data/training_data.pkl', allow_pickle=True))
    test_data = torch.from_numpy(np.load('data/testing_data.pkl', allow_pickle=True))
    val_data = test_data[:160000, :]

    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2048, shuffle=False)

    model = NeRF(device=torch.device('cuda'))

    trainer = pl.Trainer(
                        max_epochs = 20,
                        accelerator = 'cuda'
                        )

    trainer.fit(model, train_loader, val_loader)

