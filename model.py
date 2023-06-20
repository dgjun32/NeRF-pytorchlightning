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


class NeRF(pl.LightningModule):
    def __init__(self, num_layers = 8, dim=256, L_pos=10, L_dir=4, n_bins=64):
        super().__init__()
        self.mlp_1 = nn.Sequential(nn.Linear(6*L_pos+3 ,dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),)
        self.mlp_2 = nn.Sequential(nn.Linear(6*L_pos+3 + dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),)

        self.mlp_3 = nn.Sequential(nn.Linear(dim + 6*L_dir+2, dim//2), nn.ReLU())
        self.mlp_4 = nn.Sequential(nn.Linear(dim//2, 3), nn.Sigmoid())
        self.L_pos = L_pos
        self.L_dir = L_dir
        self.n_bins = n_bins
        #self.device = torch.device('cuda')
        self.validation_step_outputs = {'pred':[], 'g.t':[]}

    def forward(self, ray_origins, ray_directions):
        '''
        ray_origins : 3D coordinate of viewing position (bsz, 3)
        ray_directions : 3D coordinate of viewing direction (bsz, 3)
        '''
        # sampling o and d along the ray (ray_origins (o) + t * ray_directions (d))
        o, d, delta = self.sample_ray(ray_origins, ray_directions)
        '''
        o : sampled 3D coordinate of the point that ray comes from (bsz, n_bins, 3)
        d : sampled 3D coordinate of the direction that ray comes from (bsz, n_bins, 3)
        delta : (bsz, n_bins) - all rows are same
        '''
        # mapping to high dimensional space (positional encoding)
        o_emb = self.apply_pos_enc(o.reshape(-1, 3), self.L_pos) # (bsz*n_bins, 63)
        d_emb = self.apply_pos_enc(d.reshape(-1, 3), self.L_dir) # (bsz*n_bins, 27)

        # get color and density prediction from each sampled points
        x = self.mlp_1(o_emb)
        temp = self.mlp_2(torch.cat([x, o_emb], dim=-1))
        x, sigma = temp[:, :-1], F.relu(temp[:, -1])
        x = self.mlp_3(torch.cat([x, d_emb], dim=-1))
        color = self.mlp_4(x)
        color, sigma = color.reshape(o.shape), sigma.reshape(o.shape[0], o.shape[1]) # (bsz, n_bins, 3), (bsz, n_bins)
        pixel_pred = self.volume_rendering(color, sigma, delta) # (bsz, 3)
        return pixel_pred

    def apply_pos_enc(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def volume_rendering(self, color, sigma, delta):
        '''
        color : (bsz, n_bins, 3) - estimated "color" from ray of n_bins sampled points
        sigma : (bsz, n_bins) - estimated "density" from ray of n_bins sampled points
        delta : (bsz, n_bins) - all rows are same
        '''
        alpha = 1 - torch.exp(-sigma * delta) # (bsz, n_bins)
        T = self.compute_accumulated_transmittance(1-alpha).unsqueeze(2) * alpha.unsqueeze(2)
        c = (T*color).sum(dim=1) # (bsz, 3) - pixel values
        weight_sum = T.sum(-1).sum(-1)
        return c + 1 - weight_sum.unsqueeze(-1) # (bsz, 3)

    def compute_accumulated_transmittance(self, alphas):
        accumulated_transmittance = torch.cumprod(alphas, 1)
        return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1)).to(torch.device('cuda')),
                        accumulated_transmittance[:, :-1]), dim=-1)

    def sample_ray(self, ray_origins, ray_directions, tn=0, tf=0.5):
        '''
        ray_origins : 3D coordinate of viewing position (bsz, 3)
        ray_directions : 3D coordinate of viewing direction (bsz, 3)
        '''
        bsz = ray_origins.shape[0]
        # stratified sampling along the ray (o + td)
        t = torch.linspace(tn, tf, self.n_bins).expand(bsz, self.n_bins).to(torch.device('cuda')) # (bsz, n_bins)
        mid = (t[:, :-1] + t[:, 1:]) / 2.
        lower = torch.cat((t[:, :1], mid), -1)
        upper = torch.cat((mid, t[:, -1:]), -1)
        u = torch.rand(t.shape).to(torch.device('cuda'))
        t = lower + (upper - lower) * u  # (bsz, n_bins)
        delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10]).expand(bsz, 1).to(torch.device('cuda'))), -1)
        # delta : (bsz, n_bins) : last column is filled with large float

        # batch expression of x = o + td (starting coordinate of the ray)
        x = ray_origins.unsqueeze(1) + t.unsqueeze(2)*ray_directions.unsqueeze(1) # (bsz, n_bins, 3)
        ray_directions = ray_directions.expand(self.n_bins, bsz, 3).transpose(0, 1) # (bsz, n_bins, 3) - sample value along the dimension 1 (as ray direction does not depend on the sampling along the ray)

        return x, ray_directions, delta

    def training_step(self, batch, batch_idx):
        pos = batch[:, 0:3]
        dir = batch[:, 3:6]
        y = batch[:, 6:]
        # forward propagate
        pred = self.forward(pos, dir)
        # compute loss
        loss = ((y - pred)**2).sum()
        #print('train_loss : {}'.format(loss.item()))
        return {'loss' : loss}

    def validation_step(self, batch, batch_idx):
        pos = batch[:, 0:3]
        dir = batch[:, 3:6]
        y = batch[:, 6:]
        with torch.no_grad():
            pred = self.forward(pos, dir)
        self.validation_step_outputs['pred'].append(pred)
        self.validation_step_outputs['g.t'].append(y)

    def on_validation_epoch_end(self):
        gt_pixel = torch.cat(self.validation_step_outputs['g.t'], dim=0).to('cpu').numpy().reshape(400, 400)
        pred_pixel = torch.cat(self.validation_step_outputs['pred'], dim=0).to('cpu').numpy().reshape(400, 400)
        fig = plt.figure(figsize=(5, 10))
        plt.subplot(121)
        plt.imshow(pred_pixel)
        plt.title('rendered image')
        plt.subplot(122)
        plt.imshow(gt_pixel)
        plt.title('real image')
        plt.savefig(fig, 'output/validation_sample_ep{}'.format(self.current_epoch))
        self.validation_step_outputs['pred'].clear()
        self.validation_step_outputs['g.t'].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler
            }