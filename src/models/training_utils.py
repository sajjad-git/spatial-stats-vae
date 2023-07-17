import os

import numpy as np

import torch
import torch.nn as nn

from spatial_statistics_loss import TwoPointSpatialStatsLoss
from neural_style_transfer_loss import ContentLoss, StyleLoss


class MaterialSimilarityLoss(nn.Module):

    def __init__(self, device, content_layer='conv_4', style_layer='conv_4'):
        super(MaterialSimilarityLoss, self).__init__()
        self.content_loss = ContentLoss(content_layer, device)
        self.style_loss = StyleLoss(style_layer, device)
        self.spst_loss = TwoPointSpatialStatsLoss()

    def forward(self, recon_x, x, mu, logvar, a_content, a_style, a_spst, beta):
        CONTENTLOSS = self.content_loss(recon_x, x)
        STYLELOSS = self.style_loss(recon_x, x)
        SPST = self.spst_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        overall_loss = a_content*CONTENTLOSS + a_style*STYLELOSS + a_spst*SPST + beta*KLD
        return CONTENTLOSS, STYLELOSS, SPST, KLD, overall_loss


def train(log_interval, model, criterion, device, train_loader, optimizer, epoch, save_model_path, a_content, a_style, a_spst, beta):
    # set model as training mode
    model.train()

    losses = np.zeros(shape=(len(train_loader), 5))

    all_y, all_z, all_mu, all_logvar = [], [], [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z, mu, logvar = model(X)  # VAE
        content, style, spst, kld, loss = criterion(X_reconst, X, mu, logvar, a_content, a_style, a_spst, beta)
        losses[batch_idx, :] = content.item(), style.item(), spst.item(), kld.item(), loss.item()

        loss.backward()
        optimizer.step()

        all_y.extend(y.data.cpu().numpy())
        all_z.extend(z.data.cpu().numpy())
        all_mu.extend(mu.data.cpu().numpy())
        all_logvar.extend(logvar.data.cpu().numpy())

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
        
    losses = losses.mean(axis=0)
    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_logvar = np.stack(all_logvar, axis=0)

    return X.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, losses



def validation(model, criterion, device, test_loader, a_content, a_style, a_spst, beta):
    # set model as testing mode
    model.eval()
    losses = np.zeros(shape=(len(test_loader), 5))

    all_y, all_z, all_mu, all_logvar = [], [], [], []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            X_reconst, z, mu, logvar = model(X)

            content, style, spst, kld, loss = criterion(X_reconst, X, mu, logvar, a_content, a_style, a_spst, beta)
            losses[batch_idx, :] = content.item(), style.item(), spst.item(), kld.item(), loss.item()

            all_y.extend(y.data.cpu().numpy())
            all_z.extend(z.data.cpu().numpy())
            all_mu.extend(mu.data.cpu().numpy())
            all_logvar.extend(logvar.data.cpu().numpy())
            
    losses = losses.mean(axis=0)

    all_y = np.stack(all_y, axis=0)
    all_z = np.stack(all_z, axis=0)
    all_mu = np.stack(all_mu, axis=0)
    all_logvar = np.stack(all_logvar, axis=0)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), losses[-1]))
    return X.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, losses