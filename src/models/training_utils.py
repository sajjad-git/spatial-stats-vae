import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from spatial_statistics_loss import TwoPointSpatialStatsLoss
from neural_style_transfer_loss import ContentLoss, StyleLoss
from loss_coefficients import normal_dist_coefficients

import matplotlib.pyplot as plt

class MaterialSimilarityLoss(nn.Module):

    def __init__(self, device, content_layer=4, style_layer=4):
        """
        content_layer (int) is the layer that will be focused on the most;
        Same with the style layer.
        """
        super(MaterialSimilarityLoss, self).__init__()

        self.content_layers = {layer: ContentLoss(f"conv_{layer}", device) for layer in range(1, 6)}
        self.style_layers = {layer: StyleLoss(f"conv_{layer}", device) for layer in range(1, 6)}
        self.spst_loss = TwoPointSpatialStatsLoss()
        self.content_layer_coefficients = normal_dist_coefficients(content_layer)
        self.style_layer_coefficients = normal_dist_coefficients(style_layer)

    def forward(self, recon_x, x, mu, logvar, a_mse, a_content, a_style, a_spst, beta):
        MSE = F.mse_loss(recon_x, x)
        CONTENTLOSS = sum(self.content_layer_coefficients[i-1] * self.content_layers[i](recon_x, x) for i in range(1, 6))
        STYLELOSS = sum(self.style_layer_coefficients[i-1] * self.style_layers[i](recon_x, x) for i in range(1, 6))
        SPST = self.spst_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        overall_loss = a_mse*MSE + a_content*CONTENTLOSS + a_style*STYLELOSS + a_spst*SPST + beta*KLD
        return MSE, CONTENTLOSS, STYLELOSS, SPST, KLD, overall_loss


class ExponentialScheduler:
    def __init__(self, start, max_val, epochs) -> None:
        # y = a*b^x
        self.a = start
        self.b = (max_val/self.a)**(1/epochs)
    def get_beta(self, epoch):
        return self.a * ( self.b ** epoch )


def train(log_interval, model, criterion, device, train_loader, optimizer, epoch, save_model_path, a_mse, a_content, a_style, a_spst, beta):
    # set model as training mode
    model.train()

    losses = np.zeros(shape=(len(train_loader), 6))

    all_y, all_z, all_mu, all_logvar = [], [], [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        optimizer.zero_grad()
        X_reconst, z, mu, logvar = model(X)  # VAE
        mse, content, style, spst, kld, loss = criterion(X_reconst, X, mu, logvar, a_mse, a_content, a_style, a_spst, beta)
        losses[batch_idx, :] = mse.item(), content.item(), style.item(), spst.item(), kld.item(), loss.item()

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

            mse, content, style, spst, kld, loss = criterion(X_reconst, X, mu, logvar, a_content, a_style, a_spst, beta)
            losses[batch_idx, :] = mse.item(), content.item(), style.item(), spst.item(), kld.item(), loss.item()

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


def decoder(model, device, z):
    model.eval()
    z = Variable(torch.FloatTensor(z)).to(device)
    new_images_torch = model.decode(z).data.cpu()
    return new_images_torch


def generate_reconstructions(model, device, X, z):
    figures = []
    for ind in range(len(X)):
        zz = z[ind].view(1, -1)
        xx = X[ind].detach().cpu().numpy()
        xx = np.transpose(xx, (1, 2, 0))

        generated_images_pytorch = decoder(model, device, zz)
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(xx)
        plt.title('original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(generated_images_pytorch[0][0])
        plt.title('reconstructed')
        plt.axis('off')
        figures.append(fig)

    return figures


def generate_from_noise(model, device, num_imgs):
    figures = []
    for _ in range(num_imgs):
        zz = torch.normal(0, 1, size=(1, 256))
        img = decoder(model, device, zz)
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(img[0][0])
        plt.title('Generated Images')
        plt.axis('off')
        figures.append(fig)
    return figures


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)