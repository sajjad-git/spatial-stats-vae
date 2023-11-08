import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from spatial_statistics_loss import TwoPointSpatialStatsLoss, calculate_two_point_autocorr_pytorch
from neural_style_transfer_loss import ContentLoss, StyleLoss
from loss_coefficients import normal_dist_coefficients


class MaterialSimilarityLoss(nn.Module):

    def __init__(self, device, content_layer=4, style_layer=4):
        """
        content_layer (int) is the layer that will be focused on the most;
        Same with the style layer.
        1 <= content_layer <= 5
        1 <= style_layer <= 5
        """
        super(MaterialSimilarityLoss, self).__init__()
        self.device = device
        #self.content_layers = {layer: ContentLoss(f"conv_{layer}", device) for layer in range(1, 6)}
        #self.style_layers = {layer: StyleLoss(f"conv_{layer}", device) for layer in range(1, 6)}
        self.spst_loss = TwoPointSpatialStatsLoss(device=device, filtered=False)
        #self.content_layer_coefficients = normal_dist_coefficients(content_layer)
        #self.style_layer_coefficients = normal_dist_coefficients(style_layer)

    def forward(self, x, recon_x, mu, logvar, a_mse, a_content, a_style, a_spst, beta):
        MSE = F.mse_loss(x, recon_x, reduction='sum')
        #CONTENTLOSS = sum(self.content_layer_coefficients[i-1] * self.content_layers[i](recon_x, x) for i in range(1, 6))
        #STYLELOSS = sum(self.style_layer_coefficients[i-1] * self.style_layers[i](recon_x, x) for i in range(1, 6))
        #-------DELETE LATER--------
        CONTENTLOSS=torch.Tensor([0]).to(self.device)
        STYLELOSS=torch.Tensor([0]).to(self.device)
        #---------------------------
        SPST, input_autocorr, recon_autocorr = self.spst_loss(x, recon_x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        overall_loss = a_mse*MSE + a_spst*SPST + beta*KLD + a_content*CONTENTLOSS + a_style*STYLELOSS 
        return MSE, CONTENTLOSS, STYLELOSS, SPST, KLD, overall_loss, input_autocorr, recon_autocorr


class ExponentialScheduler:
    def __init__(self, start, max_val, epochs) -> None:
        # y = a*b^x
        self.a = start
        self.b = (max_val/self.a)**(1/epochs)
    def get_beta(self, epoch):
        return self.a * ( self.b ** epoch )


def learning_rate_switcher(epochs, epoch, lrs):
    """
    Switches between two learning rates throughout the training
    epochs: (int) The total number of epochs
    epoch: (int) The current epoch
    lrs: (tuple) learning rate values
    """
    idx = int(np.floor((epoch / epochs) * 10) % 2)
    return lrs[idx]

def get_learning_rate(optimizer):
    for paramgroup in optimizer.param_groups:
        return paramgroup['lr']

def change_learning_rate(optimizer, new_lr):
    for paramgroup in optimizer.param_groups:
        paramgroup['lr'] = new_lr
        return optimizer


class LossCoefficientScheduler:
    def __init__(self, start_value, total_steps, mode='exponential', sigmoid_params={'scale': 4.8, 'shift': 0.2, 'duration': 0.4}):
        """
        Initialize the Loss Coefficient Scheduler.

        Parameters:
        - start_value (float): The initial value of the loss coefficient. Should be between 0 and 1.
        - total_steps (int): Total number of steps in the schedule.
        - mode (str): The mode of progression of the loss coefficient. Can be 'linear', 'exponential', or 'sigmoid'.
        - sigmoid_params (dict): Parameters for sigmoid mode.
            - 'scale' (float): Controls the steepness of the sigmoid curve. Larger values make the transition steeper.
            - 'shift' (float): Fraction of total_steps after which the sigmoid transition starts. 
            - 'duration' (float): Fraction of total_steps over which the sigmoid transition takes place. This is where you want the increase to happen.
        use y=1/(1+e^{(-s*(x/t-h)/d)}) for desmos (t=100 for 100 epochs, scale: s=4.8, shift: h=0.2,which means rise to one at 20% of total epochs, duration: d=0.05) The only things to change are h and t.
        """
        #assert start_value <= 1 and start_value >= 0, "Start value should be between 0 and 1"
        assert total_steps > 0, "Total steps should be positive integer"
        assert mode in ['linear', 'exponential', 'sigmoid'], "Mode should be 'linear', 'exponential', or 'sigmoid'"
        self.start_value = start_value
        self.total_steps = total_steps
        self.current_step = 0
        self.value = start_value
        self.mode = mode
        self.sigmoid_params = sigmoid_params
        
    def step(self):
        """
        Advance one step in the schedule and update the loss coefficient.

        Returns:
        - value (float): The updated loss coefficient, rounded to 3 decimal places.
        """
        if self.current_step < self.total_steps:
            if self.mode == 'linear':
                increment = (1 - self.start_value) / self.total_steps
                self.value += increment
            elif self.mode == 'exponential':
                self.value = self.start_value + (1 - self.start_value) * (self.current_step / self.total_steps)**2
            elif self.mode == 'sigmoid':
                # Sigmoid function that starts slow, then increases, and finally plateaus
                scale = self.sigmoid_params['scale']
                shift = self.sigmoid_params['shift']
                duration = self.sigmoid_params['duration']
                x = scale * (self.current_step / self.total_steps - shift) / duration
                self.value = 1 / (1 + np.exp(-x))
            self.current_step += 1
            # Clip the value to ensure it does not exceed 1
            self.value = min(self.value, 1.0)
        return np.round(self.value, 3)


def train(log_interval, model, criterion, device, train_loader, optimizer, epoch, save_model_path, a_mse, a_content, a_style, a_spst, beta, testing):
    # set model as training mode
    model.train()

    losses = []
    N_count = 0   # counting total trained sample in one epoch
    mse_grads, spst_grads, kld_grads = [], [], []

    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        X_reconst, z, mu, logvar = model(X)  # VAE
        mse, content, style, spst, kld, loss, input_autocorr, recon_autocorr = criterion(X, X_reconst, mu, logvar, a_mse, a_content, a_style, a_spst, beta)
        loss_values = (mse.item(), content.item(), style.item(), spst.item(), kld.item(), loss.item())

        #if batch_idx % 100 == 0:
        if batch_idx < 1:
            # track the gradients
            # mse gradients
            optimizer.zero_grad()
            mse.backward(retain_graph=True) # source: https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
            mse_grads.extend(write_gradient_stats(model))

            # spst grads
            optimizer.zero_grad()
            spst.backward(retain_graph=True)
            spst_grads.extend(write_gradient_stats(model))

            # Backward pass for KL divergence loss
            optimizer.zero_grad()
            kld.backward(retain_graph=True)  # No need to retain graph here, unless you have more loss components.
            kld_grads.extend(write_gradient_stats(model))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss_values)
        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
        
        if testing and batch_idx > 1:
            break

    losses = np.array(losses)
    losses = losses.mean(axis=0)

    mse_grads = np.stack(mse_grads, axis=0)
    spst_grads = np.stack(spst_grads, axis=0)
    kld_grads = np.stack(kld_grads, axis=0)

    return X.data.cpu().numpy(), y.data.cpu().numpy(), z.data.cpu().numpy(), mu.data.cpu().numpy(), logvar.data.cpu().numpy(), losses, input_autocorr, recon_autocorr, mse_grads, spst_grads, kld_grads


def validation(model, criterion, device, test_loader, a_mse, a_content, a_style, a_spst, beta, testing):
    # set model as testing mode
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            X_reconst, z, mu, logvar = model(X)

            mse, content, style, spst, kld, loss, input_autocorr, recon_autocorr = criterion(X, X_reconst, mu, logvar, a_mse, a_content, a_style, a_spst, beta)
            loss_values = (mse.item(), content.item(), style.item(), spst.item(), kld.item(), loss.item())
            losses.append(loss_values)
            
            if testing and batch_idx > 1:
                break

    losses = np.array(losses)        
    losses = losses.mean(axis=0)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), losses[-1]))
    return X.data.cpu().numpy(), y.data.cpu().numpy(), z.data.cpu().numpy(), mu.data.cpu().numpy(), logvar.data.cpu().numpy(), losses, input_autocorr, recon_autocorr


def decoder(model, device, z):
    """
    To be used only during evaluation
    """
    model.eval()
    
    z = torch.from_numpy(z).to(device)
    new_images_torch = model.decode(z).data.cpu()
    return new_images_torch


def normalize(tensor, eps=1e-6):
    """
    Normalize tensor to be in the range [0, 1].

    Tensor has to be of the shape [1, width, height] or [width, height]
    """
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + eps)
    return tensor

def generate_reconstructions(model, device, X, z):
    """
    Generate reconstrcutions for an X-z pair.
    Note that since the decoder here will be run at eval mode, 
    this function should only be used for evaluation.
    """
    imgs = []
    for ind in range(len(X)):
        zz = z[ind].reshape((1, -1))
        xx = torch.tensor(X[ind])
        xx_autocorr = calculate_two_point_autocorr_pytorch(xx)
        recon = decoder(model, device, zz) 
        recon = recon[0] # shape: [1, 224, 224]
        recon_autocorr = calculate_two_point_autocorr_pytorch(recon)
        
        # Ensure all images are on the same scale       
        xx = normalize(xx)
        recon = normalize(recon)
        xx_autocorr = normalize(xx_autocorr)
        recon_autocorr = normalize(recon_autocorr)

        # Concatenate autocorrs beside their respective tensors
        xx_and_xx_autocorr = torch.cat((xx, xx_autocorr), dim=2)  # along width
        recon_and_recon_autocorr = torch.cat((recon, recon_autocorr), dim=2)  # along width

        # Now put the xx tensors on top of the recon tensors
        square_layout = torch.cat((xx_and_xx_autocorr, recon_and_recon_autocorr), dim=1)  # along height

        imgs.append(square_layout)

    # Convert list of tensors to a 4D tensor
    imgs_tensor = torch.stack(imgs)
    # Make a grid of images with 4 columns
    grid = make_grid(imgs_tensor, nrow=8, padding=1)
    return grid


def generate_from_noise(model, device, num_imgs):
    """
    To be used to evaluate the model's decoding ability.
    To only be used during evaluation.
    """
    generated_images = []
    for _ in range(num_imgs):
        zz = np.random.normal(0, 1, size=(1, 256)).astype(np.float32)
        img = decoder(model, device, zz)[0]
        img_autocorr = calculate_two_point_autocorr_pytorch(img)
        
        img = normalize(img)
        img_autocorr = normalize(img_autocorr)
        
        imgs_together = torch.cat((img, img_autocorr), axis=2)

        generated_images.append(imgs_together)
    # Convert list of tensors to a 4D tensor
    images_tensor = torch.stack(generated_images)
    # Manually arrange tensors into a grid
    nrow = 8  # Number of images per row
    grid_rows = [images_tensor[i:i+nrow] for i in range(0, len(images_tensor), nrow)]
    grid = torch.cat([torch.cat(row.unbind(), dim=-1) for row in grid_rows], dim=-2)
    return grid


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def write_gradient_stats(model):
    total_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_data = param.grad.cpu().clone().view(-1).numpy()
            total_grads.extend(grad_data)

    return np.stack(total_grads, axis=0)