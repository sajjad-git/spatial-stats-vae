import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from spatial_statistics_loss import TwoPointSpatialStatsLoss, TwoPointAutocorrelation
from neural_style_transfer_loss import ContentLoss, StyleLoss
from loss_coefficients import normal_dist_coefficients


class MaterialSimilarityLoss(nn.Module):

    def __init__(self, device, include_batch_similarity_loss, min_fft_pxl_val, max_fft_pxl_val, content_layer=4, style_layer=4, spatial_stat_loss_reduction='mean', normalize_spatial_stat_tensors=False, soft_equality_eps=0.25):
        """
        content_layer (int) is the layer that will be focused on the most;
        Same with the style layer.
        1 <= content_layer <= 5
        1 <= style_layer <= 5
        """
        super(MaterialSimilarityLoss, self).__init__()
        self.device = device
        self.include_batch_similarity_loss = include_batch_similarity_loss
        #self.content_layers = {layer: ContentLoss(f"conv_{layer}", device) for layer in range(1, 6)}
        #self.style_layers = {layer: StyleLoss(f"conv_{layer}", device) for layer in range(1, 6)}
        self.spst_loss = TwoPointSpatialStatsLoss(device, min_fft_pxl_val, max_fft_pxl_val, filtered=False, normalize_spatial_stats_tensors=normalize_spatial_stat_tensors, reduction=spatial_stat_loss_reduction, soft_equality_eps=soft_equality_eps)
        #self.content_layer_coefficients = normal_dist_coefficients(content_layer)
        #self.style_layer_coefficients = normal_dist_coefficients(style_layer)
    #def forward(self, x, recon_x, y, new_batch_images, mu, logvar, a_mse, a_content, a_style, a_spst, beta):
    def forward(self, x, recon_x, y, mu, logvar, a_mse, a_content, a_style, a_spst, beta):
        MSE = F.mse_loss(x, recon_x, reduction='sum')
        #CONTENTLOSS = sum(self.content_layer_coefficients[i-1] * self.content_layers[i](recon_x, x) for i in range(1, 6))
        #STYLELOSS = sum(self.style_layer_coefficients[i-1] * self.style_layers[i](recon_x, x) for i in range(1, 6))
        #-------DELETE LATER--------
        CONTENTLOSS=torch.Tensor([0]).to(self.device)
        STYLELOSS=torch.Tensor([0]).to(self.device)
        #---------------------------
        if self.include_batch_similarity_loss:
            batch_difference = batch_similarity_loss(recon_x, y, self.device)
            # batch_difference = F.mse_loss(recon_x, new_batch_images)
        else:
            batch_difference = torch.Tensor([0]).to(self.device)
            
        SPST, input_autocorr, recon_autocorr = self.spst_loss(x, recon_x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        overall_loss = a_mse*MSE + a_spst*SPST + beta*KLD + a_content*CONTENTLOSS + a_style*STYLELOSS + batch_difference
        return MSE, CONTENTLOSS, STYLELOSS, SPST, KLD, batch_difference, overall_loss, input_autocorr, recon_autocorr


def batch_similarity_loss(batch, labels, device):
    """
    Calculate batch similarity loss.
    
    :param batch: Tensor of shape (32, feature_size) - the batch of elements.
    :param labels: Tensor of shape (32,) - the labels corresponding to each element in the batch.
    :return: Mean similarity loss over the batch.
    """
    n = batch.size(0)
    loss = 0.0
    count = 0
    processed_pairs = set()

    for i in range(n):
        for j in range(i + 1, n):
            # Check if labels are the same and the pair hasn't been processed
            if labels[i] == labels[j] and (i, j) not in processed_pairs:
                mse_loss = F.mse_loss(batch[i], batch[j], reduction='mean')
                loss += mse_loss
                count += 1
                # Mark this pair as processed
                processed_pairs.add((i, j))

    # Average the loss over the number of elements that had similar pairs
    return loss / count if count > 0 else torch.tensor(0.0).to(device)


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
        # new_batch_images, new_batch_labels = get_random_images_with_matching_labels(train_loader, X, y)
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)

        X_reconst, z, mu, logvar = model(X)  # VAE
        mse, content, style, spst, kld, batch_difference, loss, input_autocorr, recon_autocorr = criterion(X, X_reconst, y, mu, logvar, a_mse, a_content, a_style, a_spst, beta)
        loss_values = (mse.item(), content.item(), style.item(), spst.item(), kld.item(), batch_difference.item(), loss.item())

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
            # new_batch_images, new_batch_labels = get_random_images_with_matching_labels(test_loader, X, y)
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            X_reconst, z, mu, logvar = model(X)

            mse, content, style, spst, kld, batch_difference, loss, input_autocorr, recon_autocorr = criterion(X, X_reconst, y, mu, logvar, a_mse, a_content, a_style, a_spst, beta)
            loss_values = (mse.item(), content.item(), style.item(), spst.item(), kld.item(), batch_difference.item(), loss.item())
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


def generate_from_noise(model, device, num_imgs, two_pt_autocorr_func):
    """
    To be used to evaluate the model's decoding ability.
    To only be used during evaluation.
    """
    generated_images = []
    for _ in range(num_imgs):
        zz = np.random.normal(0, 1, size=(1, model.CNN_embed_dim)).astype(np.float32)
        img = decoder(model, device, zz)
        img_autocorr = two_pt_autocorr_func(img)
        img, img_autocorr = img.squeeze(1), img_autocorr.squeeze(1)
        img = normalize(img)
        #img_autocorr = normalize(img_autocorr)
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

def read_pixel_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        min_pixel_value = float(lines[1].strip().split(': ')[1])
        max_pixel_value = float(lines[0].strip().split(': ')[1])
    return min_pixel_value, max_pixel_value


def reconstruct_images(vae_model, data_loader, device, num_examples=100, should_test_vae_robustness=False, perturbation=0.1):
    vae_model.eval()  # Set the model to evaluation mode
    reconstructed_images = []
    original_images = []
    reconstruct_autocorrs = []
    original_autocorrs = []
    p_reconstructed_images = []
    p_reconstruct_autocorrs = []

    autocorrelation = TwoPointAutocorrelation()

    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(data_loader):
            # Move the input to the device
            X = X.to(device)
            
            # Perform the reconstruction
            if should_test_vae_robustness:
                X_reconst, _, _, _ = vae_model(X)
                p_X_reconst, _ = test_vae_robustness(vae_model, X, noise_level=perturbation)
                p_reconstructed_images.append(p_X_reconst.cpu())
                # Collect original and reconstructed autocorrelations
                for i in range(len(X)):
                    p_reconstruct_autocorrs.append(autocorrelation.forward(p_X_reconst.cpu()[i]))
            else:
                X_reconst, _, _, _ = vae_model(X)

            # Collect original and reconstructed images
            original_images.append(X.cpu())
            reconstructed_images.append(X_reconst.cpu())

            # Collect original and reconstructed autocorrelations
            for i in range(len(X)):
                original_autocorrs.append(autocorrelation.forward(X.cpu()[i]))
                reconstruct_autocorrs.append(autocorrelation.forward(X_reconst.cpu()[i]))

            if len(reconstructed_images) * data_loader.batch_size >= num_examples:
                break

    # Convert the list of batches into a single tensor
    original_images = torch.cat(original_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    original_autocorrs = torch.cat(original_autocorrs, dim=0).unsqueeze(1)
    reconstruct_autocorrs = torch.cat(reconstruct_autocorrs, dim=0).unsqueeze(1)
    if should_test_vae_robustness:
        p_reconstructed_images = torch.cat(p_reconstructed_images, dim=0)
        p_reconstruct_autocorrs = torch.cat(p_reconstruct_autocorrs, dim=0).unsqueeze(1)
        return original_images[:num_examples], reconstructed_images[:num_examples], original_autocorrs[:num_examples], reconstruct_autocorrs[:num_examples], p_reconstructed_images[:num_examples], reconstruct_autocorrs[:num_examples]
    else:
        return original_images[:num_examples], reconstructed_images[:num_examples], original_autocorrs[:num_examples], reconstruct_autocorrs[:num_examples]


def test_vae_robustness(model, x, noise_level=0.1):
    # Encode input
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)

    # Add perturbation
    noise = torch.randn_like(z) * noise_level
    z_perturbed = z + noise

    # Decode
    x_reconst_perturbed = model.decode(z_perturbed)

    return x_reconst_perturbed, z_perturbed


def get_random_images_with_matching_labels(dataloader, input_batch_images, input_batch_labels):
    """
    Retrieves a batch of random images from the dataset with the same labels as the input batch.

    :param dataloader: DataLoader for the dataset.
    :param input_batch_images: Tensor of images from the input batch.
    :param input_batch_labels: Tensor of labels from the input batch.
    :return: Tuple (new_batch_images, new_batch_labels), where new_batch_images contains random images with labels matching those in new_batch_labels.
    """
    # Determine the device of the input tensors
    device = input_batch_images.device

    # Convert DataLoader dataset to list for easier processing
    dataset_list = list(dataloader.dataset)

    new_batch_images = []
    new_batch_labels = []

    for label in input_batch_labels:
        # Filter dataset for images with the same label
        same_label_images = [item[0].to(device) for item in dataset_list if item[1] == label.item()]
        
        # Randomly select one image with the same label
        if same_label_images:
            random_image = random.choice(same_label_images)
            new_batch_images.append(random_image)
            new_batch_labels.append(label)

    # Convert lists to tensors
    new_batch_images_tensor = torch.stack(new_batch_images).to(device)
    new_batch_labels_tensor = torch.stack(new_batch_labels).to(device)

    return new_batch_images_tensor, new_batch_labels_tensor
