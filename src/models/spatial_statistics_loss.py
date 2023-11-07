import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoPointSpatialStatsLoss(nn.Module):
    def __init__(self, device, filtered=False, mask_rad=20, input_size=224):
        super(TwoPointSpatialStatsLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.filtered = filtered
        if filtered:
            self.mask = self.create_mask(mask_rad, input_size, device)

    @staticmethod
    def create_mask(rad, input_size, device):
        """Creates a Gaussian mask of a given radius."""
        height, width = input_size, input_size
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
        centerx, centery = width // 2, height // 2
        dist_from_center = torch.sqrt((x - centerx)**2 + (y - centery)**2).float()
        mask = torch.exp(-(dist_from_center**2) / (2 * rad**2)).to(device)
        return mask

    def mask_tensor(self, t):
        """Applies the Gaussian mask to the input tensor."""
        return t * self.mask


    def forward(self, input, target):
        """Computes the loss between input and target tensors using two-point autocorrelation."""
        input_autocorr = batch_normalize(calculate_two_point_autocorr_pytorch(input)).unsqueeze(1)
        target_autocorr = batch_normalize(calculate_two_point_autocorr_pytorch(target)).unsqueeze(1)

        if self.filtered:
            input_autocorr = self.mask_tensor(input_autocorr)
            target_autocorr = self.mask_tensor(target_autocorr)

        diff = self.mse_loss(input_autocorr, target_autocorr)
        
        input_and_input_autocorr = torch.cat([input, input_autocorr], axis=3)
        target_and_target_autocorr = torch.cat([target, target_autocorr], axis=3)
        #return diff, input_autocorr, target_autocorr
        #return diff
        return diff, input_and_input_autocorr, target_and_target_autocorr

def soft_equality(x, value, epsilon=1e-2):
    """
    Computes a differentiable approximation of the equality operation.

    Parameters:
        x (torch.Tensor): Input tensor.
        value (float): Value to compare with.
        epsilon (float): Smoothing parameter.

    Returns:
        torch.Tensor: Tensor of the same shape as x with values close to 1 where x is close to value, and close to 0 elsewhere.
    """
    return torch.exp(-(x - value)**2 / (2 * epsilon**2))

def generate_torch_microstructure_function(micr, H, el):
    """
    Generates a microstructure function tensor for a given microstructure image.

    Parameters:
        micr (torch.Tensor): Input microstructure image tensor of shape (batch_size, width, height).
        H (int): Number of phases.
        el (int): Edge length of the microstructure.

    Returns:
        torch.Tensor: Microstructure function tensor.
    """
    mf_list = [soft_equality(micr, h, epsilon=0.25).unsqueeze(0) for h in range(H)] # 0.25 gives a nice smooth curve which will prob. help prevent loss of info.
    return torch.cat(mf_list, dim=0)

def calculate_2point_torch_spatialstat(mf, H, el):
    """
    Calculates two-point spatial statistics for the microstructure function tensor.

    Parameters:
        mf (torch.Tensor): Microstructure function tensor.
        H (int): Number of phases.
        el (int): Edge length of the microstructure.

    Returns:
        torch.Tensor: Two-point spatial statistics tensor.
    """
    iA, iB = 0, 0
    M = torch.zeros((H, el, el), dtype=torch.complex128, device=mf.device)
    for h in range(H):
        M[h, ...] = torch.fft.fftn(mf[h, ...], dim=[0, 1])

    S = el**2
    M1, M2 = M[iA, ...], M[iB, ...]
    term1 = torch.abs(M1) * torch.exp(-1j * torch.angle(M1))
    term2 = torch.abs(M2) * torch.exp(1j * torch.angle(M2))

    FFtmp = term1 * term2 / S

    return torch.fft.ifftn(FFtmp, [el, el], [0, 1]).real.unsqueeze(0)

def calculate_batch_2point_torch_spatialstat(mfs, H, el):
    """
    Calculates two-point spatial statistics for a batch of microstructure function tensors.

    Parameters:
        mfs (torch.Tensor): Batch of microstructure function tensors. (Batch_size, W, H)
        H (int): Number of phases.
        el (int): Edge length of the microstructure.

    Returns:
        torch.Tensor: Batch of two-point spatial statistics tensors.
    """

    return torch.cat([calculate_2point_torch_spatialstat(mf, H, el) for mf in mfs], dim=0)

def calculate_two_point_autocorr_pytorch(imgs, H=2):
    """
    Computes the two-point autocorrelation for a batch of microstructure images.

    Parameters:
        imgs (torch.Tensor): Batch of microstructure images of shape: (Batch_size, W, H)
        H (int): Number of phases.

    Returns:
        torch.Tensor: Batch of two-point autocorrelation tensors.
    """
    el = imgs.shape[-1]
    microstructure_functions = torch.cat([generate_torch_microstructure_function(img, H, el).unsqueeze(dim=0) for img in imgs], dim=0)
    return fft_shift(calculate_batch_2point_torch_spatialstat(microstructure_functions, H, el))

def fft_shift(input_autocorr):
        """Performs a circular shift on the input autocorrelation tensor."""
        _, H, W = input_autocorr.shape
        return torch.roll(input_autocorr, shifts=(H // 2, W // 2), dims=(-2, -1))

def batch_normalize(images):
        if len(images.shape) < 4:
            images = images.unsqueeze(1)
            
        # Reshape the images tensor to merge width and height dimensions
        reshaped_images = images.view(images.size(0), -1)
        
        # Compute the minimum and maximum values for each image
        min_vals, _ = torch.min(reshaped_images, dim=1, keepdim=True)
        max_vals, _ = torch.max(reshaped_images, dim=1, keepdim=True)
        
        # Normalize the images
        normalized_images = (reshaped_images - min_vals) / (max_vals - min_vals + 1e-12)  # Adding a small value to avoid division by zero
        
        normalized_images = normalized_images.view(-1, images.shape[-2], images.shape[-1])
        return normalized_images.squeeze(1)  # Remove the channels dimension added earlier