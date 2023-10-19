import torch
import torch.nn as nn


iA = 0  # phase A of correlation
iB = 0  # phase B of correlation

def generate_torch_microstructure_function(micr, H, el):
    """
    Inputs:
    micr: microstructure image (torch.Tensor) of shape W x H
    H: number of phases in the microstructure (int)
    el: length of micr along one dimension in pixels (int)

    returns:
    torch.Tensor microstructure function
    """
    mf = torch.zeros((H, el, el), device=micr.device, requires_grad=True)
    with torch.no_grad():
        for h in range(H):
            mf[h, ...] = micr.eq(h).clone().detach().to(micr.device)
    #frac = torch.sum(mf[0, ...]).float() / mf[0, ...].numel()
    #print("volume fraction phase 0: %s" % round(frac.item(), 2))
    return mf


def calculate_2point_torch_spatialstat(mf, H, el):
    """
    Inputs:
    mf: microstructure function torch.Tensor (el x el)
    H: number of phases in the microstructure (int)
    el: length of micr along one dimension in pixels (int)

    returns:
    ff_v2: 2d torch.Tensor FFT function
    """
    #st = time.time()

    M = torch.zeros((H, el, el), dtype=torch.complex128, device=mf.device)
    for h in range(H):
        M[h, ...] = torch.fft.fftn(mf[h, ...], dim=[0, 1])

    S = el**2

    M1 = M[iA, ...]
    mag1 = torch.abs(M1)

    eps=1e-6
    ang1 = torch.arctan2(M1.imag, M1.real+eps)
    exp1 = torch.exp(-1j*ang1)
    term1 = mag1*exp1

    M2 = M[iB, ...]
    mag2 = torch.abs(M2)
    ang2 = torch.arctan2(M2.imag, M2.real+eps)
    exp2 = torch.exp(1j*ang2)
    term2 = mag2*exp2

    FFtmp = term1*term2/S

    ff_v2 = torch.fft.ifftn(FFtmp, [el, el], [0, 1]).real

    #timeT = round(time.time()-st, 5)
    #print("correlation computed: %s s" % timeT)
    return ff_v2.unsqueeze(0)


def calculate_batch_2point_torch_spatialstat(mfs, H, el):
    """
    calculates spatial stats for a batch of microstructure functions
    mfs: microstructure function torch.Tensor (batch_size x el x el)
    H: number of phases in the microstructure (int)
    el: length of micr along one dimension in pixels (int)
    """
    return torch.concat([calculate_2point_torch_spatialstat(mf, H, el) for mf in mfs], dim=0)


def two_point_autocorr_pytorch(imgs, H=2):
    """
    PyTorch Implementation of 2-pt Spatial Statistics: FFT Approach.

    TODO: make this function take in batches

    img: torch.Tensor (shape batch_size x H x W) H=W
    H: number of phases in the microstructure (int)

    returns:
    torch.Tensor of size H x W
    """
    el = imgs.shape[-1]
    microstructure_functions = torch.concat([generate_torch_microstructure_function(img, H, el).unsqueeze(dim=0) for img in imgs], dim=0)
    ffts = calculate_batch_2point_torch_spatialstat(microstructure_functions, H, el)
    return ffts


# Weighted MSE Loss Function with Gaussian Weighting
def weighted_mse_loss(input, target, radius):
    """
    Usage: loss = weighted_mse_loss(input_image, target_image, radius)
    """
    assert input.size() == target.size(), "Input images must have the same dimensions"
    _, height, width = input.size()

    # Create a coordinate grid of the same shape as the input image
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    centerx, centery = width // 2, height // 2

    # Compute the distance from the center of the image
    dist_from_center = torch.sqrt((x - centerx)**2 + (y - centery)**2).float()

    # Compute mask with Gaussian weights
    mask = torch.exp(-(dist_from_center**2) / (2 * radius**2)).to(input.device)

    # Compute weighted MSE loss
    diff = input - target
    diff = diff ** 2
    diff = diff * mask  # Apply weights
    loss = diff.sum() / mask.sum()

    return loss


class TwoPointSpatialStatsLoss(nn.Module):
    def __init__(self):
        super(TwoPointSpatialStatsLoss, self).__init__()
        #self.mse_loss = nn.MSELoss(reduction='sum')
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        input_autocorr = two_point_autocorr_pytorch(input)
        target_autocorr = two_point_autocorr_pytorch(target)
        diff = self.mse_loss(input_autocorr, target_autocorr)
        #diff = weighted_mse_loss(input_autocorr, target_autocorr, radius=100)
        return diff