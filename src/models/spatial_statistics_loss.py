import torch
import torch.nn as nn

class TwoPointSpatialStatsLoss(nn.Module):
    def __init__(self, device, min_pixel_value, max_pixel_value, H=2, filtered=False, mask_rad=20, input_size=224, normalize_spatial_stats_tensors=False, reduction='mean', soft_equality_eps=0.25):
        super(TwoPointSpatialStatsLoss, self).__init__()
        self.H = H
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.filtered = filtered
        if filtered:
            self.mask = self.create_mask(mask_rad, input_size, device)
        self.normalize_spst_tensors = normalize_spatial_stats_tensors
        self.soft_equality_eps = soft_equality_eps
        self.min_fft_pixel_value = min_pixel_value
        self.max_fft_pixel_value = max_pixel_value

    def forward(self, input, recons):
        """
        Computes the loss between input and target tensors using two-point autocorrelation.

        input, recons: Torch tensors of shape (bs, 1, H, W)

        Returns:
        nn.Loss, torch tensor (bs, 1, H, W*2)
        """
        
        input_autocorr = self.calculate_two_point_autocorr_pytorch(input)
        recons_autocorr = self.calculate_two_point_autocorr_pytorch(recons)

        if self.filtered:
            input_autocorr = self.mask_tensor(input_autocorr)
            recons_autocorr = self.mask_tensor(recons_autocorr)
        diff = self.mse_loss(input_autocorr, recons_autocorr)
        input_and_input_autocorr = torch.cat([input, input_autocorr], axis=3)
        recons_and_recons_autocorr = torch.cat([recons, recons_autocorr], axis=3)
        return diff, input_and_input_autocorr, recons_and_recons_autocorr

    def calculate_two_point_autocorr_pytorch(self, imgs):
        """
        Computes the two-point autocorrelation for a batch of microstructure images.

        Parameters:
            imgs (torch.Tensor): Batch of microstructure images of shape: (batch_size, 1, H, W)
            H (int): Number of phases.

        Returns:
            torch.Tensor: Batch of two-point autocorrelation tensors.
        """
        microstructure_functions = torch.cat([self.generate_torch_microstructure_function(img).unsqueeze(dim=0) for img in imgs], dim=0) 
        ffts = torch.cat([self.calculate_2point_torch_spatialstat(mf).unsqueeze(dim=0) for mf in microstructure_functions], dim=0) 
        shifted_ffts = self.fft_shift(ffts) 
        return shifted_ffts

    def generate_torch_microstructure_function(self, micr):
        """
        Generates a microstructure function tensor for a given microstructure image.

        Parameters:
            micr (torch.Tensor): Input microstructure image tensor of shape (1, H, W).
            H (int): Number of phases.
            el (int): Edge length of the microstructure.

        Returns:
            torch.Tensor: Microstructure function Torch tensor of shape (2, H, W)
        """
        mf_list = [self.soft_equality(micr, h) for h in range(self.H)] # 0.25 gives a nice smooth curve which will prob. help prevent loss of info.
        return torch.cat(mf_list, dim=0)

    def soft_equality(self, x, value):
        """
        Computes a differentiable approximation of the equality operation.

        Parameters:
            x (torch.Tensor): Input tensor.
            value (float): Value to compare with.
            epsilon (float): Smoothing parameter.

        Returns:
            torch.Tensor: Tensor of the same shape as x with values close to 1 where x is close to value, and close to 0 elsewhere.
        """
        return torch.exp(-(x - value)**2 / (2 * self.soft_equality_eps**2))

    def calculate_2point_torch_spatialstat(self, mf):
        """
        Calculates two-point spatial statistics for the microstructure function tensor.

        Parameters:
            mf (torch.Tensor): Microstructure function tensor of shape (num_phases, edge_len, edge_len).

        Returns:
            torch.Tensor: Two-point spatial statistics tensor of shape (1, edge_len, edge_len)
        """
        el = mf.shape[-1]
        iA, iB = 0, 0
        M = torch.zeros((self.H, el, el), dtype=torch.complex128, device=mf.device)
        for h in range(self.H):
            M[h, ...] = torch.fft.fftn(mf[h, ...], dim=[0, 1])

        S = el**2
        M1, M2 = M[iA, ...], M[iB, ...]
        term1 = torch.abs(M1) * torch.exp(-1j * torch.angle(M1))
        term2 = torch.abs(M2) * torch.exp(1j * torch.angle(M2))

        FFtmp = term1 * term2 / S

        output = torch.fft.ifftn(FFtmp, [el, el], [0, 1]).real.unsqueeze(0)

        if self.normalize_spst_tensors:
            output = self.normalize(output)
        
        return output
    
    def fft_shift(self, input_autocorr):
            """Performs a circular shift on the input autocorrelation tensor."""
            _, _, H, W = input_autocorr.shape
            return torch.roll(input_autocorr, shifts=(H // 2, W // 2), dims=(-2, -1))

    def normalize(self, tensor, eps=1e-6):
        """
        Normalize tensor to be in the range [0, 1].

        Tensor has to be of the shape [1, H, W] or [H, W]
        """
        tensor = (tensor - self.min_fft_pixel_value) / (self.max_fft_pixel_value - self.min_fft_pixel_value + eps)
        return tensor

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


class TwoPointAutocorrelation:
    """
    Warning: 
    
    - Not to be used for optimization purposes. This class cannot be used for optimizing NNs because backprop won't work here.
    - The reason why backprop won't work here is because of the absense of soft inequality,
    however, it can be used to calulate the exact spatial statistics of microstructure images. 

    - To be used to calculate autocorrelation for single binary image inputs.
    """

    def __init__(self, H=2):
        # H: Number of phases (int)
        self.H = H
    
    def calculate_microstructure_function(self, img):
        """
        Inputs:
        img: image (Torch tensor of shape (1, H, W))

        Returns: Microstructure function of image (Torch tensor of shape (self.H, H, W))
        """
        el = img.shape[-1]
        mf = torch.zeros((self.H, el, el))
        for h in range(self.H):
            mf[h, ...] = img.eq(h).clone()
        return mf
    
    def calculate_2point_torch_spatialstat(self, mf):
        """
        Calculates two-point spatial statistics for the microstructure function tensor.

        Parameters:
            mf (torch.Tensor): Microstructure function tensor of shape (num_phases, edge_len, edge_len).

        Returns:
            torch.Tensor: Two-point spatial statistics tensor of shape (1, edge_len, edge_len)
        """
        iA, iB = 0, 0
        el = mf.shape[-1]
        M = torch.zeros((self.H, el, el), dtype=torch.complex128)
        for h in range(self.H):
            M[h, ...] = torch.fft.fftn(mf[h, ...], dim=[0, 1])

        S = el**2
        M1, M2 = M[iA, ...], M[iB, ...]
        term1 = torch.abs(M1) * torch.exp(-1j * torch.angle(M1))
        term2 = torch.abs(M2) * torch.exp(1j * torch.angle(M2))

        FFtmp = term1 * term2 / S

        output = torch.fft.ifftn(FFtmp, [el, el], [0, 1]).real.unsqueeze(0)        
        return output

    def fft_shift(self, input_autocorr):
        """
        Performs a circular shift on the input autocorrelation tensor.
        img: autocorrelation (Torch tensor of shape (1, H, W))

        Returns: shifted autocorrelation (Torch tensor of shape (1, H, W))
        """
        _, H, W = input_autocorr.shape
        return torch.roll(input_autocorr, shifts=(H // 2, W // 2), dims=(-2, -1))

    def forward(self, img):
        """
        calculates the two-point autocorrelation
        img: Torch tensor of shape (1, H, W)

        out: Torch tensor of shape (1, H, W)
        """
        assert (len(img.shape) == 3 and img.shape[0]==1 and img.shape[1]==img.shape[2]), "Input not a single-channel, square, individual image!"
        mf = self.calculate_microstructure_function(img)
        autocorr = self.calculate_2point_torch_spatialstat(mf)
        shifted_autocorr = self.fft_shift(autocorr)
        return shifted_autocorr
    
