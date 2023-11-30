import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(Encoder, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)  # 112x112
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # 56x56
        self.bn2 = nn.BatchNorm2d(32)
        self.fc_mu = nn.Linear(32 * 56 * 56, bottleneck_size)
        self.fc_log_var = nn.Linear(32 * 56 * 56, bottleneck_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(bottleneck_size, 32 * 56 * 56)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 112x112
        self.bn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)  # 224x224

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 56, 56)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x))
        return x

class SmallerVAE(nn.Module):
    def __init__(self, bottleneck_size):
        super(SmallerVAE, self).__init__()
        self.CNN_embed_dim = bottleneck_size
        self.encode = Encoder(bottleneck_size)
        self.decode = Decoder(bottleneck_size)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        reconstruction = self.decode(z)
        return reconstruction, z, mu, log_var
