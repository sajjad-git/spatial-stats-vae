import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from utils import Normalization


class ContentLoss(nn.Module):

    def __init__(self, content_layer, device):
        super(ContentLoss, self).__init__()
        self.cnn = VGG19Pipeline(device).to(device).model
        i = 0
        for mod in self.cnn.named_modules():
            if mod[0] == content_layer:
                break
            i += 1

        self.cnn = self.cnn[:(i+1)]

    def forward(self, input, target):
        input = input.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

        input = self.cnn(input)
        target = self.cnn(target)
        loss = F.mse_loss(input, target)
        return loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, style_layer, device):
        super(StyleLoss, self).__init__()
        self.cnn = VGG19Pipeline(device).to(device).model

        i = 0
        for mod in self.cnn.named_modules():
            if mod[0] == style_layer:
                break
            i += 1

        self.cnn = self.cnn[:(i+1)]

    def forward(self, input, target):
        input = input.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

        input = gram_matrix(self.cnn(input))
        target = gram_matrix(self.cnn(target))
        loss = F.mse_loss(input, target)
        return loss


class VGG19Pipeline(nn.Module):
    def __init__(self, device):
        super(VGG19Pipeline, self).__init__()
        # import the pretrained network
        self.cnn = models.vgg19(pretrained=True).features.eval()
        for p in self.cnn.parameters():
            p.requires_grad = False

        # normalize because vgg models are trained on normalized images
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.normalize = Normalization(self.cnn_normalization_mean,
                                       self.cnn_normalization_std,
                                       device)
        self.model = nn.Sequential(self.normalize)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)