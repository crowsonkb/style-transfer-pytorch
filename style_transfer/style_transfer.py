"""Neural style transfer in PyTorch. Implements A Neural Algorithm of Artistic
Style (http://arxiv.org/abs/1508.06576)."""

import copy
from dataclasses import dataclass
from functools import partial
import warnings

from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF


class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = models.vgg19(pretrained=True).features[:self.layers[-1] + 1]
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.MaxPool2d):
                self.model[i] = self.poolings[pooling](2, ceil_mode=True)
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    def forward(self, input, layers=None):
        layers = self.layers if layers is None else sorted(set(layers))
        feats = {}
        cur = 0
        for layer in layers:
            input = self.model[cur:layer+1](input)
            feats[layer] = input
            cur = layer + 1
        return feats


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    def forward(self, input):
        return F.mse_loss(input, self.target, reduction='sum')


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    @staticmethod
    def get_target(target):
        mat = target.flatten(-2)
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input):
        return F.mse_loss(self.get_target(input), self.target, reduction='sum')


class TVLoss(nn.Module):
    def __init__(self, p, eps=1e-8):
        super().__init__()
        assert p in {1, 2}
        self.p = p
        self.eps = eps

    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1).sqrt()
        return torch.mean(diff)


class WeightedLoss(nn.ModuleList):
    def __init__(self, losses, weights, verbose=False):
        super().__init__(losses)
        self.weights = weights
        self.verbose = verbose
        self.losses = None

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'{i}: {loss.item():g}')

    def get_scaled_loss(self):
        losses = []
        for i, crit in enumerate(self):
            if hasattr(crit, 'get_scaled_loss'):
                losses.append(crit.get_scaled_loss() * self.weights[i])
            else:
                losses.append(self.losses[i])
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)

    def forward(self, *args, **kwargs):
        self.losses = []
        for loss, weight in zip(self, self.weights):
            self.losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(self.losses)
        return sum(self.losses)


class NormalizeGrad(nn.Module):
    """Normalizes and optionally scales the enclosed module's gradient."""

    def __init__(self, module, scale=1, eps=1e-8):
        super().__init__()
        self.module = module
        self.module.register_backward_hook(self._hook)
        self.scale = scale
        self.eps = eps
        self.fac = None
        self.loss = None

    def _hook(self, module, grad_input, grad_output):
        grad, *rest = grad_input
        dims = list(range(1, grad.ndim))
        norm = abs(grad).sum(dim=dims, keepdims=True)
        self.fac = self.scale / (norm + self.eps)
        return grad * self.fac, *rest

    def get_scaled_loss(self):
        return self.loss * self.fac

    def extra_repr(self):
        return f'scale={self.scale!r}'

    def forward(self, *args, **kwargs):
        self.loss = self.module(*args, **kwargs)
        return self.loss


class LayerApply(nn.Module):
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'layer={self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


def size_to_fit(size, max_dim, scale_up=False):
    w, h = size
    if not scale_up and max(h, w) <= max_dim:
        return w, h
    new_w, new_h = max_dim, max_dim
    if h > w:
        new_w = round(max_dim * w / h)
    else:
        new_h = round(max_dim * h / w)
    return new_w, new_h


def gen_scales(start, end):
    scale = end
    i = 0
    scales = []
    while scale >= start:
        scales.insert(0, scale)
        i += 1
        scale = round(end / pow(2, i/2))
    return scales


def interpolate(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.interpolate(*args, **kwargs)


def scale_adam(state, shape):
    state = copy.deepcopy(state)
    for group in state['state'].values():
        exp_avg = group['exp_avg']
        exp_avg_sq = group['exp_avg_sq']
        group['exp_avg'] = interpolate(exp_avg, shape, mode='bicubic')
        group['exp_avg_sq'] = interpolate(exp_avg_sq, shape, mode='bilinear')
        group['exp_avg_sq'].relu_()
    return state


@dataclass
class STIterate:
    scale: int
    i: int
    loss: float


class StyleTransfer:
    def __init__(self, device='cpu', pooling='max'):
        self.device = torch.device(device)
        self.image = None

        self.content_layers = [22]

        self.style_layers = [1, 6, 11, 20, 29]
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling)
        self.model = model.to(self.device)

    def get_image(self):
        if self.image is not None:
            return TF.to_pil_image(self.image[0].clamp(0, 1))

    def stylize(self, content_img, style_imgs, *,
                style_img_weights=None,
                content_weight: float = 0.01,
                tv_weight_1: float = 0.,
                tv_weight_2: float = 0.15,
                min_scale: int = 64,
                end_scale: int = 512,
                iterations: int = 500,
                step_size: float = 0.02,
                init: str = 'content',
                style_scale_fac: float = 1.,
                style_size: int = None,
                callback=None):

        min_scale = min(min_scale, end_scale)
        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)

        if style_img_weights is None:
            style_img_weights = [1 / len(style_imgs)] * len(style_imgs)
        else:
            weight_sum = sum(style_img_weights)
            style_img_weights = [weight / weight_sum for weight in style_img_weights]
        if len(style_imgs) != len(style_img_weights):
            raise ValueError('style_imgs and style_img_weights must have the same length')

        tv_losses = [LayerApply(TVLoss(p=1), 'input'),
                     LayerApply(TVLoss(p=2), 'input')]
        tv_weights = [tv_weight_1, tv_weight_2]

        scales = gen_scales(min_scale, end_scale)

        cw, ch = size_to_fit(content_img.size, scales[0], scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_img.resize((cw, ch), Image.LANCZOS))[None]
        elif init == 'gray':
            self.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
        elif init == 'random':
            self.image = torch.rand([1, 3, ch, cw])
        else:
            raise ValueError("init must be one of 'content', 'gray', 'random'")
        self.image = self.image.to(self.device)

        opt = None

        for scale in scales:
            cw, ch = size_to_fit(content_img.size, scale, scale_up=True)
            content = TF.to_tensor(content_img.resize((cw, ch), Image.LANCZOS))[None]
            content = content.to(self.device)

            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.image.requires_grad_()

            print(f'Processing content image ({cw}x{ch})...')
            content_feats = self.model(content, layers=self.content_layers)
            content_losses = []
            for i, layer in enumerate(self.content_layers):
                weight = content_weights[i]
                target = content_feats[layer]
                loss = NormalizeGrad(LayerApply(ContentLoss(target), layer), abs(weight))
                content_losses.append(loss)

            style_targets = {}
            style_losses = []
            for i, style_img in enumerate(style_imgs):
                if style_size is None:
                    sw, sh = size_to_fit(style_img.size, round(scale * style_scale_fac))
                else:
                    sw, sh = size_to_fit(style_img.size, style_size)
                style = TF.to_tensor(style_img.resize((sw, sh), Image.LANCZOS))[None]
                style = style.to(self.device)
                print(f'Processing style image ({sw}x{sh})...')
                style_feats = self.model(style, layers=self.style_layers)
                for layer in self.style_layers:
                    target = StyleLoss.get_target(style_feats[layer]) * style_img_weights[i]
                    if layer not in style_targets:
                        style_targets[layer] = target
                    else:
                        style_targets[layer] += target
            for i, layer in enumerate(self.style_layers):
                weight = self.style_weights[i]
                target = style_targets[layer]
                loss = NormalizeGrad(LayerApply(StyleLoss(target), layer), abs(weight))
                style_losses.append(loss)

            crit = WeightedLoss([*content_losses, *style_losses, *tv_losses],
                                [*content_weights, *self.style_weights, *tv_weights])

            opt2 = optim.Adam([self.image], lr=step_size)
            if scale != scales[0]:
                opt_state = scale_adam(opt.state_dict(), (ch, cw))
                opt2.load_state_dict(opt_state)
            opt = opt2

            for i in range(1, iterations + 1):
                feats = self.model(self.image)
                feats['input'] = self.image
                loss = crit(feats)
                opt.zero_grad()
                loss.backward()
                loss2 = crit.get_scaled_loss()
                opt.step()
                with torch.no_grad():
                    self.image.clamp_(0, 1)
                if callback is not None:
                    callback(STIterate(scale, i, loss2.item()))

        return self.get_image()
