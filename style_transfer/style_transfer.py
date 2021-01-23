"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

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
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, different from the original model.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch trained VGG-19 has different parameters from the original model.
        self.model = models.vgg19(pretrained=True).features[:self.layers[-1] + 1]

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Also change ceil_mode to True.
                self.model[i] = Scale(self.poolings[pooling](2, ceil_mode=True), pool_scale)

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
        feats = {'input': input}
        cur = 0
        for layer in layers:
            input = self.model[cur:layer+1](input)
            feats[layer] = input
            cur = layer + 1
        return feats


def scaled_mse_loss(input, target, eps=1e-8):
    """A custom loss function, MSE with gradient L1 norm approximately 1."""
    diff = input - target
    return diff.pow(2).sum() / diff.abs().sum().add(eps)


class ContentLoss(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, input):
        return scaled_mse_loss(input, self.target, eps=self.eps)


class StyleLoss(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.register_buffer('eps', torch.tensor(eps))

    @staticmethod
    def get_target(target):
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input):
        return scaled_mse_loss(self.get_target(input), self.target, eps=self.eps)


class TVLoss(nn.Module):
    """Total variation loss, which supports L1 vectorial total variation (p=1) and L2 total
    variation (p=2) as in Mahendran et al."""

    def __init__(self, p, eps=1e-8):
        super().__init__()
        assert p in {1, 2}
        self.register_buffer('p', torch.tensor(p))
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = diff.add(self.eps).mean(dim=1).sqrt()
        return diff.mean()


class WeightedLoss(nn.ModuleList):
    def __init__(self, losses, weights, verbose=False):
        super().__init__(losses)
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'{i}: {loss.item():g}')

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            loss_value = loss(*args, **kwargs) * weight if weight else torch.tensor(0)
            losses.append(loss_value)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'scale={self.scale!r}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class LayerApply(nn.Module):
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'layer={self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


class EMA(nn.Module):
    """A bias-corrected exponential moving average (as in Adam)."""

    def __init__(self, input, decay):
        super().__init__()
        self.register_buffer('value', torch.zeros_like(input))
        self.register_buffer('decay', torch.tensor(decay))
        self.register_buffer('accum', torch.tensor(1.))
        self.update(input)

    def get(self):
        return self.value / (1 - self.accum)

    def update(self, input):
        self.accum *= self.decay
        self.value *= self.decay
        self.value += (1 - self.decay) * input

    def mutate(self, new_value):
        self.value = new_value * (1 - self.accum)


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
    scales = set()
    while scale >= start:
        scales.add(scale)
        i += 1
        scale = round(end / pow(2, i/2))
    return sorted(scales)


def interpolate(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.interpolate(*args, **kwargs)


def scale_adam(state, shape):
    """Prepares a state dict to warm-start the Adam optimizer at a new scale."""
    state = copy.deepcopy(state)
    for group in state['state'].values():
        exp_avg, exp_avg_sq = group['exp_avg'], group['exp_avg_sq']
        group['exp_avg'] = interpolate(exp_avg, shape, mode='bicubic')
        group['exp_avg_sq'] = interpolate(exp_avg_sq, shape, mode='bilinear').relu_()
        if 'max_exp_avg_sq' in group:
            max_exp_avg_sq = group['max_exp_avg_sq']
            group['max_exp_avg_sq'] = interpolate(max_exp_avg_sq, shape, mode='bilinear').relu_()
    return state


@dataclass
class STIterate:
    w: int
    h: int
    i: int
    i_max: int
    loss: float


class StyleTransfer:
    def __init__(self, device='cpu', pooling='max'):
        self.device = torch.device(device)
        self.image = None
        self.average = None

        self.content_layers = [22]

        self.style_layers = [1, 6, 11, 20, 29]
        style_weights = [256, 64, 16, 4, 1]  # Custom weighting of style layers.
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling)
        self.model = model.to(self.device)

    def get_image(self):
        if self.average is not None:
            return TF.to_pil_image(self.average.get()[0].clamp(0, 1))

    def stylize(self, content_image, style_images, *,
                style_weights=None,
                content_weight: float = 0.01,
                tv_weight_1: float = 0.,
                tv_weight_2: float = 0.15,
                min_scale: int = 128,
                end_scale: int = 512,
                iterations: int = 500,
                initial_iterations: int = 1000,
                step_size: float = 0.02,
                avg_decay: float = 0.99,
                init: str = 'content',
                style_scale_fac: float = 1.,
                style_size: int = None,
                callback=None):

        min_scale = min(min_scale, end_scale)
        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)

        if style_weights is None:
            style_weights = [1 / len(style_images)] * len(style_images)
        else:
            weight_sum = sum(abs(w) for w in style_weights)
            style_weights = [weight / weight_sum for weight in style_weights]
        if len(style_images) != len(style_weights):
            raise ValueError('style_images and style_weights must have the same length')

        tv_losses = [LayerApply(TVLoss(p=1), 'input'),
                     LayerApply(TVLoss(p=2), 'input')]
        tv_weights = [tv_weight_1, tv_weight_2]

        scales = gen_scales(min_scale, end_scale)

        cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_image.resize((cw, ch), Image.LANCZOS))[None]
        elif init == 'gray':
            self.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
        elif init == 'random':
            self.image = torch.rand([1, 3, ch, cw])
        else:
            raise ValueError("init must be one of 'content', 'gray', 'random'")
        self.image = self.image.to(self.device)
        self.average = EMA(self.image, avg_decay)

        opt = None

        # Stylize the image at successively finer scales, each greater by a factor of sqrt(2).
        for scale in scales:
            cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
            content = TF.to_tensor(content_image.resize((cw, ch), Image.LANCZOS))[None]
            content = content.to(self.device)

            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            # Warm-start the average iterate with the old scale's average iterate.
            new_avg = interpolate(self.average.get(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.average.mutate(new_avg)
            self.image.requires_grad_()

            print(f'Processing content image ({cw}x{ch})...')
            content_feats = self.model(content, layers=self.content_layers)
            content_losses = []
            for layer in self.content_layers:
                target = content_feats[layer]
                content_losses.append(LayerApply(ContentLoss(target), layer))

            style_targets = {}
            style_losses = []
            for i, image in enumerate(style_images):
                if style_size is None:
                    sw, sh = size_to_fit(image.size, round(scale * style_scale_fac))
                else:
                    sw, sh = size_to_fit(image.size, style_size)
                style = TF.to_tensor(image.resize((sw, sh), Image.LANCZOS))[None]
                style = style.to(self.device)
                print(f'Processing style image ({sw}x{sh})...')
                style_feats = self.model(style, layers=self.style_layers)
                # Take the weighted average of multiple style targets (Gram matrices).
                for layer in self.style_layers:
                    target = StyleLoss.get_target(style_feats[layer]) * style_weights[i]
                    if layer not in style_targets:
                        style_targets[layer] = target
                    else:
                        style_targets[layer] += target
            for layer in self.style_layers:
                target = style_targets[layer]
                style_losses.append(LayerApply(StyleLoss(target), layer))

            crit = WeightedLoss([*content_losses, *style_losses, *tv_losses],
                                [*content_weights, *self.style_weights, *tv_weights])

            opt2 = optim.Adam([self.image], lr=step_size)
            # Warm-start the Adam optimizer if this is not the first scale.
            if scale != scales[0]:
                opt_state = scale_adam(opt.state_dict(), (ch, cw))
                opt2.load_state_dict(opt_state)
            opt = opt2

            actual_its = initial_iterations if scale == scales[0] else iterations
            for i in range(1, actual_its + 1):
                feats = self.model(self.image)
                loss = crit(feats)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # Enforce box constraints.
                with torch.no_grad():
                    self.image.clamp_(0, 1)
                self.average.update(self.image)
                if callback is not None:
                    callback(STIterate(cw, ch, i, actual_its, loss.item()))

            # Initialize each new scale with the previous scale's averaged iterate.
            with torch.no_grad():
                self.image.copy_(self.average.get())

        return self.get_image()
