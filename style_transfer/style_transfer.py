"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import copy
from dataclasses import dataclass
from functools import partial
import time
import warnings

import numpy as np
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

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = models.vgg19(pretrained=True).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

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

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = torch.device(devices[i])
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        input = self.normalize(input)
        for i in range(max(layers) + 1):
            input = self.model[i](input.to(self.devices[i]))
            if i in layers:
                feats[i] = input
        return feats


class ScaledMSELoss(nn.Module):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1.
    This differs from Gatys at al. (2015) and Johnson et al."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def extra_repr(self):
        return f'eps={self.eps:g}'

    def forward(self, input, target):
        diff = input - target
        return diff.pow(2).sum() / diff.abs().sum().add(self.eps)


class ContentLoss(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    def forward(self, input):
        return self.loss(input, self.target)


class StyleLoss(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    @staticmethod
    def get_target(target):
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. (2015) and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input):
        return self.loss(self.get_target(input), self.target)


class TVLoss(nn.Module):
    """L2 total variation loss, as in Mahendran et al."""

    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        return (x_diff**2 + y_diff**2).mean()


class SumLoss(nn.ModuleList):
    def __init__(self, losses, verbose=False):
        super().__init__(losses)
        self.verbose = verbose

    def forward(self, *args, **kwargs):
        losses = [loss(*args, **kwargs) for loss in self]
        if self.verbose:
            for i, loss in enumerate(losses):
                print(f'({i}): {loss.item():g}')
        return sum(loss.to(losses[-1].device) for loss in losses)


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class LayerApply(nn.Module):
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'(layer): {self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


class EMA(nn.Module):
    """A bias-corrected exponential moving average, as in Kingma et al. (Adam)."""

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
    time: float
    gpu_ram: int


class StyleTransfer:
    def __init__(self, devices=['cpu'], pooling='max'):
        self.devices = [torch.device(device) for device in devices]
        self.image = None
        self.average = None

        # The default content and style layers follow Gatys et al. (2015).
        self.content_layers = [22]
        self.style_layers = [1, 6, 11, 20, 29]

        # The weighting of the style layers differs from Gatys et al. (2015) and Johnson et al.
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        self.model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling)

        if len(self.devices) == 1:
            device_plan = {0: self.devices[0]}
        elif len(self.devices) == 2:
            device_plan = {0: self.devices[0], 5: self.devices[1]}
        else:
            raise ValueError('Only 1 or 2 devices are supported.')

        self.model.distribute_layers(device_plan)

    def get_image_tensor(self):
        return self.average.get().detach()[0].clamp(0, 1)

    def get_image(self, image_type='pil'):
        if self.average is not None:
            image = self.get_image_tensor()
            if image_type.lower() == 'pil':
                return TF.to_pil_image(image)
            elif image_type.lower() == 'np_uint16':
                arr = image.cpu().movedim(0, 2).numpy()
                return np.uint16(np.round(arr * 65535))
            else:
                raise ValueError("image_type must be 'pil' or 'np_uint16'")

    def stylize(self, content_image, style_images, *,
                style_weights=None,
                content_weight: float = 0.015,
                tv_weight: float = 2.,
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

        tv_loss = Scale(LayerApply(TVLoss(), 'input'), tv_weight)

        scales = gen_scales(min_scale, end_scale)

        cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_image.resize((cw, ch), Image.LANCZOS))[None]
        elif init == 'gray':
            self.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
        elif init == 'uniform':
            self.image = torch.rand([1, 3, ch, cw])
        elif init == 'style_mean':
            means = []
            for i, image in enumerate(style_images):
                means.append(TF.to_tensor(image).mean(dim=(1, 2)) * style_weights[i])
            self.image = torch.rand([1, 3, ch, cw]) / 255 + sum(means)[None, :, None, None]
        else:
            raise ValueError("init must be one of 'content', 'gray', 'uniform', 'style_mean'")
        self.image = self.image.to(self.devices[0])

        opt = None

        # Stylize the image at successively finer scales, each greater by a factor of sqrt(2).
        # This differs from the scheme given in Gatys et al. (2016).
        for scale in scales:
            if self.devices[0].type == 'cuda':
                torch.cuda.empty_cache()

            cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
            content = TF.to_tensor(content_image.resize((cw, ch), Image.LANCZOS))[None]
            content = content.to(self.devices[0])

            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.average = EMA(self.image, avg_decay)
            self.image.requires_grad_()

            print(f'Processing content image ({cw}x{ch})...')
            content_feats = self.model(content, layers=self.content_layers)
            content_losses = []
            for layer, weight in zip(self.content_layers, content_weights):
                target = content_feats[layer]
                content_losses.append(Scale(LayerApply(ContentLoss(target), layer), weight))

            style_targets, style_losses = {}, []
            for i, image in enumerate(style_images):
                if style_size is None:
                    sw, sh = size_to_fit(image.size, round(scale * style_scale_fac))
                else:
                    sw, sh = size_to_fit(image.size, style_size)
                style = TF.to_tensor(image.resize((sw, sh), Image.LANCZOS))[None]
                style = style.to(self.devices[0])
                print(f'Processing style image ({sw}x{sh})...')
                style_feats = self.model(style, layers=self.style_layers)
                # Take the weighted average of multiple style targets (Gram matrices).
                for layer in self.style_layers:
                    target = StyleLoss.get_target(style_feats[layer]) * style_weights[i]
                    if layer not in style_targets:
                        style_targets[layer] = target
                    else:
                        style_targets[layer] += target
            for layer, weight in zip(self.style_layers, self.style_weights):
                target = style_targets[layer]
                style_losses.append(Scale(LayerApply(StyleLoss(target), layer), weight))

            crit = SumLoss([*content_losses, *style_losses, tv_loss])

            opt2 = optim.Adam([self.image], lr=step_size)
            # Warm-start the Adam optimizer if this is not the first scale.
            if scale != scales[0]:
                opt_state = scale_adam(opt.state_dict(), (ch, cw))
                opt2.load_state_dict(opt_state)
            opt = opt2

            if self.devices[0].type == 'cuda':
                torch.cuda.empty_cache()

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
                    gpu_ram = 0
                    for device in self.devices:
                        if device.type == 'cuda':
                            gpu_ram = max(gpu_ram, torch.cuda.max_memory_allocated(device))
                    callback(STIterate(w=cw, h=ch, i=i, i_max=actual_its, loss=loss.item(),
                                       time=time.time(), gpu_ram=gpu_ram))

            # Initialize each new scale with the previous scale's averaged iterate.
            with torch.no_grad():
                self.image.copy_(self.average.get())

        return self.get_image()
