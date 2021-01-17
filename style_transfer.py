#!/usr/bin/env python3

"""Neural style transfer in PyTorch."""

import argparse
import copy
from pathlib import Path
import warnings

from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF


class VGGFeatures(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = sorted(set(layers))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = models.vgg19(pretrained=True).features[:self.layers[-1]+1]
        self.model.eval()
        self.model.requires_grad_(False)

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
    def forward(self, input):
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        return torch.sum(diff)


class WeightedLoss(nn.ModuleList):
    def __init__(self, losses, weights, verbose=False):
        super().__init__(losses)
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'({i}) {self[i]!r}: {loss.item():g}')

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)


class Normalize(nn.Module):
    def __init__(self, module, scale=1, eps=1e-8):
        super().__init__()
        self.module = module
        self.module.register_backward_hook(self._hook)
        self.scale = scale
        self.eps = eps

    def _hook(self, module, grad_input, grad_output):
        i, *rest = grad_input
        dims = list(range(1, i.ndim))
        norm = abs(i).sum(dim=dims, keepdims=True) + self.eps
        return i * self.scale / norm, *rest

    def extra_repr(self):
        return f'scale={self.scale!r}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LayerApply(nn.Module):
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'layer={self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


def load_image(path):
    return Image.open(path).convert('RGB')


def save_image_t(input, path, **kwargs):
    TF.to_pil_image(input).save(path, **kwargs)


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


def scales(start, n):
    for i in range(n):
        yield round(start * pow(2, 1/2)**i)


def scale_adam(state, shape):
    state = copy.deepcopy(state)
    for group in state['state'].values():
        exp_avg = group['exp_avg']
        exp_avg_sq = group['exp_avg_sq']
        group['exp_avg'] = F.interpolate(exp_avg, shape, mode='bicubic')
        group['exp_avg_sq'] = F.interpolate(exp_avg_sq, shape, mode='bilinear')
        group['exp_avg_sq'].relu_()
    return state


def main():
    warnings.simplefilter('ignore', UserWarning)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('content', type=Path, help='the content image')
    p.add_argument('style', type=Path, help='the style image')
    p.add_argument('output', type=Path, nargs='?', default=Path('out.png'),
                   help='the output image')
    p.add_argument('--device', type=str, default=None,
                   help='the device name to use')
    p.add_argument('--content-weight', '-cw', type=float, default=0.01,
                   help='the content weight')
    p.add_argument('--tv-weight', '-tw', type=float, default=2e-7,
                   help='the smoothing weight')

    args = p.parse_args()

    content_img = load_image(args.content)
    style_img = load_image(args.style)

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('Using device:', device)
    torch.tensor(0).to(device)

    content_layers = [22]
    content_weights = [args.content_weight / len(content_layers)] * len(content_layers)

    style_layers = [1, 6, 11, 20, 29]
    style_weights = [256, 64, 16, 4, 1]
    weight_sum = sum(abs(w) for w in style_weights)
    style_weights = [w / weight_sum for w in style_weights]

    tv_loss = LayerApply(TVLoss(), 'input')

    print('Loading model...')
    model = VGGFeatures(layers=style_layers + content_layers).to(device)

    initial_scale = 64
    init_with_content = True

    cw, ch = size_to_fit(content_img.size, initial_scale, scale_up=True)
    if init_with_content:
        image = TF.to_tensor(content_img.resize((cw, ch), Image.LANCZOS))[None]
    else:
        image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
    image = image.to(device)

    try:
        for scale in scales(initial_scale, 9):
            cw, ch = size_to_fit(content_img.size, scale, scale_up=True)
            sw, sh = size_to_fit(style_img.size, scale)

            content = TF.to_tensor(content_img.resize((cw, ch), Image.LANCZOS))[None]
            style = TF.to_tensor(style_img.resize((sw, sh), Image.LANCZOS))[None]
            content, style = content.to(device), style.to(device)

            image = F.interpolate(image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            image.requires_grad_()

            print(f'Processing content image ({cw}x{ch})...')
            content_feats = model(content, layers=content_layers)
            content_losses = []
            for i, layer in enumerate(content_layers):
                weight = content_weights[i]
                target = content_feats[layer]
                loss = LayerApply(Normalize(ContentLoss(target), weight), layer)
                content_losses.append(loss)

            print(f'Processing style image ({sw}x{sh})...')
            style_feats = model(style, layers=style_layers)
            style_losses = []
            for i, layer in enumerate(style_layers):
                weight = style_weights[i]
                target = StyleLoss.get_target(style_feats[layer])
                loss = LayerApply(Normalize(StyleLoss(target), weight), layer)
                style_losses.append(loss)

            crit = WeightedLoss([*content_losses, *style_losses, tv_loss],
                                [*content_weights, *style_weights, args.tv_weight])

            opt = optim.Adam([image], lr=5/255)

            # if scale != initial_scale:
            #     opt_state = scale_adam(opt.state_dict(), (ch, cw))
            #     opt.load_state_dict(opt_state)

            for i in range(1, 501):
                feats = model(image)
                feats['input'] = image
                loss = crit(feats)
                print(f'{i} {loss.item() / image.numel():g}')
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    image.clamp_(0, 1)

    except KeyboardInterrupt:
        pass

    print(f'Writing image to {args.output}.')
    save_image_t(image[0], args.output)


if __name__ == '__main__':
    main()
