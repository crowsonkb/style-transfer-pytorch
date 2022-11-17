"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

from pathlib import Path

srgb_profile = (Path(__file__).resolve().parent / 'sRGB Profile.icc').read_bytes()
del Path

from .style_transfer import STIterate, StyleTransfer
from .web_interface import WebInterface
