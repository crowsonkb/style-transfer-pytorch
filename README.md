# style-transfer-pytorch

An implementation of neural style transfer ([A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)) in PyTorch, supporting CPUs and Nvidia GPUs. It performs automatic multi-scale (coarse-to-fine) stylization for good quality output at high resolutions.

## Installation

Python 3.8+ is required (it may work on 3.7 but has not been tested).

[PyTorch](https://pytorch.org) is required: follow [their installation instructions](https://pytorch.org/get-started/locally/) before proceeding.

To install `style-transfer-pytorch`, first clone the repository, then run the command:

```sh
pip install -e PATH_TO_REPO
```

This will install the `style_transfer` CLI tool.

## Basic usage

```sh
style_transfer CONTENT_IMAGE STYLE_IMAGE [STYLE_IMAGE ...] [-o OUTPUT_IMAGE]
```

`style_transfer` has many optional arguments: run it with the `--help` argument to see a full list.

## References

L. Gatys, A. Ecker, M. Bethge (2015), "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)"

L. Gatys, A. Ecker, M. Bethge, A. Hertzmann, E. Shechtman (2016), "[Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865)"

J. Johnson, A. Alahi, L. Fei-Fei (2016), "[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)"

P. Blomgren, T. Chan (1998), "[Color TV: Total Variation Methods for Restoration of Vector-Valued Images](https://ieeexplore.ieee.org/document/661180)"

A. Mahendran, A. Vedaldi (2014), "[Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035)"

D. Kingma, J. Ba (2014), "[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)"

K. Simonyan, A. Zisserman (2014), "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)"
