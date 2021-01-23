# style-transfer-pytorch

An implementation of neural style transfer ([A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)) in PyTorch, supporting CPUs and Nvidia GPUs. It performs automatic multi-scale (coarse-to-fine) stylization for good quality output at high resolutions, as well as other quality-increasing techniques such as gradient normalization and iterate averaging.

## Installation

Python 3.8+ is required (it may work on 3.7 but has not been tested).

[PyTorch](https://pytorch.org) is required: follow [their installation instructions](https://pytorch.org/get-started/locally/) before proceeding.

To install `style-transfer-pytorch`, first clone the repository, then run the command:

```sh
pip install -e PATH_TO_REPO
```

This will install the `style_transfer` CLI tool. `style_transfer` uses a pre-trained VGG-19 model (Simonyan et al.), which is 548MB in size, and will download it when first run.

## Basic usage

```sh
style_transfer CONTENT_IMAGE STYLE_IMAGE [STYLE_IMAGE ...] [-o OUTPUT_IMAGE]
```

`style_transfer` has many optional arguments: run it with the `--help` argument to see a full list. Particularly notable ones include:

- `--device` manually sets the PyTorch device name. It can be set to `cpu` to force it to run on the CPU on a machine with a supported GPU, or to e.g. `cuda:1` (zero indexed) to select the second CUDA GPU. `style_transfer` will automatically use the first visible CUDA GPU, falling back to the CPU, if it is omitted.

- `-s` (`--end-scale`) sets the maximum image dimension (height and width) of the output. A large image (e.g. 2896x2172) can take around fifteen minutes to generate on an RTX 3090 and will require nearly all of its 24GB of memory. Since both memory usage and runtime increase linearly in the number of pixels (quadratically in the value of the `--end-scale` parameter), users with less GPU memory or who do not want to wait very long are encouraged to use smaller resolutions. The default is 512.

- `--style-weights` specifies factors for the weighted average of multiple styles if there is more than one style image specified. These factors are automatically normalized to sum to 1. If omitted, the styles will be blended equally.

- `-cw` (`--content-weight`) sets the degree to which features from the content image are included in the output image. The default is 0.01.

- `-tw` (`--tv-weight-2`) sets the strength of the smoothness prior. The default is 0.15.

## References

L. Gatys, A. Ecker, M. Bethge (2015), "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)"

L. Gatys, A. Ecker, M. Bethge, A. Hertzmann, E. Shechtman (2016), "[Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865)"

J. Johnson, A. Alahi, L. Fei-Fei (2016), "[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)"

P. Blomgren, T. Chan (1998), "[Color TV: Total Variation Methods for Restoration of Vector-Valued Images](https://ieeexplore.ieee.org/document/661180)"

A. Mahendran, A. Vedaldi (2014), "[Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035)"

D. Kingma, J. Ba (2014), "[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)"

K. Simonyan, A. Zisserman (2014), "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)"
