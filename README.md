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
style_transfer CONTENT_IMAGE STYLE_IMAGE [-o OUTPUT_IMAGE]
```

`style_transfer` has many optional arguments: run it with the `--help` argument to see a full list.
