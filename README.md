# style-transfer-pytorch

An implementation of neural style transfer ([A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)) in PyTorch, supporting CPUs and Nvidia GPUs. It does automatic multi-scale (coarse-to-fine) stylization to produce high-quality high resolution stylizations, even up to print resolution if the GPUs have sufficient memory. If two GPUs are available, they can both be used to increase the maximum resolution. (Using two GPUs is not faster than using one.)

The algorithm has been modified from that in the literature by:

- Using the PyTorch pre-trained VGG-19 weights instead of the original VGG-19 weights

- Changing the padding mode of the first layer of VGG-19 to 'replicate', to reduce edge artifacts

- When using average or L2 pooling, scaling the result by an empirically derived factor to ensure that the magnitude of the result stays the same on average (Gatys et al. (2015) did not do this)

- Using an approximation to the MSE loss such that its gradient L1 norm is approximately 1 for content and style losses (in order to approximate the effects of gradient normalization, which produces better visual quality)

- Normalizing the Gram matrices by the number of elements in each feature map channel rather than by the total number of elements (Johnson et al.) or not normalizing (Gatys et al. (2015))

- Taking an exponential moving average over the iterates to reduce iterate noise (each new scale is initialized with the previous scale's averaged iterate)

- Warm-starting the Adam optimizer with scaled-up versions of its first and second moment buffers at the beginning of each new scale, to prevent noise from being added to the iterates at the beginning of each scale

- Using non-equal weights for the style layers to improve visual quality

- Stylizing the image at progressively larger scales, each greater by a factor of sqrt(2) (this is improved from the multi-scale scheme given in Gatys et al. (2016))

## Example outputs (click for the full-sized version)

<a href="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst195.jpg"><img src="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst195_small.jpg" width="512" height="401"></a>

<a href="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst235.jpg"><img src="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst235_small.jpg" width="512" height="401"></a>

<a href="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst256.jpg"><img src="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst256_small.jpg" width="512" height="401"></a>

<a href="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst201.jpg"><img src="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst201_small.jpg" width="512" height="401"></a>

<a href="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst310.jpg"><img src="https://9285c52c-d9b1-40d1-8ac1-e75634aad92d.s3-us-west-2.amazonaws.com/mst310_small.jpg" width="512" height="401"></a>

## Installation

[Python](https://www.python.org/downloads/) 3.6+ is required.

[PyTorch](https://pytorch.org) is required: follow [their installation instructions](https://pytorch.org/get-started/locally/) before proceeding. If you do not have an Nvidia GPU, select None for CUDA. On Linux, you can find out your CUDA version using the `nvidia-smi` command. PyTorch packages for CUDA versions lower than yours will work, but select the highest you can.

To install `style-transfer-pytorch`, first clone the repository, then run the command:

```sh
pip install -e PATH_TO_REPO
```

This will install the `style_transfer` CLI tool. `style_transfer` uses a pre-trained VGG-19 model (Simonyan et al.), which is 548MB in size, and will download it when first run.

If you have a supported GPU and `style_transfer` is using the CPU, try using the argument `--device cuda:0` to force it to try to use the first CUDA GPU. This should print an informative error message.

## Colab

You can try `style_transfer` without installing it locally by using the [official Colab](https://colab.research.google.com/drive/1Tmuwmncao5E3D-5tTIVQjRy2YQ8JdpoB?usp=sharing).

## Basic usage

```sh
style_transfer CONTENT_IMAGE STYLE_IMAGE [STYLE_IMAGE ...] [-o OUTPUT_IMAGE]
```

Input images will be converted to sRGB when loaded, and output images have the sRGB colorspace. If the output image is a TIFF file, it will be written with 16 bits per channel. Alpha channels in the inputs will be ignored.

`style_transfer` has many optional arguments: run it with the `--help` argument to see a full list. Particularly notable ones include:

- `--web` enables a simple web interface while the program is running that allows you to watch its progress. It runs on port 8080 by default, but you can change it with `--port`. If you just want to view the current image and refresh it manually, you can go to `/image`.

- `--devices` manually sets the PyTorch device names. It can be set to `cpu` to force it to run on the CPU on a machine with a supported GPU, or to e.g. `cuda:1` (zero indexed) to select the second CUDA GPU. Two GPUs can be specified, for instance `--devices cuda:0 cuda:1`. `style_transfer` will automatically use the first visible CUDA GPU, falling back to the CPU, if it is omitted.

- `-s` (`--end-scale`) sets the maximum image dimension (height and width) of the output. A large image (e.g. 2896x2172) can take around fifteen minutes to generate on an RTX 3090 and will require nearly all of its 24GB of memory. Since both memory usage and runtime increase linearly in the number of pixels (quadratically in the value of the `--end-scale` parameter), users with less GPU memory or who do not want to wait very long are encouraged to use smaller resolutions. The default is 512.

- `-sw` (`--style-weights`) specifies factors for the weighted average of multiple styles if there is more than one style image specified. These factors are automatically normalized to sum to 1. If omitted, the styles will be blended equally.

- `-cw` (`--content-weight`) sets the degree to which features from the content image are included in the output image. The default is 0.015.

- `-tw` (`--tv-weight`) sets the strength of the smoothness prior. The default is 2.

## References

1. L. Gatys, A. Ecker, M. Bethge (2015), "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)"

1. L. Gatys, A. Ecker, M. Bethge, A. Hertzmann, E. Shechtman (2016), "[Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/abs/1611.07865)"

1. J. Johnson, A. Alahi, L. Fei-Fei (2016), "[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)"

1. A. Mahendran, A. Vedaldi (2014), "[Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035)"

1. D. Kingma, J. Ba (2014), "[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)"

1. K. Simonyan, A. Zisserman (2014), "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)"
