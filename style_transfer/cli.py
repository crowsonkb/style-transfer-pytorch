"""Neural style transfer in PyTorch. Implements A Neural Algorithm of Artistic
Style (http://arxiv.org/abs/1508.06576)."""

import argparse
import sys

from PIL import Image
import torch

from . import StyleTransfer


def load_image(path):
    return Image.open(path).convert('RGB')


def setup_exceptions():
    try:
        from IPython.core.ultratb import AutoFormattedTB
        sys.excepthook = AutoFormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


def main():
    setup_exceptions()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def arg_info(arg):
        defaults = StyleTransfer.stylize.__kwdefaults__
        default_types = StyleTransfer.stylize.__annotations__
        return {'default': defaults[arg], 'type': default_types[arg]}

    p.add_argument('content', type=str, help='the content image')
    p.add_argument('style', type=str, help='the style image')
    p.add_argument('output', type=str, nargs='?', default='out.png',
                   help='the output image')
    p.add_argument('--device', type=str, help='the device name to use (omit for auto)')
    p.add_argument('--content-weight', '-cw', **arg_info('content_weight'),
                   help='the content weight')
    p.add_argument('--tv-weight', '-tw', **arg_info('tv_weight'),
                   help='the smoothing weight')
    p.add_argument('--initial-scale', '-is', **arg_info('initial_scale'),
                   help='the initial scale, in pixels')
    p.add_argument('--scales', '-s', **arg_info('scales'),
                   help='the number of scales')
    p.add_argument('--iterations', '-i', **arg_info('iterations'),
                   help='the number of iterations per scale')
    p.add_argument('--step-size', '-ss', **arg_info('step_size'),
                   help='the step size (learning rate)')
    args = p.parse_args()

    content_img = load_image(args.content)
    style_img = load_image(args.style)

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('Using device:', device)
    torch.tensor(0).to(device)

    print('Loading model...')
    st = StyleTransfer(device=device)
    defaults = StyleTransfer.stylize.__kwdefaults__
    st_kwargs = {k: v for k, v in args.__dict__.items() if k in defaults}

    try:
        st.stylize(content_img, style_img, **st_kwargs)
    except KeyboardInterrupt:
        pass

    output_image = st.get_image()
    if output_image is not None:
        print(f'Writing image to {args.output}.')
        output_image.save(args.output)


if __name__ == '__main__':
    main()
