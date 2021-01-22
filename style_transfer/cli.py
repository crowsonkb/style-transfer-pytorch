"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import argparse
import sys

from PIL import Image
import torch
from tqdm import tqdm

from . import StyleTransfer


def load_image(path):
    try:
        return Image.open(path).convert('RGB')
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_image(image, path):
    tqdm.write(f'Writing image to {path}.')
    try:
        image.save(path)
    except (OSError, ValueError) as err:
        print_error(err)
        sys.exit(1)


def setup_exceptions():
    try:
        from IPython.core.ultratb import FormattedTB
        sys.excepthook = FormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


def print_error(err):
    print('\033[31m{}:\033[0m {}'.format(type(err).__name__, err), file=sys.stderr)


def main():
    setup_exceptions()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def arg_info(arg):
        defaults = StyleTransfer.stylize.__kwdefaults__
        default_types = StyleTransfer.stylize.__annotations__
        return {'default': defaults[arg], 'type': default_types[arg]}

    p.add_argument('content', type=str, help='the content image')
    p.add_argument('styles', type=str, nargs='+', help='the style images')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output image')
    p.add_argument('--style-img-weights', type=float, nargs='+', default=None,
                   help='the relative weights for each style image')
    p.add_argument('--device', type=str, help='the device name to use (omit for auto)')
    p.add_argument('--random-seed', '-r', type=int, default=0,
                   help='the random seed')
    p.add_argument('--content-weight', '-cw', **arg_info('content_weight'),
                   help='the content weight')
    p.add_argument('--tv-weight-1', '-tw1', **arg_info('tv_weight_1'),
                   help='the L1 smoothing weight')
    p.add_argument('--tv-weight-2', '-tw', '-tw2', **arg_info('tv_weight_2'),
                   help='the L2 smoothing weight')
    p.add_argument('--min-scale', '-ms', **arg_info('min_scale'),
                   help='the minimum scale (max image dim), in pixels')
    p.add_argument('--end-scale', '-s', **arg_info('end_scale'),
                   help='the final scale (max image dim), in pixels')
    p.add_argument('--iterations', '-i', **arg_info('iterations'),
                   help='the number of iterations per scale')
    p.add_argument('--initial-iterations', '-ii', **arg_info('initial_iterations'),
                   help='the number of iterations on the first scale')
    p.add_argument('--step-size', '-ss', **arg_info('step_size'),
                   help='the step size (learning rate)')
    p.add_argument('--avg-decay', '-ad', **arg_info('avg_decay'),
                   help='the EMA decay rate for iterate averaging')
    p.add_argument('--init', **arg_info('init'), choices=['content', 'gray', 'random'],
                   help='the initial image')
    p.add_argument('--style-scale-fac', **arg_info('style_scale_fac'),
                   help='the relative scale of the style to the content')
    p.add_argument('--style-size', **arg_info('style_size'),
                   help='the fixed scale of the style at different content scales')
    p.add_argument('--pooling', type=str, default='max', choices=['max', 'average', 'l2'],
                   help='the model\'s pooling mode')
    p.add_argument('--mono', default=False, action='store_true',
                   help='use monochrome mode')

    args = p.parse_args()

    content_img = load_image(args.content)
    style_imgs = [load_image(img) for img in args.styles]

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('Using device:', device)
    torch.tensor(0).to(device)
    torch.manual_seed(args.random_seed)

    print('Loading model...')
    st = StyleTransfer(device=device, pooling=args.pooling)
    defaults = StyleTransfer.stylize.__kwdefaults__
    st_kwargs = {k: v for k, v in args.__dict__.items() if k in defaults}

    progress = None

    def callback(iterate):
        nonlocal progress
        if iterate.i == 1:
            progress = tqdm(total=iterate.i_max, dynamic_ncols=True)
        tqdm.write(f'{iterate.w}x{iterate.h} {iterate.i} {iterate.loss:g}')
        progress.update()
        if iterate.i == iterate.i_max:
            progress.close()
            if max(iterate.w, iterate.h) != args.end_scale:
                save_image(st.get_image(), args.output)
        elif iterate.i % 50 == 0:
            save_image(st.get_image(), args.output)

    try:
        st.stylize(content_img, style_imgs, **st_kwargs, callback=callback)
    except KeyboardInterrupt:
        pass
    finally:
        if progress is not None:
            progress.close()

    output_image = st.get_image()
    if output_image is not None:
        save_image(output_image, args.output)


if __name__ == '__main__':
    main()
