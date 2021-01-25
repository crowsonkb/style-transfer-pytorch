"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import argparse
import atexit
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
import webbrowser

from PIL import Image
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from . import StyleTransfer, WebInterface


def load_image(path):
    try:
        return Image.open(path).convert('RGB')
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_image(image, path):
    path = Path(path)
    kwargs = {}
    if path.suffix.lower() in {'.jpg', '.jpeg'}:
        kwargs = {'quality': 95, 'subsampling': 0}
    tqdm.write(f'Writing image to {path}.')
    try:
        image.save(path, **kwargs)
    except (OSError, ValueError) as err:
        print_error(err)
        sys.exit(1)


def setup_exceptions():
    try:
        from IPython.core.ultratb import FormattedTB
        sys.excepthook = FormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


def fix_start_method():
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')


def print_error(err):
    print('\033[31m{}:\033[0m {}'.format(type(err).__name__, err), file=sys.stderr)


class Callback:
    def __init__(self, st, args, start_time, web_interface=None):
        self.st = st
        self.args = args
        self.start_time = start_time
        self.web_interface = web_interface
        self.iterates = []
        self.progress = None

    def __call__(self, iterate):
        self.iterates.append(asdict(iterate))
        self.iterates[-1]['time'] = time.time() - self.start_time
        if iterate.i == 1:
            self.progress = tqdm(total=iterate.i_max, dynamic_ncols=True)
        msg = 'Size: {}x{}, iteration: {}, loss: {:g}'
        tqdm.write(msg.format(iterate.w, iterate.h, iterate.i, iterate.loss))
        self.progress.update()
        if self.web_interface is not None:
            self.web_interface.put_iterate(iterate, self.st.get_image_tensor())
        if iterate.i == iterate.i_max:
            self.progress.close()
            if max(iterate.w, iterate.h) != self.args.end_scale:
                save_image(self.st.get_image(), self.args.output)
            else:
                if self.web_interface is not None:
                    self.web_interface.put_done()
        elif iterate.i % self.args.save_every == 0:
            save_image(self.st.get_image(), self.args.output)

    def close(self):
        if self.progress is not None:
            self.progress.close()

    def get_trace(self):
        return {'args': self.args.__dict__, 'iterates': self.iterates}


def main():
    start_time = time.time()
    setup_exceptions()
    fix_start_method()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def arg_info(arg):
        defaults = StyleTransfer.stylize.__kwdefaults__
        default_types = StyleTransfer.stylize.__annotations__
        return {'default': defaults[arg], 'type': default_types[arg]}

    p.add_argument('content', type=str, help='the content image')
    p.add_argument('styles', type=str, nargs='+', metavar='style', help='the style images')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output image')
    p.add_argument('--style-weights', '-sw', type=float, nargs='+', default=None,
                   metavar='STYLE_WEIGHT', help='the relative weights for each style image')
    p.add_argument('--device', type=str, help='the device name to use (omit for auto)')
    p.add_argument('--random-seed', '-r', type=int, default=0,
                   help='the random seed')
    p.add_argument('--content-weight', '-cw', **arg_info('content_weight'),
                   help='the content weight')
    p.add_argument('--tv-weight', '-tw', **arg_info('tv_weight'),
                   help='the smoothing weight')
    p.add_argument('--min-scale', '-ms', **arg_info('min_scale'),
                   help='the minimum scale (max image dim), in pixels')
    p.add_argument('--end-scale', '-s', **arg_info('end_scale'),
                   help='the final scale (max image dim), in pixels')
    p.add_argument('--iterations', '-i', **arg_info('iterations'),
                   help='the number of iterations per scale')
    p.add_argument('--initial-iterations', '-ii', **arg_info('initial_iterations'),
                   help='the number of iterations on the first scale')
    p.add_argument('--save-every', type=int, default=50,
                   help='save the image every SAVE_EVERY iterations')
    p.add_argument('--step-size', '-ss', **arg_info('step_size'),
                   help='the step size (learning rate)')
    p.add_argument('--avg-decay', '-ad', **arg_info('avg_decay'),
                   help='the EMA decay rate for iterate averaging')
    p.add_argument('--init', **arg_info('init'),
                   choices=['content', 'gray', 'uniform', 'style_mean'],
                   help='the initial image')
    p.add_argument('--style-scale-fac', **arg_info('style_scale_fac'),
                   help='the relative scale of the style to the content')
    p.add_argument('--style-size', **arg_info('style_size'),
                   help='the fixed scale of the style at different content scales')
    p.add_argument('--pooling', type=str, default='max', choices=['max', 'average', 'l2'],
                   help='the model\'s pooling mode')
    p.add_argument('--web', default=False, action='store_true', help='enable the web interface')
    p.add_argument('--host', type=str, default='0.0.0.0',
                   help='the host the web interface binds to')
    p.add_argument('--port', type=int, default=8080,
                   help='the port the web interface binds to')
    p.add_argument('--browser', type=str, default='', nargs='?',
                   help='open a web browser (specify the browser if not system default)')

    args = p.parse_args()

    content_img = load_image(args.content)
    style_imgs = [load_image(img) for img in args.styles]

    web_interface = WebInterface(args.host, args.port) if args.web else None
    if web_interface is not None:
        atexit.register(web_interface.close)

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print('Using device:', device)
    torch.tensor(0).to(device)
    torch.manual_seed(args.random_seed)

    print('Loading model...')
    st = StyleTransfer(device=device, pooling=args.pooling)
    callback = Callback(st, args, start_time, web_interface=web_interface)
    atexit.register(callback.close)

    url = f'http://{args.host}:{args.port}/'
    if args.web:
        if args.browser:
            webbrowser.get(args.browser).open(url)
        elif args.browser is None:
            webbrowser.open(url)

    defaults = StyleTransfer.stylize.__kwdefaults__
    st_kwargs = {k: v for k, v in args.__dict__.items() if k in defaults}
    try:
        st.stylize(content_img, style_imgs, **st_kwargs, callback=callback)
    except KeyboardInterrupt:
        pass

    output_image = st.get_image()
    if output_image is not None:
        save_image(output_image, args.output)
    with open('trace.json', 'w') as fp:
        json.dump(callback.get_trace(), fp, indent=4)


if __name__ == '__main__':
    main()
