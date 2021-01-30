"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import argparse
import atexit
from dataclasses import asdict
import io
import json
from pathlib import Path
import platform
import sys
import webbrowser

import numpy as np
from PIL import Image, ImageCms
from tifffile import TIFF, TiffWriter
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from . import srgb_profile, StyleTransfer, WebInterface


def prof_to_prof(image, src_prof, dst_prof, **kwargs):
    src_prof = io.BytesIO(src_prof)
    dst_prof = io.BytesIO(dst_prof)
    return ImageCms.profileToProfile(image, src_prof, dst_prof, **kwargs)


def load_image(path, proof_prof=None):
    src_prof = dst_prof = srgb_profile
    try:
        image = Image.open(path)
        if 'icc_profile' in image.info:
            src_prof = image.info['icc_profile']
        else:
            image = image.convert('RGB')
        if proof_prof is None:
            if src_prof == dst_prof:
                return image.convert('RGB')
            return prof_to_prof(image, src_prof, dst_prof, outputMode='RGB')
        proof_prof = Path(proof_prof).read_bytes()
        cmyk = prof_to_prof(image, src_prof, proof_prof, outputMode='CMYK')
        return prof_to_prof(cmyk, proof_prof, dst_prof, outputMode='RGB')
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_pil(path, image):
    try:
        kwargs = {'icc_profile': srgb_profile}
        if path.suffix.lower() in {'.jpg', '.jpeg'}:
            kwargs['quality'] = 95
            kwargs['subsampling'] = 0
        elif path.suffix.lower() == '.webp':
            kwargs['quality'] = 95
        image.save(path, **kwargs)
    except (OSError, ValueError) as err:
        print_error(err)
        sys.exit(1)


def save_tiff(path, image):
    tag = ('InterColorProfile', TIFF.DATATYPES.BYTE, len(srgb_profile), srgb_profile, False)
    try:
        with TiffWriter(path) as writer:
            writer.save(image, photometric='rgb', resolution=(72, 72), extratags=[tag])
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_image(path, image):
    path = Path(path)
    tqdm.write(f'Writing image to {path}.')
    if isinstance(image, Image.Image):
        save_pil(path, image)
    elif isinstance(image, np.ndarray) and path.suffix.lower() in {'.tif', '.tiff'}:
        save_tiff(path, image)
    else:
        raise ValueError('Unsupported combination of image type and extension')


def get_safe_scale(w, h, dim):
    """Given a w x h content image and that a dim x dim square does not
    exceed GPU memory, compute a safe end_scale for that content image."""
    return int(pow(w / h if w > h else h / w, 1/2) * dim)


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
    def __init__(self, st, args, image_type='pil', web_interface=None):
        self.st = st
        self.args = args
        self.image_type = image_type
        self.web_interface = web_interface
        self.iterates = []
        self.progress = None

    def __call__(self, iterate):
        self.iterates.append(asdict(iterate))
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
                save_image(self.args.output, self.st.get_image(self.image_type))
            else:
                if self.web_interface is not None:
                    self.web_interface.put_done()
        elif iterate.i % self.args.save_every == 0:
            save_image(self.args.output, self.st.get_image(self.image_type))

    def close(self):
        if self.progress is not None:
            self.progress.close()

    def get_trace(self):
        return {'args': self.args.__dict__, 'iterates': self.iterates}


def main():
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
    p.add_argument('--devices', type=str, default=[], nargs='+',
                   help='the device names to use (omit for auto)')
    p.add_argument('--random-seed', '-r', type=int, default=0,
                   help='the random seed')
    p.add_argument('--content-weight', '-cw', **arg_info('content_weight'),
                   help='the content weight')
    p.add_argument('--tv-weight', '-tw', **arg_info('tv_weight'),
                   help='the smoothing weight')
    p.add_argument('--min-scale', '-ms', **arg_info('min_scale'),
                   help='the minimum scale (max image dim), in pixels')
    p.add_argument('--end-scale', '-s', type=str, default='512',
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
    p.add_argument('--proof', type=str, default=None,
                   help='the ICC color profile (CMYK) for soft proofing the content and styles')
    p.add_argument('--web', default=False, action='store_true', help='enable the web interface')
    p.add_argument('--host', type=str, default='0.0.0.0',
                   help='the host the web interface binds to')
    p.add_argument('--port', type=int, default=8080,
                   help='the port the web interface binds to')
    p.add_argument('--browser', type=str, default='', nargs='?',
                   help='open a web browser (specify the browser if not system default)')

    args = p.parse_args()

    content_img = load_image(args.content, args.proof)
    style_imgs = [load_image(img, args.proof) for img in args.styles]

    image_type = 'pil'
    if Path(args.output).suffix.lower() in {'.tif', '.tiff'}:
        image_type = 'np_uint16'

    devices = [torch.device(device) for device in args.devices]
    if not devices:
        devices = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')]
    if len(set(device.type for device in devices)) != 1:
        print('Devices must all be the same type.')
        sys.exit(1)
    if not 1 <= len(devices) <= 2:
        print('Only 1 or 2 devices are supported.')
        sys.exit(1)
    print('Using devices:', ' '.join(str(device) for device in devices))

    if devices[0].type == 'cpu':
        print('CPU threads:', torch.get_num_threads())
    if devices[0].type == 'cuda':
        for i, device in enumerate(devices):
            props = torch.cuda.get_device_properties(device)
            print(f'GPU {i} type: {props.name} (compute {props.major}.{props.minor})')
            print(f'GPU {i} RAM:', round(props.total_memory / 1024 / 1024), 'MB')

    end_scale = int(args.end_scale.rstrip('+'))
    if args.end_scale.endswith('+'):
        end_scale = get_safe_scale(*content_img.size, end_scale)
    args.end_scale = end_scale

    web_interface = None
    if args.web:
        web_interface = WebInterface(args.host, args.port)
        atexit.register(web_interface.close)

    for device in devices:
        torch.tensor(0).to(device)
    torch.manual_seed(args.random_seed)

    print('Loading model...')
    st = StyleTransfer(devices=devices, pooling=args.pooling)
    callback = Callback(st, args, image_type=image_type, web_interface=web_interface)
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

    output_image = st.get_image(image_type)
    if output_image is not None:
        save_image(args.output, output_image)
    with open('trace.json', 'w') as fp:
        json.dump(callback.get_trace(), fp, indent=4)


if __name__ == '__main__':
    main()
