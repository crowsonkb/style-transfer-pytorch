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
from numpy.core.fromnumeric import argsort
from tifffile import TIFF, TiffWriter
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from style_transfer import srgb_profile, StyleTransfer, WebInterface, HRNet
from style_transfer.style_transfer_HRNet import *
import os




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
    tag = ('InterColorProfile', TIFF.DATATYPES.BYTE,
           len(srgb_profile), srgb_profile, False)
    try:
        with TiffWriter(path) as writer:
            writer.save(image, photometric='rgb',
                        resolution=(72, 72), extratags=[tag])
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
    print('\033[31m{}:\033[0m {}'.format(
        type(err).__name__, err), file=sys.stderr)


class Callback:
    def __init__(self, st, args, image_type='pil', web_interface=None):
        self.st = st
        self.args = args
        self.image_type = image_type
        self.web_interface = web_interface
        self.iterates = []
        self.progress = None

    def __call__(self, iterate):
        # print(iterate.i)
        self.iterates.append(asdict(iterate))
        if iterate.i == 1:
            self.progress = tqdm(total=iterate.i_max, dynamic_ncols=True)
        msg = 'Size: {}x{}, iteration: {}, loss: {:g}'
        tqdm.write(msg.format(iterate.w, iterate.h, iterate.i, iterate.loss))
        # print(self.progress)
        self.progress.update()
        if self.web_interface is not None:
            self.web_interface.put_iterate(iterate, self.st.get_image_tensor())
        if iterate.i == iterate.i_max:
            self.progress.close()
            if max(iterate.w, iterate.h) != self.args.end_scale:
                save_image(self.args.output,
                           self.st.get_image(self.image_type))
            else:
                if self.web_interface is not None:
                    self.web_interface.put_done()
        elif iterate.i % self.args.save_every == 0:
            save_image(self.args.output, self.st.get_image(
                self.image_type))  # save intermediate results

    def close(self):
        if self.progress is not None:
            self.progress.close()

    def get_trace(self):
        return {'args': self.args.__dict__, 'iterates': self.iterates}

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

loader = transforms.Compose([
    transforms.ToTensor()]) 

def PIL_to_tensor(image, device):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def train(model, st_model,
                content_image, sky_mask, style_images, *,
                style_weights=None,
                content_weight: float = 0.04,
                grad_weight: float = 20,
                sky_weight: float = 1,
                tv_weight: float = 2.,
                min_scale: int = 128,
                end_scale: int = 512,
                iterations: int = 500,
                initial_iterations: int = 1000,
                step_size: float = 0.02,
                avg_decay: float = 0.99,
                init: str = 'content',
                style_scale_fac: float = 1.,
                style_size: int = None,
                callback=None,):
    model.train()
    batch_data = content_image
    min_scale = min(min_scale, end_scale)
    content_weights = [content_weight /
                        len(st_model.content_layers)] * len(st_model.content_layers)

    # style weights among multiple style images
    if style_weights is None:
        style_weights = [1 / len(style_images)] * len(style_images)
    else:
        weight_sum = sum(abs(w) for w in style_weights)
        style_weights = [weight / weight_sum for weight in style_weights]
    if len(style_images) != len(style_weights):
        raise ValueError(
            'style_images and style_weights must have the same length')

    # add TVloss -> the sum of the absolute differences for neighboring pixel-values in the result image
    tv_loss = Scale(LayerApply(TVLoss(), 'input'), tv_weight)

    # get a sequence of scales, from small to large
    scales = gen_scales(min_scale, end_scale)

    # set the initial image and load it to device
    cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)
    if init == 'content':
        st_model.image = TF.to_tensor(
            content_image.resize((cw, ch), Image.LANCZOS))[None]
    elif init == 'gray':
        st_model.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
    elif init == 'uniform':
        st_model.image = torch.rand([1, 3, ch, cw])
    elif init == 'style_mean':
        means = []
        for i, image in enumerate(style_images):
            means.append(TF.to_tensor(image).mean(
                dim=(1, 2)) * style_weights[i])
        st_model.image = torch.rand([1, 3, ch, cw]) / \
            255 + sum(means)[None, :, None, None]
    else:
        raise ValueError(
            "init must be one of 'content', 'gray', 'uniform', 'style_mean'")
    
    # Stylize the image at successively finer scales, each greater by a factor of sqrt(2).
    # This differs from the scheme given in Gatys et al. (2016).
    for scale in scales:
        if st_model.devices[0].type == 'cuda':
            torch.cuda.empty_cache()

        # resize the content image to be smaller than [scale * scale] -> target size
        cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
        content = TF.to_tensor(content_image.resize(
            (cw, ch), Image.LANCZOS))[None]
        content = content.to(st_model.devices[0])

        # resize the mask along with the content iamge
        mask = TF.to_tensor(sky_mask.resize((cw, ch), Image.LANCZOS))[None]
        mask = mask.to(st_model.devices[0])

        grad_loss = Scale(LayerApply(GradientLoss(
            content, mask, sky_weight), 'input'), grad_weight)

        # add ContentLoss
        content_feats = st_model.model(content, layers=st_model.content_layers)
        content_losses = []
        for layer, weight in zip(st_model.content_layers, content_weights):
            target = content_feats[layer]  # target content feature
            # how to calculate content loss?
            content_losses.append(
                Scale(LayerApply(ContentLoss(target), layer), weight))

        style_targets, style_losses = {}, []
        # add StyleLoss
        for i, image in enumerate(style_images):
            # resize the image and load it to GPU
            if style_size is None:
                sw, sh = size_to_fit(
                    image.size, round(scale * style_scale_fac))
            else:
                sw, sh = size_to_fit(image.size, style_size)
            style = TF.to_tensor(image.resize(
                (sw, sh), Image.LANCZOS))[None]
            style = style.to(st_model.devices[0])

            print(f'Processing style image ({sw}x{sh})...')
            style_feats = st_model.model(style, layers=st_model.style_layers)
            # Take the weighted average of multiple style targets (Gram matrices).
            for layer in st_model.style_layers:
                target = StyleLoss.get_target(
                    style_feats[layer]) * style_weights[i]
                if layer not in style_targets:
                    style_targets[layer] = target
                else:
                    style_targets[layer] += target
        for layer, weight in zip(st_model.style_layers, st_model.style_weights):
            target = style_targets[layer]
            style_losses.append(
                Scale(LayerApply(StyleLoss(target), layer), weight))

        # Construct a list of losses
        crit = SumLoss(
            [*content_losses, *style_losses, tv_loss, grad_loss], verbose=False)

        # Warm-start the Adam optimizer if this is not the first scale. (load the previous optimizer state)
        opt2 = optim.Adam([st_model.image], lr=step_size)
        if scale != scales[0]:
            opt_state = scale_adam(opt.state_dict(), (ch, cw))
            opt2.load_state_dict(opt_state)
        opt = opt2

        # empty GPU cache
        if st_model.devices[0].type == 'cuda':
            torch.cuda.empty_cache()

    for batch_count in range(iterations):
        st_model.image = TF.to_tensor(
            batch_data.resize((cw, ch), Image.LANCZOS))[None]
        # the original input
        # interpolate the initial image to the target size
        st_model.image = interpolate(
            st_model.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
        # averaging across the time??
        st_model.average = EMA(st_model.image, avg_decay)
        st_model.image.requires_grad_()

        feats = st_model.model(st_model.image)
        loss = crit(feats).cuda()  # calculate all the losses at the same time
        opt.zero_grad()
        loss.backward()
        opt.step()
        batch_data = st_model.get_image()
        with torch.no_grad():
            st_model.image.clamp_(0, 1)

        # do averaging along time (to be investigated)
        st_model.average.update(st_model.image)
        
        if callback is not None:
                    gpu_ram = 0
                    for device in st_model.devices:
                        if device.type == 'cuda':
                            gpu_ram = max(
                                gpu_ram, torch.cuda.max_memory_allocated(device))
                    callback(STIterate(w=cw, h=ch, i=batch_count+1, i_max=iterations, loss=loss.item(),
                                       time=time.time(), gpu_ram=gpu_ram))

        if ((batch_count+1)%50==0 or (batch_count+1)==iterations):
            print("========Iteration {}/{}========".format(batch_count, iterations))
            checkpoint_path = os.path.join("checkpoint", str(batch_count+1) + ".pth")
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved HRNet checkpoint file at {}".format(checkpoint_path))
            image_type = 'pil'
            sample_img = st_model.get_image(image_type)
            sample_img_path = os.path.join("results", "batch_result/"+str(batch_count+1)+'.jpg')
            save_image(sample_img_path, sample_img)

def main():
    setup_exceptions()
    fix_start_method()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def arg_info(arg):
        defaults = StyleTransfer.stylize.__kwdefaults__
        default_types = StyleTransfer.stylize.__annotations__
        return {'default': defaults[arg], 'type': default_types[arg]}

    content_dir = './images/content/'
    style_dir = './images/styles/'
    my_content_image = 'ust1.jpg'
    my_style_images = ['neon1.jpg']
    output_name = './results/neon1/neon1_ust1_cw0.04_gw20_sw1.5_mask_laplacian.png'

    # files
    p.add_argument('--content', type=str, default=(content_dir +
                   my_content_image), help='the content image')
    p.add_argument('--sky_mask', type=str, default=(content_dir +
                   my_content_image.split('.')[0]+'_skymask.jpg'))
    p.add_argument('--styles', type=str, default=[(style_dir+i)
                   for i in my_style_images], nargs='+', metavar='style', help='the style images')
    p.add_argument('--output', '-o', type=str,
                   default=output_name, help='the output image')

    # training param
    p.add_argument('--style-weights', '-sw', type=float, nargs='+', default=None,
                   metavar='STYLE_WEIGHT', help='the relative weights for each style image')
    p.add_argument('--content-weight', '-cw', **
                   arg_info('content_weight'), help='the content weight')
    p.add_argument('--grad-weight', '-gw', **
                   arg_info('grad_weight'), help='the grad weight')
    p.add_argument('--sky-weight', '-sky', **
                   arg_info('sky_weight'), help='the sky weight')

    p.add_argument('--tv-weight', '-tw', **arg_info('tv_weight'),
                   help='the smoothing weight')
    p.add_argument('--step-size', '-ss', **arg_info('step_size'),
                   help='the step size (learning rate)')
    p.add_argument('--avg-decay', '-ad', **arg_info('avg_decay'),
                   help='the EMA decay rate for iterate averaging')
    p.add_argument('--pooling', type=str, default='average',
                   choices=['max', 'average', 'l2'], help='the model\'s pooling mode')
    p.add_argument('--devices', type=str,
                   default=['cuda:0'], nargs='+', help='the device names to use (omit for auto)')

    p.add_argument('--min-scale', '-ms', **arg_info('min_scale'),
                   help='the minimum scale (max image dim), in pixels')
    p.add_argument('--end-scale', '-s', type=str, default='512',
                   help='the final scale (max image dim), in pixels')

    p.add_argument('--random-seed', '-r', type=int,
                   default=0, help='the random seed')
    p.add_argument('--iterations', '-i', **arg_info('iterations'),
                   help='the number of iterations per scale')
    p.add_argument('--initial-iterations', '-ii', **arg_info('initial_iterations'),
                   help='the number of iterations on the first scale')
    p.add_argument('--save-every', type=int, default=50,
                   help='save the image every SAVE_EVERY iterations')

    p.add_argument('--init', **arg_info('init'),
                   choices=['content', 'gray', 'uniform', 'style_mean'], help='the initial image')
    p.add_argument('--style-scale-fac', **arg_info('style_scale_fac'),
                   help='the relative scale of the style to the content')
    p.add_argument('--style-size', **arg_info('style_size'),
                   help='the fixed scale of the style at different content scales')

    p.add_argument('--proof', type=str, default=None,
                   help='the ICC color profile (CMYK) for soft proofing the content and styles')
    p.add_argument('--web', default=False, action='store_true',
                   help='enable the web interface')
    p.add_argument('--host', type=str, default='0.0.0.0',
                   help='the host the web interface binds to')
    p.add_argument('--port', type=int, default=8080,
                   help='the port the web interface binds to')
    p.add_argument('--browser', type=str, default='', nargs='?',
                   help='open a web browser (specify the browser if not system default)')

    args = p.parse_args()

    # load images
    content_img = load_image(args.content, args.proof)
    sky_mask = load_image(args.sky_mask, args.proof)
    style_imgs = [load_image(img, args.proof) for img in args.styles]
    image_type = 'pil'
    if Path(args.output).suffix.lower() in {'.tif', '.tiff'}:
        image_type = 'np_uint16'

    # find device
    devices = [torch.device(device) for device in args.devices]
    if not devices:
        devices = [torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')]
    if len(set(device.type for device in devices)) != 1:
        print('Devices must all be the same type.')
        sys.exit(1)
    if not 1 <= len(devices) <= 2:
        print('Only 1 or 2 devices are supported.')
        sys.exit(1)
    print('Using devices:', ' '.join(str(device) for device in devices))

    # print device information
    if devices[0].type == 'cpu':
        print('CPU threads:', torch.get_num_threads())
    if devices[0].type == 'cuda':
        for i, device in enumerate(devices):
            props = torch.cuda.get_device_properties(device)
            print(
                f'GPU {i} type: {props.name} (compute {props.major}.{props.minor})')
            print(f'GPU {i} RAM:', round(
                props.total_memory / 1024 / 1024), 'MB')

    # verify end scale
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

    # load the model
    st = StyleTransfer(devices=devices, pooling=args.pooling)
    feedforward_net = HRNet.HRNet().cuda()

    callback = Callback(st, args, image_type=image_type,
                        web_interface=web_interface)
    atexit.register(callback.close)

    # setup online monitor
    url = f'http://{args.host}:{args.port}/'
    if args.web:
        if args.browser:
            webbrowser.get(args.browser).open(url)
        elif args.browser is None:
            webbrowser.open(url)

    # do style transfer
    # get the default keyword dictionary
    defaults = StyleTransfer.stylize.__kwdefaults__
    # find modified args and put them into an array
    st_kwargs = {k: v for k, v in args.__dict__.items() if k in defaults}
    try:
        train(feedforward_net, st, content_img, sky_mask, style_imgs, **
                   st_kwargs, callback=callback)  # training
    except KeyboardInterrupt:
        pass

    # get the result image
    output_image = st.get_image(image_type)
    if output_image is not None:
        save_image(args.output, output_image)
    with open('trace.json', 'w') as fp:
        json.dump(callback.get_trace(), fp, indent=4)


if __name__ == '__main__':
    main()
