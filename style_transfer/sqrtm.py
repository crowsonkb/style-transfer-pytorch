"""Matrix square roots with backward passes.

Cleaned up from https://github.com/msubhransu/matrix-sqrt.
"""

import torch


def sqrtm_ns(a, num_iters=10):
    if a.ndim < 2:
        raise RuntimeError('tensor of matrices must have at least 2 dimensions')
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError('tensor must be batches of square matrices')
    if num_iters < 0:
        raise RuntimeError('num_iters must not be negative')
    norm_a = a.pow(2).sum(dim=[-2, -1], keepdim=True).sqrt()
    y = a / norm_a
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype) * 3
    z = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    z = z.repeat([*a.shape[:-2], 1, 1])
    for i in range(num_iters):
        t = (eye - z @ y) / 2
        y = y @ t
        z = t @ z
    return y * norm_a.sqrt()


class _MatrixSquareRootNSLyap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, num_iters, num_iters_backward):
        z = sqrtm_ns(a, num_iters)
        ctx.save_for_backward(z, torch.tensor(num_iters_backward))
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, num_iters = ctx.saved_tensors
        norm_z = z.pow(2).sum(dim=[-2, -1], keepdim=True).sqrt()
        a = z / norm_z
        eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype) * 3
        q = grad_output / norm_z
        for i in range(num_iters):
            eye_a_a = eye - a @ a
            q = q = (q @ eye_a_a - a.transpose(-2, -1) @ (a.transpose(-2, -1) @ q - q @ a)) / 2
            if i < num_iters - 1:
                a = a @ eye_a_a / 2
        return q / 2, None, None


def sqrtm_ns_lyap(a, num_iters=10, num_iters_backward=None):
    if num_iters_backward is None:
        num_iters_backward = num_iters
    if num_iters_backward < 0:
        raise RuntimeError('num_iters_backward must not be negative')
    return _MatrixSquareRootNSLyap.apply(a, num_iters, num_iters_backward)


class _MatrixSquareRootEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    if a.ndim < 2:
        raise RuntimeError('tensor of matrices must have at least 2 dimensions')
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError('tensor must be batches of square matrices')
    return _MatrixSquareRootEig.apply(a)
