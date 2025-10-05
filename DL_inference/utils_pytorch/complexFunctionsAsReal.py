

"""
@author: spopoff
"""

import torch
from torch.nn.functional import (
    avg_pool2d,
    dropout,
    dropout2d,
    interpolate,
    max_pool2d,
    relu,
    leaky_relu,
    sigmoid,
    tanh,
)


from torch.nn.functional import max_pool2d, max_pool3d, avg_pool2d, dropout, dropout2d, interpolate
from torch import tanh, relu, sigmoid


def complex_matmul(A, B):
    """
    Performs the matrix product between two complex matrices
    """

    outp_real = torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
    outp_imag = torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real)

    return torch.complex(outp_real, outp_imag)


def complex_avg_pool2d(inp, *args, **kwargs):
    """
    Perform complex average pooling.
    """
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)

    return torch.complex(absolute_value_real, absolute_value_imag)


def complex_normalize(inp):
    """
    Perform complex normalization
    """
    real_value, imag_value = inp.real, inp.imag
    real_norm = (real_value - real_value.mean()) / real_value.std()
    imag_norm = (imag_value - imag_value.mean()) / imag_value.std()
    return torch.complex(real_norm, imag_norm)


def complex_relu(inp):
    return torch.complex(relu(inp.real), relu(inp.imag))


def complex_leaky_relu(inp):
    return torch.complex(leaky_relu(inp.real), leaky_relu(inp.imag))


def complex_sigmoid(inp):
    return torch.complex(sigmoid(inp.real), sigmoid(inp.imag))


def complex_tanh(inp):
    return torch.complex(tanh(inp.real), tanh(inp.imag))


def complex_opposite(inp):
    return -inp.real.type(torch.complex64) + 1j * (-inp.imag.type(torch.complex64))


def complex_stack(inp, dim):
    inp_real = [x.real for x in inp]
    inp_imag = [x.imag for x in inp]
    return torch.stack(inp_real, dim).type(torch.complex64) + 1j * torch.stack(
        inp_imag, dim
    ).type(torch.complex64)


def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output

def _retrieve_elements_from_indices_3d(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-3)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-3)
    ).view_as(indices)
    return output


def complex_upsample(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the real and imaginary part and recombining
    """
    outp_real = interpolate(
        inp.real,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    outp_imag = interpolate(
        inp.imag,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return torch.complex(outp_real, outp_imag)


def complex_upsample2(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the amplitude and phase part and recombining
    """
    outp_abs = interpolate(
        inp.abs(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    angle = torch.atan2(inp.imag, inp.real)
    outp_angle = interpolate(
        angle,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return torch.complex(outp_abs * torch.cos(outp_angle), outp_abs * torch.sin(outp_angle))


def complex_max_pool2d(
    inp,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    Perform complex max pooling by selecting on the absolute value on the complex values.
    """
    absolute_value, indices = max_pool2d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresponding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(inp.imag, inp.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (
        torch.cos(angle).type(torch.complex64)
        + 1j * torch.sin(angle).type(torch.complex64)
    )


def complex_dropout(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complex_dropout2d(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complex_max_pool3d(
    inp,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    Perform complex max pooling in 3D by selecting on the absolute value of the complex values.
    """
    # Compute absolute values and perform max pooling on them
    absolute_value, indices = max_pool3d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    
    # Adjust the absolute values to complex type
    # absolute_value = absolute_value.type(torch.complex64)

    # Compute the phase values (angle)
    angle = torch.atan2(inp.imag, inp.real)

    # Retrieve the phase values corresponding to the max pooled indices
    angle = _retrieve_elements_from_indices_3d(angle, indices)

    # Reconstruct the complex numbers using the pooled magnitudes and the selected phase angles
    pooled_complex = torch.complex(absolute_value * torch.cos(angle), absolute_value * torch.sin(angle))

    return pooled_complex