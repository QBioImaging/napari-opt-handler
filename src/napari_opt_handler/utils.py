#!/usr/bin/env python

import numpy as np
import warnings

"""
Utility functions for stack operations
- currently used for napari widget and in the Correct class
TODO: perform timing on real data and consider threading.
"""


def select_roi(stack: np.ndarray,
               ul_corner: tuple, height: int,
               width: int) -> tuple[np.ndarray, tuple]:
    """
    Select Region of Interest (ROI) relative to the Upper Left (UL) corner
    point. If points layer contains more than one point, the last point
    is considered.

    Args:
        stack (np.ndarray): Stack of images, first dimension is the
            stacking dimension
        ul_corner (tuple): tuple of x_coord, y_coord representing the upper
            left corner of the ROI
        height (int): Height of the ROI
        width (int): Width of the ROI

    Raises:
        ValueError: If ul_corner is not a 2-tuple,
            or if the array is not 2D or 3D,
            or if the height and width of the ROI are not positive,
            or if the UL corner is not within the array.

    Returns:
        tuple[np.ndarray, tuple]: A tuple containing the ROI's image data
            from the stack and a tuple of the ROI's parameters (x1, x2, y1, y2)
    """

    # TODO: the errors on the napari should handled on the widget level
    # confirm that height and width are positive
    if height < 0 or width < 0:
        raise ValueError('Height and width of ROI have to be positive.')

    try:
        # x, y assumed to be the last two elements of the tuple
        x1, y1 = [int(k) for k in ul_corner[-2:]]
    except ValueError:
        raise ValueError(
            'UL corner must be defined by tuple of (x_coord, y_coord).')

    # ensure that x1 and y1 are within the array
    if x1 < 0 or y1 < 0:
        raise ValueError('UL corner must be within the array.')

    # ensure that array is 2D or 3D
    if stack.ndim not in [2, 3]:
        raise ValueError("Array dimension not supported")

    # ensure that limits of the arrays are respected
    x2 = min(x1 + height, stack.shape[-2])
    y2 = min(y1 + width, stack.shape[-1])
    # takes care of ROI beyond the image size
    if stack.ndim == 2:
        roi = stack[x1: x2, y1: y2]

    elif stack.ndim == 3:
        roi = stack[:, x1: x2, y1: y2]

    roi_pars = (x1, x2, y1, y2)
    return roi, roi_pars


# TODO: should there be sum method allowed? For small signals makes sense.
# TODO: rois work for 2D arrays, binning should too.
def bin_3d(stack: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Bin stack of images applying mean on the binned pixels.
    First dim is the stacking one. Binning is along axis 1, 2.
    If the stack is not divisible by bin_factor, the last pixels are
    discarded.
    Result is casted on integer.

    Args:
        stack (np.ndarray): stack of images
        bin_factor (int): how many pixels are binned along x and y.
            Binning is the same along x and y.

    Raises:
        IndexError: Stack is not 3 dimensional.

    Returns:
        np.ndarray: Binned stack of images.
    """
    if stack.ndim != 3:
        raise IndexError('Stack has to have three dimensions.')

    # array preallocation
    height_dim = stack.shape[1] // bin_factor
    width_dim = stack.shape[2] // bin_factor
    ans = np.empty((stack.shape[0], height_dim, width_dim),
                   dtype=int,
                   )

    # subarray of stack that equals height_dim * bin_factor
    if (height_dim * bin_factor != stack.shape[1] or
            width_dim * bin_factor != stack.shape[2]):

        stack = stack[:, :height_dim * bin_factor, :width_dim * bin_factor]

    # binning is done along axis 1 and 2, mean is applied
    for i in range(stack.shape[0]):
        ans[i] = stack[i].reshape(height_dim, bin_factor,
                                  width_dim, bin_factor).mean(3).mean(1)

    return ans


######################
# Funcs used in Correct class
######################
def norm_img(img: np.array, ret_type='float') -> np.array:
    """
    Normalize np.array image to 1.

    Args:
        img (np.array): img to normalize
        ret_type (str, optional): result can be casted to any valid dtype.
            Defaults to 'float'.

    Returns:
        np.array: normalized array to 1
    """
    return img/np.amax(img) if ret_type == 'float' else (
        img/np.amax(img)).astype(ret_type)


def img_to_int_type(img: np.array, dtype: np.dtype = np.int_) -> np.array:
    """
    After corrections, resulting array can be dtype float. Two steps are
    taken here. First convert to a chosed dtype and then clip values as if it
    was unsigned int, which the images are.

    Args:
        img (np.array): img to convert
        dtype (np.dtype): either np.int8 or np.int16 currently,
            Defaults to np.int_

    Returns:
        np.array: array of type int
    """
    # TODO: take care of 12 bit images, how to identify them in order
    # to normalize on 2**12-1 but witll on 16bit. Memory saving in practice
    if dtype == np.uint8:
        ans = np.clip(img, 0, 255).astype(dtype)
    elif dtype == np.uint16:
        # 4095 would be better for our 12bit camera
        ans = np.clip(img, 0, 2**16 - 1).astype(dtype)
    else:
        ans = np.clip(img, 0, np.amax(img)).astype(np.int_)

    return ans


def is_positive(img, corr_type='Unknown'):
    if np.any(img < 0):
        warnings.warn(
            f'{corr_type} correction: Some pixel < 0, casting them to 0.',
            )
        # return for testing purposes, can be better?
        return 1
    return 0


# function to rescale image to dtype range, for instance 0-255 for uint8
def rescale_img(img, dtype):
    """
    Rescale image to the range of the dtype.

    Args:
        img (np.array): img to rescale
        dtype (np.dtype): dtype to rescale to

    Returns:
        np.array: rescaled img
    """
    if dtype == np.uint8:
        if np.amax(img) == np.amin(img):
            return np.ones(img.shape, dtype=dtype) * 255
        # shift by min value and scale to 0-255
        return ((img - np.amin(img)) /
                (np.amax(img) - np.amin(img)) * 255).astype(dtype)

    elif dtype == np.uint16:
        if np.amax(img) == np.amin(img):
            return np.ones(img.shape, dtype=dtype) * (2**16-1)
        # shift by min value and scale to 0-2**16
        return ((img - np.amin(img)) /
                (np.amax(img) - np.amin(img)) * (2**16-1)).astype(dtype)
    else:
        return img
