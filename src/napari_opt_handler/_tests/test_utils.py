#!/usr/bin/env python

"""
Testing the low level helper functions
TODO: fix binning based on tests
TODO: add test cases, especially edge cases
TODO: functional tests of all corrections in combinations, closer to real life.
"""

import numpy as np
import pytest

from napari_opt_handler.utils import (
    bin_3d,
    img_to_int_type,
    is_positive,
    norm_img,
    rescale_img,
    select_roi,
)

from ..plotting import PlotDialog

__author__ = "David Palecek"
__credits__ = ["Teresa M Correia", "Giorgia Tortora"]
__license__ = "GPL"


@pytest.mark.parametrize(
    "arr_in, ULCorner, height, width, arr_out, roi_out",
    [
        (np.ones((5, 5)), (2, 2), 2, 2, np.ones((2, 2)), (2, 4, 2, 4)),
        (np.ones((5, 5)), (2, 2), 3, 3, np.ones((3, 3)), (2, 5, 2, 5)),
        # ROI beyond the image
        (np.ones((5, 5)), (2, 2), 4, 4, np.ones((3, 3)), (2, 5, 2, 5)),
        # empty ROI
        (np.ones((5, 5)), (2, 2), 0, 0, np.ones((0, 0)), (2, 2, 2, 2)),
    ],
)
def test_select_roi(arr_in, ULCorner, height, width, arr_out, roi_out):
    """
    Test function for the select_roi function.

    Parameters:
    - arr_in: Input array.
    - ULCorner: Upper-left corner coordinates of the region of interest.
    - height: Height of the region of interest.
    - width: Width of the region of interest.
    - arr_out: Expected output array.
    - roi_out: Expected output region of interest parameters.

    Returns:
    - None
    """
    out, roi_pars = select_roi(arr_in, ULCorner, height, width)
    assert roi_pars == roi_out
    np.testing.assert_array_equal(out, arr_out)


# this is failing for negative dimensions,
@pytest.mark.parametrize(
    "arr_in, ULCorner, height, width, out",
    [
        (
            np.ones((5, 5)),
            (2, 2),
            -1,
            -1,
            (ValueError, "Height and width of ROI have to be positive."),
        ),
        (
            np.ones((5, 5)),
            (-1, -1),
            3,
            3,
            (ValueError, "UL corner must be within the array."),
        ),
        (
            np.ones(5),
            (2, 2),
            3,
            3,
            (ValueError, "Array dimension not supported"),
        ),
    ],
)
def test_select_roi_exceptions(arr_in, ULCorner, height, width, out):
    """
    Test function for the select_roi function with invalid inputs.

    This function tests the select_roi function with invalid inputs and
    expects it to raise an error.

    Parameters:
    - arr_in: Input array.
    - ULCorner: Upper left corner coordinates of the region of interest.
    - height: Height of the region of interest.
    - width: Width of the region of interest.
    - out: Tuple containing the expected exception and error message.

    Returns:
    - None
    """
    with pytest.raises(out[0]) as error:
        select_roi(arr_in, ULCorner, height, width)
    assert str(error.value) == out[1]


# this is failing for unmatched dimensions,
@pytest.mark.parametrize(
    "arr_in, bin_factor, expected",
    [
        (np.ones((5, 4, 4)), 2, np.ones((5, 2, 2))),
        (np.ones((5, 4, 4)), 3, np.ones((5, 1, 1))),
        (np.ones((5, 5, 5)), 3, np.ones((5, 1, 1))),
        (np.ones((5, 6, 6)), 3, np.ones((5, 2, 2))),
        (np.ones((5, 6, 6)), 2, np.ones((5, 3, 3))),
        (
            np.ones((5, 6)),
            2,
            (IndexError, "Stack has to have three dimensions."),
        ),
    ],
)
def test_bin_3d(arr_in, bin_factor, expected):
    """
    Test the bin_3d function.

    Parameters:
    - arr_in: The input 3D array to be binned.
    - bin_factor: The factor by which the array should be binned.
    - expected: The expected output after binning.

    Returns:
    None
    """
    # catching the exception
    if type(expected) is tuple:
        with pytest.raises(expected[0], match=expected[1]):
            bin_3d(arr_in, bin_factor)
    else:  # all normal runs
        out = bin_3d(arr_in, bin_factor)
        np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "arr_in, ret_type, expected",
    [
        (2 * np.ones((4, 4)), "float", np.ones((4, 4))),
        (2 * np.ones((4, 4)), "int", np.ones((4, 4))),
        (2.5 * np.ones((4, 4)), "float", np.ones((4, 4))),
    ],
)
def test_norm_img(arr_in, ret_type, expected):
    """
    Test the norm_img function.

    Parameters:
    - arr_in: Input array to be normalized.
    - ret_type: Desired data type of the output array.
    - expected: Expected output array after normalization.

    Returns:
    - None

    Raises:
    - AssertionError: If the output array dtype or values do not match the
        expected values.
    """
    out = norm_img(arr_in, ret_type)
    assert out.dtype == ret_type
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "arr_in, dtype, expected",
    [
        (np.ones((4, 4)), np.uint8, np.ones((4, 4))),
        (np.ones((4, 4)) * 255, np.uint8, np.ones((4, 4)) * 255),
        (np.ones((4, 4)) * 256, np.uint8, np.ones((4, 4)) * 255),
    ],
)
def test_img2intType(arr_in, dtype, expected):
    """
    Test the img_to_int_type function.

    Parameters:
    - arr_in: Input array to be converted.
    - dtype: Desired data type of the output array.
    - expected: Expected output array after conversion.

    Returns:
    - None

    Raises:
    - AssertionError: If the output array dtype or values do not match the
        expected values.
    """
    out = img_to_int_type(arr_in, dtype)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ((np.ones((4, 4)), "dark_corr"), 0),
        ((np.ones((4, 4)) - 2, "dark_corr"), 1),
        ((np.ones((4, 4)) - 2,), 1),
    ],
)
def testIsPositive(inputs, expected):
    """
    Test the is_positive function.

    Parameters:
    - inputs: Tuple containing the input array and an optional string.
    - expected: Expected output after the function call.

    Returns:
    - None

    Raises:
    - AssertionError: If the output does not match the expected value.
    """
    out = is_positive(*inputs)
    assert out == expected


arr1 = np.ones((4, 4)) * 127
arr1[1, 1] = -128
arr1 = arr1.astype(np.int8)

res1 = np.ones((4, 4)) * 255
res1[1, 1] = 0
res1 = res1.astype(np.uint8)

arr2 = np.ones((4, 4)) * 32767
arr2[1, 1] = -32768
arr2 = arr2.astype(np.int16)

res2 = np.ones((4, 4)) * 2**16 - 1
res2[1, 1] = 0
res2 = res2.astype(np.uint16)


# test for rescale_img
@pytest.mark.parametrize(
    "img, dtype, expected",
    [
        (np.ones((4, 4)), np.uint8, (np.ones((4, 4)) * 255).astype(np.uint8)),
        (
            np.ones((4, 4)) * 255,
            np.uint8,
            (np.ones((4, 4)) * 255).astype(np.uint8),
        ),
        (
            np.ones((4, 4)) * 256,
            np.uint8,
            (np.ones((4, 4)) * 255).astype(np.uint8),
        ),
        (np.ones((4, 4)) * 65535, np.uint16, np.ones((4, 4)) * 65535),
        (np.ones((4, 4)) * 65536, np.uint16, np.ones((4, 4)) * 65535),
        (arr1, np.uint8, res1),
        (arr2, np.uint16, res2),
    ],
)
def test_rescale_img(img, dtype, expected):
    """
    Test the rescale_img function.

    Args:
        img: Input array to be rescaled.
        dtype: Desired data type of the output array.
        expected: Expected output array after rescaling.

    Returns:
        None

    Raises:
        AssertionError: If the output array dtype or values do not match the
            expected values.
    """
    out = rescale_img(img, dtype)
    np.testing.assert_array_equal(out, expected)


# test plotting module
def test_PlotDialog(request):
    """
    Test the PlotDialog class.

    Parameters:
    - None

    Returns:
    - None

    Raises:
    - AssertionError: If the plot dialog is not created.
    """
    _, widget = request.getfixturevalue("prepare_widget_data1")
    report = {
        "stack_orig_int": [1, 2, 3, 4, 5],
        "stack_corr_int": [2, 2, 2, 2, 2],
    }
    plot = PlotDialog(widget, report)
    assert plot is not None
    assert plot.canvas.ax1 is not None
    assert plot.canvas.ax1.get_xlabel() == "OPT step number"
    assert plot.canvas.ax1.get_ylabel() == "Intensity"

    handlers = widget.log.handlers[:]
    for hndlr in handlers:
        widget.log.removeHandler(hndlr)
        hndlr.close()
