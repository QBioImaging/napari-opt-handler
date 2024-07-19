#!/usr/bin/env python

"""
Tests for correction class
"""

import numpy as np
import pytest

from napari_opt_handler.corrections import Correct

__author__ = "David Palecek"
__credits__ = ["Teresa M Correia", "Giorgia Tortora"]
__license__ = "GPL"

hot1 = dark1 = bright1 = np.ones((5, 5))
# define hot one as a copy of hot1 with one pixel changed
hot2 = hot1.copy()
hot2[2, 1] = 10

hot3 = np.ones((5, 5)) * 10
hot3[1, 2] = 1
hot3[2, 1] = 30

# define a test image
img1 = np.ones((5, 5)) * 5
img2 = img1.copy()
img2[2, 1] = 1

img_bad = np.ones((5, 5)) * 5
img_bad[1, 2] = 2  # dead pixel
img_bad[2, 1] = 15  # hot pixel

img_bad_hot_corr_n4 = img_bad.copy()
img_bad_hot_corr_n4[2, 1] = 5

img_bad_hot_corr_n8 = img_bad.copy()
img_bad_hot_corr_n8[2, 1] = 4

img_bad_dead_corr_n4 = img_bad.copy()
img_bad_dead_corr_n4[1, 2] = 5

img_bad_dead_corr_n8 = img_bad.copy()
img_bad_dead_corr_n8[1, 2] = 6

img3 = np.ones((5, 5)) * 3
img4 = np.ones((5, 5)) * 4

# for int correction testing
bright2 = np.ones((5, 5)) * 10
stack1 = np.ones((5, 5, 5)) * 5

return_stack1 = np.ones((5, 5, 5)) * 10

stack2 = np.ones((5, 5, 5)) * 5
stack2[0] = np.ones((5, 5)) * 10

fl_bleach_corr2 = np.ones((5, 5)) * 5
fl_bleach_corr2[0] = 10

stack3 = np.ones((6, 4, 3)) * 5
stack3[0] = np.ones((4, 3)) * 10
stack3[1] = np.ones((4, 3)) * 7

fl_bleach_corr3 = np.ones((6, 4)) * 5
fl_bleach_corr3[0] = 10
fl_bleach_corr3[1] = 7


@pytest.mark.parametrize(
    "init_vals, expected",
    [
        # no xorrections
        ((None,), (None, 7, None, None)),
        ((hot1,), (None, 7, None, None)),
        # with five the cutoff is 10.17
        (
            (
                hot2,
                4.85,
            ),
            (None, 4.85, None, None),
        ),
        (
            (
                hot2,
                -4.85,
            ),
            (None, 7.0, None, None),
        ),
        (
            (
                hot2,
                0,
            ),
            (None, 7.0, None, None),
        ),
    ],
)
def test_init(init_vals, expected):
    corr = Correct(*init_vals)
    assert corr.hot_pxs == expected[0]
    assert corr.std_mult == expected[1]
    assert corr.dark is expected[2]
    assert corr.bright_corr is expected[3]


@pytest.mark.parametrize(
    "init_vals, expected",
    [
        ((), (None, 7, None, None)),  # no xorrections
        ((None,), (None, 7, None, None)),
        ((hot1,), ([], 7, None, None)),
        # with five the cutoff is 10.17
        ((hot2, 5), ([], 5, None, None)),
        ((hot2, 4.85), ([(2, 1)], 4.85, None, None)),
    ],
)
def test_get_bad_pxs(init_vals, expected):
    corr = Correct(*init_vals)
    if expected[0] is not None:
        corr.get_bad_pxs()
        assert corr.hot_pxs == expected[0]
        assert corr.std_mult == expected[1]
        assert corr.dark is expected[2]
        assert corr.bright_corr is expected[3]
    else:
        with pytest.raises(ValueError, match="No bad pixel array provided"):
            corr.get_bad_pxs()


@pytest.mark.parametrize(
    "init_vals, mode, expected",
    [
        ((), "hot", (None, 7, None, None)),  # no corrections
        ((), "dead", (None, 7, None, None)),  # no corrections
        ((), "both", (None, 7, None, None)),  # no corrections
        ((), "", (None, 7, None, None)),  # no corrections
        ((None,), "hot", (None, 7, None, None)),
        ((None,), "dead", (None, 7, None, None)),
        ((None,), "both", (None, 7, None, None)),
        ((hot1,), "hot", ([], [], 7, None, None)),
        ((hot1,), "dead", ([], [], 7, None, None)),
        ((hot1,), "both", ([], [], 7, None, None)),
        ((hot1,), "", ([], [], 7, None, None)),
        # with five the cutoff is 10.17
        ((hot2, 5), "hot", ([], [], 5, None, None)),
        ((hot2, 5), "dead", ([], [], 5, None, None)),
        ((hot2, 5), "both", ([], [], 5, None, None)),
        ((hot2, 5), "", ([], [], 5, None, None)),
        ((hot2, 4.85), "hot", ([(2, 1)], [], 4.85, None, None)),
        ((hot2, 4.85), "dead", ([], [], 4.85, None, None)),
        ((hot2, 4.85), "both", ([(2, 1)], [], 4.85, None, None)),
        ((hot2, 4.85), "", ([(2, 1)], [], 4.85, None, None)),
        ((hot3, 4), "hot", ([(2, 1)], [], 4, None, None)),
        ((hot3, 2), "dead", ([], [(1, 2)], 2, None, None)),
        ((hot3, 2), "both", ([(2, 1)], [(1, 2)], 2, None, None)),
        ((hot3, 3), "both", ([(2, 1)], [], 3, None, None)),
        ((hot3, 4.85), "", ([], [], 4.85, None, None)),
    ],
)
def test_get_bad_pxs2(init_vals, mode, expected):
    corr = Correct(*init_vals)
    if expected[0] is None:
        with pytest.raises(ValueError, match="No bad pixel array provided"):
            corr.get_bad_pxs(mode=mode)
    elif mode not in ["hot", "dead", "both"]:
        with pytest.raises(
            ValueError,
            match="Unknown mode option, valid is hot, dead and both.",
        ):
            corr.get_bad_pxs(mode=mode)
    elif expected[0] is not None:
        corr.get_bad_pxs(mode=mode)
        assert corr.hot_pxs == expected[0]
        assert corr.dead_pxs == expected[1]
        assert corr.std_mult == expected[2]
        assert corr.dark is expected[3]
        assert corr.bright_corr is expected[4]
    else:
        assert 1 == 0


@pytest.mark.parametrize(
    "inits, img, expected",
    [
        ((hot1,), img1, img1),
        ((hot2, 5), img1, img1),
        ((hot2, 4), img1, img1),
        ((hot2, 4), img2, img1),
    ],
)
def test_correct_hot(inits, img, expected):
    corr = Correct(*inits)
    corr.get_bad_pxs()
    corrected = corr.correctBadPxs(img)
    assert np.array_equal(corrected, expected)


# testing n4, n8 methods, this data has only 1 hot pixel
@pytest.mark.parametrize(
    "inits, mode_corr, neigh, img, expected",
    [
        ((hot1,), "hot", "n4", img1, (img1, [], [])),
        ((hot2, 5), "hot", "n4", img1, (img1, [], [])),
        ((hot2, 4), "hot", "n4", img1, (img1, [(2, 1)], [])),
        ((hot2, 4), "hot", "n4", img2, (img1, [(2, 1)], [])),
        ((hot1,), "dead", "n4", img1, (img1, [], [])),
        ((hot2, 5), "dead", "n4", img1, (img1, [], [])),
        ((hot2, 4), "dead", "n4", img1, (img1, [], [])),
        ((hot2, 4), "dead", "n4", img2, (img2, [], [])),
        ((hot1,), "both", "n4", img1, (img1, [], [])),
        ((hot2, 5), "both", "n4", img1, (img1, [], [])),
        ((hot2, 4), "both", "n4", img1, (img1, [(2, 1)], [])),
        ((hot2, 4), "both", "n4", img2, (img1, [(2, 1)], [])),
        ((hot1,), "hot", "n8", img1, (img1, [], [])),
        ((hot2, 5), "hot", "n8", img1, (img1, [], [])),
        ((hot2, 4), "hot", "n8", img1, (img1, [(2, 1)], [])),
        ((hot2, 4), "hot", "n8", img2, (img1, [(2, 1)], [])),
        ((hot1,), "dead", "n8", img1, (img1, [], [])),
        ((hot2, 5), "dead", "n8", img1, (img1, [], [])),
        ((hot2, 4), "dead", "n8", img1, (img1, [], [])),
        ((hot2, 4), "dead", "n8", img2, (img2, [], [])),
        ((hot1,), "both", "n8", img1, (img1, [], [])),
        ((hot2, 5), "both", "n8", img1, (img1, [], [])),
        ((hot2, 4), "both", "n8", img1, (img1, [(2, 1)], [])),
        ((hot2, 4), "both", "n8", img2, (img1, [(2, 1)], [])),
    ],
)
def test_correct_hot2(inits, mode_corr, neigh, img, expected):
    corr = Correct(*inits)
    corr.get_bad_pxs(mode=mode_corr)
    assert corr.hot_pxs == expected[1]
    assert corr.dead_pxs == expected[2]
    corrected = corr.correctBadPxs(img, mode=neigh)
    assert np.array_equal(corrected, expected[0])
    assert corr.badPxs == set(expected[1] + expected[2])


# testing n4, n8 methods, this data has only 1 hot pixel
# for hot3, cutoff for hot is 4, for dead is 2
@pytest.mark.parametrize(
    "inits, mode_corr, neigh, img, expected",
    [
        ((hot3,), "hot", "n4", img1, (img1, [], [])),
        ((hot3, 5), "hot", "n4", img_bad, (img_bad, [], [])),
        ((hot3, 4), "hot", "n4", img_bad, (img_bad_hot_corr_n4, [(2, 1)], [])),
        ((hot3,), "dead", "n4", img1, (img1, [], [])),
        ((hot3, 3), "dead", "n4", img_bad, (img_bad, [], [])),
        (
            (hot3, 2),
            "dead",
            "n4",
            img_bad,
            (img_bad_dead_corr_n4, [], [(1, 2)]),
        ),
        ((hot3,), "both", "n4", img1, (img1, [], [])),
        (
            (hot3, 4),
            "both",
            "n4",
            img_bad,
            (img_bad_hot_corr_n4, [(2, 1)], []),
        ),
        ((hot3, 2), "both", "n4", img_bad, (img1, [(2, 1)], [(1, 2)])),
        ((hot3,), "hot", "n8", img1, (img1, [], [])),
        ((hot3, 5), "hot", "n8", img_bad, (img_bad, [], [])),
        ((hot3, 4), "hot", "n8", img_bad, (img_bad_hot_corr_n8, [(2, 1)], [])),
        ((hot3,), "dead", "n8", img1, (img1, [], [])),
        ((hot3, 3), "dead", "n8", img_bad, (img_bad, [], [])),
        (
            (hot3, 2),
            "dead",
            "n8",
            img_bad,
            (img_bad_dead_corr_n8, [], [(1, 2)]),
        ),
        ((hot3,), "both", "n8", img1, (img1, [], [])),
        (
            (hot3, 4),
            "both",
            "n8",
            img_bad,
            (img_bad_hot_corr_n8, [(2, 1)], []),
        ),
        ((hot3, 2), "both", "n8", img_bad, (img1, [(2, 1)], [(1, 2)])),
    ],
)
def test_correct_hot3(inits, mode_corr, neigh, img, expected):
    corr = Correct(*inits)
    corr.get_bad_pxs(mode=mode_corr)
    assert corr.hot_pxs == expected[1]
    assert corr.dead_pxs == expected[2]
    corrected = corr.correctBadPxs(img, mode=neigh)
    assert np.array_equal(corrected, expected[0])
    assert corr.badPxs == set(expected[1] + expected[2])


###########################
# correct dark and bright #
###########################
# corr_inputs are, stack, useDark, useBright, modality
@pytest.mark.parametrize(
    "inits, corr_inputs, expected",
    [
        ((hot2, 4, dark1, bright2), (stack1,), np.ones((5, 5, 5)) * 4),
        # explicit inputs
        (
            (hot2, 4, dark1, bright2),
            (stack1, True, False, "Transmission"),
            np.ones((5, 5, 5)) * 4,
        ),
        (
            (hot2, 4, dark1, bright2),
            (stack1, True, False, "Emission"),
            np.ones((5, 5, 5)) * 4,
        ),
        # no dark, should return a copy
        (
            (hot2, 4, dark1, bright2),
            (stack1, False, False, "Transmission"),
            stack1,
        ),
        (
            (hot2, 4, dark1, bright2),
            (stack1, False, False, "Emission"),
            stack1,
        ),
        # now if bright is used
        (
            (hot2, 4, dark1, bright2),
            (stack1, True, True, "Transmission"),
            (np.ones((5, 5, 5)) * 4 / 9 * (2**16 - 1)).astype(np.uint16),
        ),
        (
            (hot2, 4, dark1, bright2),
            (stack1, True, True, "Emission"),
            (np.zeros((5, 5, 5))).astype(np.uint16),
        ),
        # no dark
        (
            (hot2, 4, dark1, bright2),
            (stack1, False, True, "Transmission"),
            (np.ones((5, 5, 5)) * 5 / 10 * (2**16 - 1)).astype(np.uint16),
        ),
        (
            (hot2, 4, dark1, bright2),
            (stack1, False, True, "Emission"),
            (np.zeros((5, 5, 5))).astype(np.uint16),
        ),
    ],
)
def test_correct_dark_bright(inits, corr_inputs, expected):
    corr = Correct(*inits)
    corrected = corr.correct_dark_bright(
        *corr_inputs,
    )
    assert np.array_equal(corrected, expected)


@pytest.mark.parametrize(
    "inits, corr_inputs, expected",
    [
        (
            (hot2, 4, dark1, bright2),
            (stack1, True, False, None),
            "Unknown modality option, valid is Transmission and Emission.",
        ),
        (
            (hot2, 4, dark1, bright2),
            (stack1, True, False, "Nonsence"),
            "Unknown modality option, valid is Transmission and Emission.",
        ),
    ],
)
def test_correct_dark_bright_except(inits, corr_inputs, expected):
    corr = Correct(*inits)
    with pytest.raises(ValueError, match=expected):
        corr.correct_dark_bright(
            *corr_inputs,
        )


# TODO: parametrize integral_bottom
@pytest.mark.parametrize(
    "inits, corr_int_inputs, expected",
    [
        # this one uses bright correction, so ref is take from there
        (
            (hot2, 4, dark1, bright2),
            (stack1,),
            (return_stack1, {"mode": "integral", "ref": 10.0}),
        ),
        # explicit inputs
        (
            (hot2, 4, dark1, bright2),
            (stack1, "integral", True, 50, True),
            (return_stack1, {"mode": "integral", "ref": 10.0}),
        ),
        # no bright correction
        (
            (hot2, 4, dark1, bright2),
            (stack1, "integral", False, 50, True),
            (stack1, {"mode": "integral", "ref": 5.0}),
        ),
        (
            (hot2, -1, dark1, bright2),
            (stack1, "integral", False, 2, True),
            (stack1, {"mode": "integral", "ref": 5.0}),
        ),
    ],
)
def test_correct_int(inits, corr_int_inputs, expected):
    corr = Correct(*inits)
    corrected, report = corr.correct_int(
        *corr_int_inputs,
    )
    assert corr.ref == expected[1]["ref"]

    # assert dimensions of the stacks
    assert corrected.shape == expected[0].shape
    assert np.array_equal(corrected, expected[0])
    assert report["mode"] == expected[1]["mode"]


def test_correct_int_except():
    corr = Correct(hot2, 4, dark1, bright2)
    with pytest.raises(
        ValueError,
        match="Unknown mode option, valid is integral, integral_bottom.",
    ):
        corr.correct_int(stack1, "nonsense", True, 50, True)


def test_correct_int_except2():
    corr = Correct(hot2, 4, dark1)
    with pytest.raises(ValueError, match="No bright field image provided."):
        corr.correct_int(stack1, "integral", True, 50, False)


# test correct_fl_bleach method
# parametrize inputs
@pytest.mark.parametrize(
    "inits, corr_inputs, expected",
    [
        # this one is flat input array, will only lead to normalization
        (
            (hot2, 4, dark1, bright2),
            stack1,
            (np.ones(stack1.shape), {"corr_factors": stack1[0]}),
        ),
        # stack2 has values 10, will only lead to normalization
        (
            (),
            stack2,
            (
                np.ones(stack2.shape),
                {"corr_factors": fl_bleach_corr2},
            ),
        ),
        (
            (),
            stack3,
            (
                np.ones(stack3.shape),
                {"corr_factors": fl_bleach_corr3},
            ),
        ),
    ],
)
def test_correct_fl_bleach(inits, corr_inputs, expected):
    corr = Correct(*inits)
    corrected, dict_corr = corr.correct_fl_bleach(corr_inputs)
    assert np.array_equal(corrected, expected[0])
    # assert number of keys in the dictionary
    assert len(dict_corr) == len(expected[1])
    # assert the keys in the dictionary
    assert dict_corr.keys() == expected[1].keys()
    assert np.array_equal(
        dict_corr["corr_factors"], expected[1]["corr_factors"]
    )


# test clip_and_convert_data method
@pytest.mark.parametrize(
    "input_vals, expected",
    [
        (
            {"data": np.ones((10, 5, 5)) * 10},
            (np.ones((10, 5, 5)) * 65535).astype(np.uint16),
        ),
        (
            {"data": np.ones((10, 5, 5)) * -1},
            np.zeros((10, 5, 5)),
        ),
        (
            {"data": np.array([[0.1, 5], [0.1, -1]])},
            (np.array([[0.1, 1], [0.1, 0]]) * 65535).astype(np.uint16),
        ),
    ],
)
def test_clip_and_convert(input_vals, expected):
    corr = Correct()
    ans = corr.clip_and_convert_data(input_vals["data"])
    assert np.allclose(ans, expected)


# test subtract images
@pytest.mark.parametrize(
    "input_vals, expected",
    [
        (
            {
                "data1": np.ones((10, 5, 5)) * 10,
                "data2": np.ones((10, 5, 5)) * 5,
            },
            np.ones((10, 5, 5)) * 5,
        ),
        (
            {
                "data1": np.ones((10, 5, 5)) * 10,
                "data2": np.ones((10, 5, 5)) * 10,
            },
            np.zeros((10, 5, 5)),
        ),
        (
            {
                "data1": np.ones((10, 5, 5)) * 10,
                "data2": np.ones((10, 5, 5)) * 15,
            },
            np.zeros((10, 5, 5)),
        ),
    ],
)
def test_subtract_images(input_vals, expected):
    corr = Correct()
    ans = corr.subtract_images(input_vals["data1"], input_vals["data2"])
    assert np.allclose(ans, expected)


# inputs for tess_set_dark
arr = np.ones((10, 5, 5))
arr[:, 1, :] = 0

arr1 = np.ones((10, 5, 5))
arr1[:, 1, :] = -1

ret = np.ones((10, 5, 5)) * 2**16 - 1
ret[:, 1, :] = 0


# test set_dark method
@pytest.mark.parametrize(
    "input_vals, expected",
    [
        ({"data": np.ones((10, 5, 5)) * 10}, np.ones((10, 5, 5)) * 10),
        (
            {"data": np.ones((10, 5, 5)) * -1},
            np.ones((10, 5, 5)) * 2**16 - 1,
        ),
        (
            {"data": arr},
            arr,
        ),
        (
            {"data": arr * 0.1},
            arr * 0.1,
        ),
        (
            {"data": arr1},
            ret,
        ),
        (
            {"data": np.array([[0.1, 2**16 - 1], [0.1, -1]])},
            np.array([[1, 2**16 - 1], [1, 0]]).astype(np.uint16),
        ),
    ],
)
def test_set_dark(input_vals, expected):
    corr = Correct()
    corr.set_dark(input_vals["data"])
    assert np.allclose(corr.dark, expected)


# test set_dark method
@pytest.mark.parametrize(
    "input_vals, expected",
    [
        ({"data": np.ones((10, 5, 5)) * 10}, np.ones((10, 5, 5)) * 10),
        (
            {"data": np.ones((10, 5, 5)) * -1},
            np.ones((10, 5, 5)) * 2**16 - 1,
        ),
        (
            {"data": arr},
            arr,
        ),
        (
            {"data": arr * 0.1},
            arr * 0.1,
        ),
        (
            {"data": arr1},
            ret,
        ),
        (
            {"data": np.array([[0.1, 2**16 - 1], [0.1, -1]])},
            np.array([[1, 2**16 - 1], [1, 0]]).astype(np.uint16),
        ),
    ],
)
def test_set_bright(input_vals, expected):
    corr = Correct()
    corr.set_bright(input_vals["data"])
    assert np.allclose(corr.bright, expected)


# test set_dark method
@pytest.mark.parametrize(
    "input_vals, expected",
    [
        ({"data": np.ones((10, 5, 5)) * 10}, np.ones((10, 5, 5)) * 10),
        (
            {"data": np.ones((10, 5, 5)) * -1},
            np.ones((10, 5, 5)) * 2**16 - 1,
        ),
        (
            {"data": arr},
            arr,
        ),
        (
            {"data": arr * 0.1},
            arr * 0.1,
        ),
        (
            {"data": arr1},
            ret,
        ),
        (
            {"data": np.array([[0.1, 2**16 - 1], [0.1, -1]])},
            np.array([[1, 2**16 - 1], [1, 0]]).astype(np.uint16),
        ),
    ],
)
def test_set_bad(input_vals, expected):
    corr = Correct()
    corr.set_bad(input_vals["data"])
    assert np.allclose(corr.bad, expected)


# test set_std_mult method
@pytest.mark.parametrize(
    "input_vals, expected",
    [
        ({"std_mult": 5}, 5),
        ({"std_mult": -1}, 7),
        ({"std_mult": 0}, 7),
        ({"std_mult": 4.5}, 4.5),
    ],
)
def test_set_std_mult(input_vals: dict, expected: float):
    corr = Correct()
    if input_vals["std_mult"] <= 0:
        with pytest.raises(
            ValueError, match="STD multiplier should be positive"
        ):
            corr.set_std_mult(input_vals["std_mult"])
    else:
        corr.set_std_mult(input_vals["std_mult"])
        assert isinstance(corr.std_mult, float)
        assert corr.std_mult == expected
