#!/usr/bin/env python

"""
Tests for Bad pixel correction because it contains
a dialog window
"""

import pytest
import numpy as np
from qtpy.QtWidgets import QMessageBox

ans = np.zeros((5, 5))
ans1 = np.zeros((5, 5))
ans1[2, 1] = 1

# dead pixel should be identified too
ans2 = ans1.copy()
ans2[1, 2] = 0.1


@pytest.mark.parametrize(
    'init_vals, expected',
    [((False, False, 2, 'hot'), ((5, 5), ans1)),
     ((False, True, 2, 'hot'), ((5, 5), ans1)),
     ((False, True, 5, 'hot'), ((5, 5), ans)),
     ((True, True, 2, 'hot'), ((5, 5), ans1)),
     ((True, False, 2, 'hot'), ((5, 5), ans1)),
     ((True, True, 5, 'hot'), ((5, 5), ans)),
     ],
)
def test_bad_pixels1_no_corr(init_vals, expected, request, monkeypatch):
    viewer, widget = request.getfixturevalue("prepare_widget_data1")

    widget.inplace.val, widget.track.val = init_vals[0], init_vals[1]
    widget.std_cutoff.val = init_vals[2]
    widget.flagBad = init_vals[3]

    monkeypatch.setattr(widget, "ask_correct_bad_pixels",
                        lambda *args: QMessageBox.No)

    widget.correctBadPixels()
    assert viewer.layers['Bad-pixels'].data.dtype == np.uint8
    assert viewer.layers['Bad-pixels'].data.shape == expected[0]
    assert np.allclose(viewer.layers['Bad-pixels'].data, expected[1])

    handlers = widget.log.handlers[:]
    for hndlr in handlers:
        widget.log.removeHandler(hndlr)
        hndlr.close()


# test ask_correct_bad_pixels
def test_ask_correct_bad_pixels(request, monkeypatch):
    viewer, widget = request.getfixturevalue("prepare_widget_data1")

    monkeypatch.setattr(widget, "ask_correct_bad_pixels",
                        lambda *args: QMessageBox.Yes)
    img = np.ones((5, 5, 5)) * 10
    img[0, 1,  2] = 1
    img[0, 2, 1] = 30

    ret = img.copy()
    ret[0, 2,  1] = 10
    viewer.layers['img'].data = img
    widget.std_cutoff.val = 2.0
    widget.correctBadPixels()

    assert viewer.layers['img'].data.shape == img.shape
    assert np.array_equal(viewer.layers['img'].data[0], ret[0])
    assert np.array_equal(viewer.layers['img'].data, ret)

    handlers = widget.log.handlers[:]
    for hndlr in handlers:
        widget.log.removeHandler(hndlr)
        hndlr.close()
