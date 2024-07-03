#!/usr/bin/env python

"""
Tests for backtrack class
"""

import numpy as np
import pytest
from napari import Viewer

from napari_opt_handler._qtwidget import PreprocessingnWidget as pw

__author__ = "David Palecek"
__credits__ = ["Teresa M Correia", "Giorgia Tortora"]
__license__ = "GPL"


@pytest.mark.parametrize(
    "backtrack_init, expected",
    [
        ("backtrack_true_false", (True, False)),
        ("backtrack_none_none", (True, False)),
        ("backtrack_true_true", (True, True)),
        ("backtrack_false_true", (False, True)),
        ("backtrack_false_false", (False, False)),
        ("backtrack_1_0", (True, False)),
    ],
)
def test_set_settings_fixture(backtrack_init, expected, request):
    backtrack = request.getfixturevalue(backtrack_init)

    assert backtrack.inplace == expected[0]
    assert backtrack.track == expected[1]


# this works
@pytest.mark.parametrize(
    "backtrack_init, expected",
    [
        ("backtrack_true_false", False),
        ("backtrack_none_none", False),
        ("backtrack_true_true", True),
        ("backtrack_false_true", False),
        ("backtrack_false_false", False),
        ("backtrack_1_0", False),
    ],
)
def test_update_compatible(backtrack_init, expected, request):
    backtrack = request.getfixturevalue(backtrack_init)

    assert backtrack._update_compatible() is expected


# TODO: use fixture in order to avoid the backtrack hanging
@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        ("backtrack_true_false", np.ones((5, 5)), (False, {})),
        ("backtrack_none_none", np.ones((5, 5)), (False, {})),
        (
            "backtrack_true_true",
            np.ones((5, 5)),
            (
                True,
                {"operation": "roi", "data": np.ones((5, 5)), "roi_def": ()},
            ),
        ),
        ("backtrack_false_true", np.ones((5, 5)), (False, {})),
        ("backtrack_false_false", np.ones((5, 5)), (False, {})),
        ("backtrack_1_0", np.ones((5, 5)), (False, {})),
    ],
)
class TestClass:
    def test_update_history_roi(
        self, make_napari_viewer, backtrack_init, input_vals, expected, request
    ):
        backtrack = request.getfixturevalue(backtrack_init)
        assert backtrack.raw_data is None
        assert backtrack.roi_def == ()
        assert backtrack.history_item == {}
        assert backtrack._update_compatible() is expected[0]

        img_old = input_vals
        img_new = np.ones((2, 2))
        viewer: Viewer = make_napari_viewer()
        viewer.add_image(img_old, name="test")
        data_dict = {
            "operation": "roi",
            "data": img_new,
            "roi_def": (1, 1, 3, 3),
        }
        out = backtrack.update_history(viewer.layers["test"], data_dict)

        assert np.array_equal(out, data_dict["data"])
        assert np.array_equal(backtrack.raw_data, viewer.layers["test"].data)
        np.testing.assert_equal(backtrack.history_item, expected[1])


@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        ("backtrack_true_false", (np.ones((5, 5)), 4), (False, {})),
        ("backtrack_none_none", (np.ones((5, 5)), 4), (False, {})),
        (
            "backtrack_true_true",
            (np.ones((5, 5)), 4),
            (
                True,
                {"operation": "bin", "data": np.ones((5, 5)), "bin_factor": 2},
            ),
        ),
        (
            "backtrack_true_true",
            (np.ones((6, 6)), 9),
            (
                True,
                {"operation": "bin", "data": np.ones((6, 6)), "bin_factor": 3},
            ),
        ),
    ],
)
def test_update_history_bin1(
    make_napari_viewer, backtrack_init, input_vals, expected, request
):
    backtrack = request.getfixturevalue(backtrack_init)
    assert backtrack.raw_data is None
    assert backtrack.roi_def == ()
    assert backtrack.history_item == {}
    assert backtrack._update_compatible() is expected[0]

    img_old, binned_value = input_vals
    img_new = np.ones((2, 2)) * binned_value

    viewer: Viewer = make_napari_viewer()
    viewer.add_image(img_old, name="test")
    data_dict = {
        "operation": "bin",
        "data": img_new,
        "bin_factor": 2,
    }
    if expected[1] != {}:
        data_dict["bin_factor"] = expected[1]["bin_factor"]
    out = backtrack.update_history(viewer.layers["test"], data_dict)

    assert np.array_equal(out, data_dict["data"])
    assert isinstance(out, type(data_dict["data"]))
    assert np.array_equal(backtrack.raw_data, viewer.layers["test"].data)
    np.testing.assert_equal(backtrack.history_item, expected[1])


@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        ("backtrack_true_false", np.ones((5, 5)), (False, {})),
        ("backtrack_none_none", np.ones((5, 5)), (False, {})),
        (
            "backtrack_true_true",
            np.ones((5, 5)),
            (
                True,
                {
                    "operation": "corrInt",
                    "data": np.ones((5, 5)),
                    "rect_dim": 50,
                },
            ),
        ),
        (
            "backtrack_true_true",
            np.ones((6, 6)),
            (
                True,
                {
                    "operation": "corrInt",
                    "data": np.ones((6, 6)),
                    "rect_dim": 50,
                },
            ),
        ),
    ],
)
def test_update_history_corrInt(
    make_napari_viewer, backtrack_init, input_vals, expected, request
):
    backtrack = request.getfixturevalue(backtrack_init)
    assert backtrack.raw_data is None
    assert backtrack.roi_def == ()
    assert backtrack.history_item == {}
    assert backtrack._update_compatible() is expected[0]

    img_old = input_vals
    img_new = np.ones((2, 2))

    viewer: Viewer = make_napari_viewer()
    viewer.add_image(img_old, name="test")
    data_dict = {
        "operation": "corrInt",
        "data": img_new,
        "rect_dim": 50,
    }
    out = backtrack.update_history(viewer.layers["test"], data_dict)

    assert np.array_equal(out, data_dict["data"])
    assert isinstance(out, type(data_dict["data"]))
    assert np.array_equal(backtrack.raw_data, viewer.layers["test"].data)
    np.testing.assert_equal(backtrack.history_item, expected[1])


#############
# test undo #
#############
@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        (
            "backtrack_true_true",
            2,
            (
                True,
                {
                    "operation": "bin",
                    "data": np.ones((10, 5, 5)) * 10,
                    "bin_factor": 2,
                },
                np.ones((10, 2, 2)) * 10,
            ),
        ),
        (
            "backtrack_true_true",
            3,
            (
                True,
                {
                    "operation": "bin",
                    "data": np.ones((10, 5, 5)) * 10,
                    "bin_factor": 3,
                },
                np.ones((10, 1, 1)) * 10,
            ),
        ),
    ],
)
def test_undo_bin(
    make_napari_viewer, backtrack_init, input_vals, expected, request
):
    viewer: Viewer = make_napari_viewer()

    _widget = pw(viewer)
    _widget.history = request.getfixturevalue(backtrack_init)
    viewer.window.add_dock_widget(_widget, area="right")
    _widget.bin_factor.val = input_vals

    assert len(viewer.window._dock_widgets) == 1
    assert _widget.history._update_compatible() is expected[0]

    img, dark, bright, bad = request.getfixturevalue("data1")

    viewer.add_image(img, name="img")
    _widget.image_layer_select.value = viewer.layers["img"]
    assert viewer.layers["img"].data.shape == (10, 5, 5)

    viewer.add_image(dark, name="dark")
    _widget.dark_layer_select.value = viewer.layers["dark"]
    viewer.add_image(bright, name="bright")
    _widget.bright_layer_select.value = viewer.layers["bright"]
    viewer.add_image(bad, name="bad")
    _widget.bad_layer_select.value = viewer.layers["bad"]

    _widget.bin_stack()
    np.testing.assert_equal(_widget.history.history_item, expected[1])
    assert np.array_equal(viewer.layers["img"].data, expected[2])

    _widget.undoHistory()
    assert np.array_equal(viewer.layers["img"].data, expected[1]["data"])
    assert _widget.history.history_item == {}
    handlers = _widget.log.handlers[:]
    for hndlr in handlers:
        _widget.log.removeHandler(hndlr)
        hndlr.close()


@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        ("backtrack_true_true", 1, (True, {})),
        ("backtrack_true_false", 4, (False, {})),
    ],
)
def test_undo_bin_exception1(
    make_napari_viewer, backtrack_init, input_vals, expected, request
):
    viewer: Viewer = make_napari_viewer()

    _widget = pw(viewer)
    _widget.history = request.getfixturevalue(backtrack_init)
    viewer.window.add_dock_widget(_widget, area="right")
    _widget.bin_factor.val = input_vals

    assert len(viewer.window._dock_widgets) == 1
    assert _widget.history._update_compatible() is expected[0]

    img, dark, bright, bad = request.getfixturevalue("data1")

    viewer.add_image(img, name="img")
    _widget.image_layer_select.value = viewer.layers["img"]
    assert viewer.layers["img"].data.shape == (10, 5, 5)

    viewer.add_image(dark, name="dark")
    _widget.dark_layer_select.value = viewer.layers["dark"]
    viewer.add_image(bright, name="bright")
    _widget.bright_layer_select.value = viewer.layers["bright"]
    viewer.add_image(bad, name="bad")
    _widget.bad_layer_select.value = viewer.layers["bad"]

    _widget.bin_stack()
    np.testing.assert_equal(_widget.history.history_item, expected[1])

    with pytest.raises(ValueError) as info:
        _widget.undoHistory()
    assert str(info.value) == "No State to revert to."

    handlers = _widget.log.handlers[:]
    for hndlr in handlers:
        _widget.log.removeHandler(hndlr)
        hndlr.close()


@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        ("backtrack_false_true", 7, (False, {})),
    ],
)
def test_undo_bin_exception2(
    make_napari_viewer, backtrack_init, input_vals, expected, request
):
    viewer: Viewer = make_napari_viewer()

    _widget = pw(viewer)
    _widget.history = request.getfixturevalue(backtrack_init)
    viewer.window.add_dock_widget(_widget, area="right")
    _widget.bin_factor.val = input_vals

    assert len(viewer.window._dock_widgets) == 1
    assert _widget.history._update_compatible() is expected[0]

    img, dark, bright, bad = request.getfixturevalue("data1")

    viewer.add_image(img, name="img")
    _widget.image_layer_select.value = viewer.layers["img"]
    assert viewer.layers["img"].data.shape == (10, 5, 5)

    viewer.add_image(dark, name="dark")
    _widget.dark_layer_select.value = viewer.layers["dark"]
    viewer.add_image(bright, name="bright")
    _widget.bright_layer_select.value = viewer.layers["bright"]
    viewer.add_image(bad, name="bad")
    _widget.bad_layer_select.value = viewer.layers["bad"]

    with pytest.raises(ValueError) as info:
        _widget.bin_stack()
    assert str(info.value) == "Bin factor is too large for the image size."
    np.testing.assert_equal(_widget.history.history_item, expected[1])

    with pytest.raises(ValueError) as info:
        _widget.undoHistory()
    assert str(info.value) == "No State to revert to."

    handlers = _widget.log.handlers[:]
    for hndlr in handlers:
        _widget.log.removeHandler(hndlr)
        hndlr.close()


######################
# undo ROI selection #
######################
@pytest.mark.parametrize(
    "backtrack_init, input_vals, expected",
    [
        (
            "backtrack_true_true",
            np.array([[1, 1]]),
            (
                True,
                {
                    "operation": "roi",
                    "data": np.ones((10, 5, 5)) * 10,
                    "roi_def": (),
                },
                np.ones((10, 4, 4)) * 10,
                (1, 5, 1, 5),
            ),
        ),
        (
            "backtrack_true_true",
            np.array([[2, 1]]),
            (
                True,
                {
                    "operation": "roi",
                    "data": np.ones((10, 5, 5)) * 10,
                    "roi_def": (),
                },
                np.ones((10, 3, 4)) * 10,
                (2, 5, 1, 5),
            ),
        ),
    ],
)
def test_undo_roi(
    make_napari_viewer, backtrack_init, input_vals, expected, request
):
    viewer: Viewer = make_napari_viewer()

    _widget = pw(viewer)
    _widget.history = request.getfixturevalue(backtrack_init)
    viewer.window.add_dock_widget(_widget, area="right")

    assert len(viewer.window._dock_widgets) == 1
    assert _widget.history._update_compatible() is expected[0]

    img, dark, bright, bad = request.getfixturevalue("data1")

    viewer.add_image(img, name="img")
    _widget.image_layer_select.value = viewer.layers["img"]
    assert viewer.layers["img"].data.shape == (10, 5, 5)

    viewer.add_image(dark, name="dark")
    _widget.dark_layer_select.value = viewer.layers["dark"]
    viewer.add_image(bright, name="bright")
    _widget.bright_layer_select.value = viewer.layers["bright"]
    viewer.add_image(bad, name="bad")
    _widget.bad_layer_select.value = viewer.layers["bad"]

    # add points layer
    viewer.add_points(input_vals, name="points")
    _widget.points_layer_select.value = viewer.layers["points"]

    _widget.select_ROIs()
    np.testing.assert_equal(_widget.history.history_item, expected[1])
    assert np.array_equal(viewer.layers["img"].data, expected[2])
    assert np.array_equal(viewer.layers["points"].data, input_vals)
    assert _widget.history.roi_def == expected[3]

    _widget.undoHistory()
    assert np.array_equal(viewer.layers["img"].data, expected[1]["data"])
    assert _widget.history.history_item == {}

    # continue with binning or second roi selection
    _widget.bin_factor.val = 2
    _widget.bin_stack()
    assert np.array_equal(viewer.layers["img"].data, np.ones((10, 2, 2)) * 10)

    handlers = _widget.log.handlers[:]
    for hndlr in handlers:
        _widget.log.removeHandler(hndlr)
        hndlr.close()
