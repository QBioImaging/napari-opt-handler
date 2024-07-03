import numpy as np
import pytest
from napari import Viewer

from napari_opt_handler._qtwidget import PreprocessingnWidget as pw
from napari_opt_handler.backtrack import Backtrack

bad_px = np.ones((5, 5))
bad_px[2, 1] = 100
bad_px[1, 2] = 0.1


@pytest.fixture(scope="function")
def backtrack_true_false():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=True, track_value=False)
    yield backtrack
    del backtrack


@pytest.fixture(scope="function")
def backtrack_none_none():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=None, track_value=None)
    yield backtrack
    del backtrack


@pytest.fixture(scope="function")
def backtrack_true_true():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=True, track_value=True)
    yield backtrack
    del backtrack


@pytest.fixture(scope="function")
def backtrack_false_true():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=False, track_value=True)
    yield backtrack
    del backtrack


@pytest.fixture(scope="function")
def backtrack_false_false():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=False, track_value=False)
    yield backtrack
    del backtrack


@pytest.fixture(scope="function")
def backtrack_1_0():
    backtrack = Backtrack()
    backtrack.set_settings(inplace_value=1, track_value=0)
    yield backtrack
    del backtrack


@pytest.fixture
def prepare_widget_data1(make_napari_viewer, request):
    viewer: Viewer = make_napari_viewer()

    _widget = pw(viewer)
    viewer.window.add_dock_widget(_widget, area="right")

    img, dark, bright, bad = request.getfixturevalue("data1")

    assert np.allclose(bad, bad_px)

    viewer.add_image(img, name="img")
    _widget.image_layer_select.value = viewer.layers["img"]

    viewer.add_image(dark, name="dark")
    _widget.dark_layer_select.value = viewer.layers["dark"]

    viewer.add_image(bright, name="bright")
    _widget.bright_layer_select.value = viewer.layers["bright"]

    viewer.add_image(bad, name="bad")
    _widget.bad_layer_select.value = viewer.layers["bad"]

    return viewer, _widget


@pytest.fixture
def prepare_widget_data2(make_napari_viewer, request):
    viewer: Viewer = make_napari_viewer()

    _widget = pw(viewer)
    viewer.window.add_dock_widget(_widget, area="right")

    img, dark, bright, bad = request.getfixturevalue("data2")

    viewer.add_image(img, name="img")
    _widget.image_layer_select.value = viewer.layers["img"]

    viewer.add_image(dark, name="dark")
    _widget.dark_layer_select.value = viewer.layers["dark"]
    viewer.add_image(bright, name="bright")
    _widget.bright_layer_select.value = viewer.layers["bright"]
    viewer.add_image(bad, name="bad")
    _widget.bad_layer_select.value = viewer.layers["bad"]
    return viewer, _widget


# fixture for image layer, dark and bright layer and bad pixel layer
@pytest.fixture(scope="function")
def data1():
    img = np.ones((10, 5, 5)) * 10
    dark = np.ones((5, 5)) * 0.1
    bright = np.ones((5, 5)) * 11
    bad_px = np.ones((5, 5))
    bad_px[2, 1] = 100
    bad_px[1, 2] = 0.1
    return img, dark, bright, bad_px


# fixture for image layer, dark and bright layer and bad pixel layer
# Not very reasonable but kinda valid
@pytest.fixture(scope="function")
def data2():
    img = np.ones((10, 5, 5)) * 10
    dark = np.ones((5, 5)) * 0.1
    bright = np.ones((5, 5)) * 8.8
    bad_px = np.ones((5, 5)) * 0.1
    bad_px[2, 1] = 10
    return img, dark, bright, bad_px
