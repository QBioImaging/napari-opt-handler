#!/usr/bin/env python

import napari
from _qtwidget import PreprocessingnWidget

DEBUG = True


viewer = napari.Viewer()
optWidget = PreprocessingnWidget(viewer)
viewer.window.add_dock_widget(optWidget, name="OPT Preprocessing")
# warnings.filterwarnings('ignore')
if DEBUG:
    import glob

    # load example data from data folder
    viewer.open("src/napari_opt_handler/sample_data/corr_hot.tiff", name="bad")
    viewer.open(
        "src/napari_opt_handler/sample_data/dark_field.tiff", name="dark"
    )
    viewer.open(
        "src/napari_opt_handler/sample_data/flat_field.tiff", name="bright"
    )

    # open OPT stack
    viewer.open(
        glob.glob("src/napari_opt_handler/sample_data/16*"),
        stack=True,
        name="OPT data",
    )
    # set image layer to OPT data
    optWidget.image_layer_select.value = viewer.layers["OPT data"]
    optWidget.bad_layer_select.value = viewer.layers["bad"]
    optWidget.dark_layer_select.value = viewer.layers["dark"]
    optWidget.bright_layer_select.value = viewer.layers["bright"]
napari.run()
