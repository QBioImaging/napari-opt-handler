"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

import numpy as np
import pooch
from skimage.io import imread

base_url = (
    "https://raw.githubusercontent.com/QBioImaging/napari-opt-handler/"
    "main/src/napari_opt_handler/sample_data"
)


def make_sample_data():
    """Generates an image"""
    # load image from sample data
    data = []
    dlist = [
        "16-48-16_0000.tiff",
        "16-48-17_0001.tiff",
        "16-48-18_0002.tiff",
        "16-48-18_0003.tiff",
        "16-48-19_0004.tiff",
        "16-48-20_0005.tiff",
        "16-48-21_0006.tiff",
    ]

    for filename in dlist:
        url = f"{base_url}/{filename}"
        file = pooch.retrieve(url=url, known_hash=None)
        data.append(imread(file))
    data = np.stack(data, axis=0)
    data_hot = imread(
        pooch.retrieve(url=f"{base_url}/corr_hot.tiff", known_hash=None),
    )

    data_dark = imread(
        pooch.retrieve(url=f"{base_url}/dark_field.tiff", known_hash=None),
    )

    data_bright = imread(
        pooch.retrieve(url=f"{base_url}/flat_field.tiff", known_hash=None),
    )

    # TODO no idea how to set layers to optWidget.image_layer_select.value
    # Did I missed something in napari.current_viewer()
    return [
        (data_hot, {"name": "bad"}, "Image"),
        (data_dark, {"name": "dark"}, "Image"),
        (data_bright, {"name": "bright"}, "Image"),
        (data, {"name": "OPT data"}, "Image"),
    ]
