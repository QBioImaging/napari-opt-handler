# napari-opt-handler

[![License BSD-3](https://img.shields.io/pypi/l/napari-opt-handler.svg?color=green)](https://raw.githubusercontent.com/QBioImaging/napari-opt-handler/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-opt-handler.svg?color=green)](https://pypi.org/project/napari-opt-handler)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-opt-handler.svg?color=green)](https://python.org)
[![tests](https://github.com/QBioImaging/napari-opt-handler/workflows/tests/badge.svg)](https://github.com/QBioImaging/napari-opt-handler/actions)
[![codecov](https://codecov.io/gh/QBioImaging/napari-opt-handler/branch/main/graph/badge.svg)](https://codecov.io/gh/QBioImaging/napari-opt-handler)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-opt-handler)](https://napari-hub.org/plugins/napari-opt-handler)

Optical Projection Tomography preprocessing plugin for napari

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

place for the gif
<img src="" width="700"/>

Jump to:
- [Usage](#usage)
  - [Starting point](#starting-point)
  - [Global settings](#settings)
  - [Corrections](#corrections)
  - [Other](#other)
- [Installation](#installation)
- [Troubleshooting installation](#troubleshooting-installation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🛀 Usage

### 🏁 Starting point
1. Data streamed from ImSwitch OPT widget (see details here[LINK])
2. Loaded images as data stack
3. Other stack 3D volume data formats
4. Small example data loaded via `File -> Open Sample`

### 🌍 Global settings
TODO
Tracking

Inplace operations

### 📷 Corrections
Dark-field, Bright field and Bad-pixel corrections are directly related to the camera acquisition. Intensity correction deals on the other hand with the instability of the light source.
#### Dark-field correction
TODO
#### Bright-field correction
TODO
#### Bad-pixel correction
TODO
#### Intensity correction
TODO
- If you want to correct for fluorescence photo-bleaching, current version of the plugin does not provide it. Please submit a feature request or upvote an existing one.

### ✂️ Other
TODO
#### Binning
#### ROI
#### -Log

## 💻 Installation

You can install `napari-opt-handler` via [pip]:

    pip install napari-opt-handler

To install latest development version :

    pip install git+https://github.com/QBioImaging/napari-opt-handler.git


## 🎄 Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## 🚓 License

Distributed under the terms of the [BSD-3] license,
"napari-opt-handler" is free and open source software

## 🔨 Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/QBioImaging/napari-opt-handler/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
