import numpy as np
import logging

import napari
from napari.layers import Image, Points
from magicgui.widgets import create_widget

from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton,
    QRadioButton, QLabel,
    QButtonGroup, QGroupBox,
    QMessageBox, QTabWidget, QPlainTextEdit,
)
from typing import List, Tuple
from enum import Enum
from .plotting import PlotDialog

from .widget_settings import Settings, Combo_box
from .corrections import Correct
from .backtrack import Backtrack
from .utils import (
    select_roi,
    bin_3d,
)


DEBUG = True
badPxDict = {
    'Identify Hot pxs': 'hot',
    'Identify Dead pxs': 'dead',
    'Identify Both': 'both',
}


class neighbours_choice_modes(Enum):
    n4 = 1
    n8 = 2


def layer_container_and_selection(
        viewer=None,
        layer_type=Image,
        container_name='Layer'):
    """
    Create a container and a dropdown widget to select the layer.

    Returns:
        layer_selection_container (qtpy.QWidget): Widget for displaying
        the layer selection container,
        layer_select (qtpy.QWidget): Widget containing the selection options
            for the layer.
    """
    layer_selection_container = QWidget()
    layer_selection_container.setLayout(QHBoxLayout())
    layer_selection_container.layout().addWidget(QLabel(container_name))
    layer_select = create_widget(annotation=layer_type, label="layer")
    layer_selection_container.layout().addWidget(layer_select.native)
    layer_selection_container.layout().setContentsMargins(0, 0, 0, 0)

    if viewer is not None and viewer.layers.selection.active is not None:
        layer_select.value = viewer.layers.selection.active

    return layer_selection_container, layer_select


class QTextEditLogger(logging.Handler):
    def __init__(self):
        super().__init__()
        self.widget = QPlainTextEdit()
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class PreprocessingnWidget(QWidget):
    name = 'Preprocessor'

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.logger = self.init_logger()

        self.viewer = viewer
        self.history = Backtrack()
        self.corr = Correct()
        self.setup_ui(self.logger)

        # setup_ui initializes track and inplace values
        # here I update history instance flags
        self.updateHistoryFlags()

    def correctDarkBright(self):
        """
        Corrects the dark and bright levels of the image.

        This method applies correction to the dark and bright levels
        of the image based on the specified parameters. It uses the
        `correct_dark_bright` method from the `corr` object
        to perform the correction.

        Parameters:
            None

        Returns:
            None

        Raises:
            None
        """
        logging.info('Correcting Dark and Bright')

        original_image = self.image_layer_select.value
        data_corr = self.corr.correct_dark_bright(
                        original_image.data,
                        useDark=self.flagDark.val,
                        useBright=self.flagBright.val,
                        modality=self.flagExp,
                        )

        # history update
        if self.history.inplace:
            new_data = {'operation': 'corrDB',
                        'data': data_corr}
            data_corr = self.history.update_history(original_image,
                                                    new_data)

        # display corrected image
        self.show_image(data_corr,
                        'Dark-Bright-corr_' + original_image.name,
                        original_image.contrast_limits)

    def ask_correct_bad_pixels(self, hotPxs: List[Tuple],
                               deadPxs: List[Tuple]) -> Enum:
        """Ask the user if they want to correct all the bad pixels.

        This method displays a message box asking the user if they want
        to correct all the bad pixels. The message box includes information
        about the number of hot pixels and dead pixels.

        Args:
            hotPxs (List[Tuple]): A list of tuples representing the
                coordinates of hot pixels.
            deadPxs (List[Tuple]): A list of tuples representing the
                coordinates of dead pixels.

        Returns:
            Enum: The button code of the user's choice. It can be either
                QMessageBox.Yes or QMessageBox.No.
        """
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Bad Pixel correction")
        dlg.setText("Do you want to correct all those pixels! \n"
                    "It can take a while. \n"
                    f"N_hot: {len(hotPxs)} + N_dead: {len(deadPxs)}")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return dlg.exec_()

    def add_bad_pixels_labels(self) -> None:
        """Adds labels for bad pixels to the viewer.

        This method takes the coordinates of hot and dead pixels and adds
        labels to the viewer accordingly. The labels are represented
        by different integer values (1 for hot pixels, 2 for dead pixels).

        Returns:
            None
        """
        labels = np.zeros(self.bad_layer_select.value.data.shape,
                          dtype=np.uint8)
        if len(self.corr.hot_pxs) > 0:
            labels[tuple(zip(*self.corr.hot_pxs))] = 1
        if len(self.corr.dead_pxs) > 0:
            labels[tuple(zip(*self.corr.dead_pxs))] = 2
        self.viewer.add_labels(labels,
                               name='Bad-pixels')

    def correctBadPixels(self) -> None:
        """Corrects hot pixels in the image.

        This method corrects hot pixels in the image using a correction
        algorithm. It calculates the number of hot pixels and dead pixels,
        and provides an option to either correct the hot pixels or display
        them. If the correction is performed in-place, it also
        updates the history of the image.

        Returns:
            None
        """
        original_image = self.image_layer_select.value

        # init correction class
        logging.info('Correcting hot pixels.')
        self.corr.std_mult = self.std_cutoff.val
        self.corr.get_bad_pxs(mode=self.flagBad)
        hotPxs, deadPxs = self.corr.hot_pxs, self.corr.dead_pxs
        logging.info(f'Hot pixels: {len(hotPxs)}, Dead pixels: {len(deadPxs)}')

        if self.ask_correct_bad_pixels(hotPxs, deadPxs) == QMessageBox.No:
            self.add_bad_pixels_labels()
            return

        # Correction is done, TODO: ooptimized for the volumes and threaded
        # TODO: show progress bar in the widget.
        data_corr = np.zeros(original_image.data.shape,
                             dtype=original_image.data.dtype)
        for i, img in enumerate(original_image.data):
            data_corr[i] = self.corr.correctBadPxs(img)

        # history update
        if self.history.inplace:
            new_data = {'operation': 'corrBP',
                        'data': data_corr,
                        'mode': self.flagBad,
                        'hot_pxs': hotPxs,
                        'dead_pxs': deadPxs,
                        }
            data_corr = self.history.update_history(original_image,
                                                    new_data)

        self.show_image(data_corr,
                        'Bad-px-corr_' + original_image.name,
                        original_image.contrast_limits)

    def correctIntensity(self) -> None:
        """Corrects the intensity of the image.

        This method performs intensity correction on the selected image layer.
        It uses the `Correct` class to perform the correction and updates the
        image data accordingly. If the correction is performed in-place, it
        also updates the history of the image.
        """
        logging.info('Correcting intensity.')

        # data to correct
        original_image = self.image_layer_select.value

        # intensity correction
        data_corr, corr_dict = self.corr.correct_int(
                                    original_image.data,
                                    use_bright=False,
                                    rect_dim=self.rectSize.val)

        # open widget with a intensity plots
        self.plotDialog = PlotDialog(self, corr_dict)
        self.plotDialog.resize(800, 400)
        self.plotDialog.show()

        # history update
        if self.history.inplace:
            new_data = {'operation': 'corrInt',
                        'data': data_corr,
                        'rect_dim': self.rectSize.val,
                        }
            data_corr = self.history.update_history(original_image, new_data)

        self.show_image(data_corr,
                        'Int-corr_' + original_image.name,
                        original_image.contrast_limits)

    def select_ROIs(self):
        """Selects regions of interest (ROIs) based on the given points.

        This method takes the selected image layer and the points layer as
        input. It calculates the upper-left corner coordinates of the last
        point in the points layer, and displays them in a message box. Then,
        it uses the `select_roi` function to extract the selected ROI from
        the original image data, based on the last point, ROI height, and ROI
        width. If the `inplace` flag is set in the `history` object,
        it updates the history with the ROI operation.
        """
        original_image = self.image_layer_select.value
        try:
            points = self.points_layer_select.value.data
        except AttributeError:
            logging.error('No points layer selected.')
            return

        logging.info(f'UL corner coordinates: {points[0][1:].astype(int)}')

        # DP: do I need roi pars for tracking?
        selected_roi, roi_pars = select_roi(original_image.data,
                                            points[-1],  # last point
                                            self.roi_height.val,
                                            self.roi_width.val)

        # history update
        if self.history.inplace:
            new_data = {'operation': 'roi',
                        'data': selected_roi,
                        'roi_def': roi_pars,
                        }
            selected_roi = self.history.update_history(original_image,
                                                       new_data)

        self.show_image(selected_roi,
                        'ROI_' + original_image.name,
                        original_image.contrast_limits)

    def bin_stack(self):
        """Bins the selected image stack by the specified bin factor.

        If the bin factor is 1, nothing is done and an info notification is
        shown. Otherwise, the selected image stack is binned by the bin factor
        and the binned stack is displayed. The original and binned stack shapes
        are also shown in an info notification.

        If the history is set to inplace, the binning operation is added to the
        history with the updated binned stack.
        """
        if self.bin_factor.val == 1:
            logging.info('Bin factor is 1, nothing to do.')
            return

        original_image = self.image_layer_select.value

        # evaluate maximum possible bin factor
        if self.bin_factor.val > min(original_image.data.shape):
            raise ValueError('Bin factor is too large for the image size.')

        binned_roi = bin_3d(original_image.data, self.bin_factor.val)
        logging.info(
            'Original shape: %s \n binned shape: %s',
            original_image.data.shape,
            binned_roi.shape)

        # history update
        if self.history.inplace:
            new_data = {'operation': 'bin',
                        'data': binned_roi,
                        'bin_factor': self.bin_factor.val,
                        }
            binned_roi = self.history.update_history(original_image,
                                                     new_data)

        self.show_image(binned_roi,
                        'binned_' + original_image.name,
                        original_image.contrast_limits)

    def calcLog(self):
        """Calculate the logarithm of the image data.

        This method calculates the logarithm of the image data and updates
        the displayed image accordingly. It also updates the history if the
        operation is performed in-place.
        """
        logging.info('Calculating -Log')
        original_image = self.image_layer_select.value
        log_image = -np.log10(original_image.data)

        if self.history.inplace:
            new_data = {'operation': 'log',
                        'data': log_image,
                        }
            log_image = self.history.update_history(original_image, new_data)

        self.show_image(log_image,
                        '-Log_' + original_image.name,
                        )

    def undoHistory(self):
        """Undo the last operation in the history.

        This method undoes the last operation in the history and updates the
        displayed image accordingly. If the history is empty, an info
        notification is shown.
        """
        self.image_layer_select.value.data, last_op = self.history.undo()

        # reset contrast limits
        self.image_layer_select.value.reset_contrast_limits()
        logging.info(f'Undoing {last_op}')

    ##################
    # Helper methods #
    ##################
    def showEvent(self, event) -> None:
        """
        This method is called when the widget is shown on the screen.

        Args:
            event: A QShowEvent object representing the show event.

        Returns:
            None
        """
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        """
        Reset the choices for all layer selects.

        Args:
            event (Event, optional): The event that triggered the reset.
                Defaults to None.
        """
        self.image_layer_select.reset_choices(event)
        self.bad_layer_select.reset_choices(event)
        self.dark_layer_select.reset_choices(event)
        self.bright_layer_select.reset_choices(event)
        self.points_layer_select.reset_choices(event)

    def show_image(self, image_values: np.ndarray,
                   fullname: str,
                   contrast: List[float] = None):
        """
        Show an image in the napari viewer.

        Args:
            image_values (np.ndarray): The image data to be displayed.
            fullname (str): The name of the image layer.
            contrast (List[float], optional): The contrast limits for the
                image. Defaults to None.
        """
        if self.inplace.val:
            fullname = self.image_layer_select.value.name
            self.viewer.layers[fullname].data = image_values
            if contrast is not None:
                self.viewer.layers[fullname].contrast_limits = contrast
            else:
                self.viewer.layers[fullname].reset_contrast_limits()

        else:
            self.viewer.add_image(image_values,
                                  name=fullname,
                                  contrast_limits=contrast,
                                  interpolation2d='linear')

    def set_preprocessing(self, *args):
        """ Sets preprocessing parameters
        """
        if hasattr(self, 'h'):
            self.h.inplace_val = self.inplace.val
            self.h.track_val = self.track.val
            self.h.cutoff_val = self.std_cutoff.val
            self.h.height_val = self.roi_height.val
            self.h.width_val = self.roi_width.val
            self.h.binning_val = self.bin_factor.val
            self.h.rect_val = self.rectSize.val
            self.h.neigh_mode_val = self.neigh_mode.val

    def updateHistoryFlags(self):
        """Update the history flags based on the current values of
        inplace and track.
        """
        self.history.set_settings(self.inplace.val, self.track.val)

    def set_bad_layer(self):
        """ Set the bad pixel layer for correction."""
        if self.bad_layer_select.value is not None:
            logging.info('Setting bad pixel layer')
            self.corr.set_bad(self.bad_layer_select.value.data)

    def set_dark_layer(self):
        """ Set the dark layer for correction."""
        if self.dark_layer_select.value is not None:
            logging.info('Setting dark layer')
            self.corr.set_dark(self.dark_layer_select.value.data)

    def set_bright_layer(self):
        """ Set the bright layer for correction."""
        if self.bright_layer_select.value is not None:
            logging.info('Setting bright layer')
            self.corr.set_bright(self.bright_layer_select.value.data)

    ##############
    # UI methods #
    ##############
    def setup_ui(self, logger: logging.Logger):
        """
        Set up the user interface for the widget.

        Args:
            logger (logging.Logger): The logger object for logging messages.

        Returns:
            None
        """
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        mode_layout = QVBoxLayout()
        layout.addLayout(mode_layout)
        self.selectProcessingMode(mode_layout)

        # layers selection containers
        (image_selection_container,
         self.image_layer_select) = layer_container_and_selection(
                                        viewer=self.viewer,
                                        layer_type=Image,
                                        container_name='Image to analyze',
                                        )

        (bad_image_selection_container,
         self.bad_layer_select) = layer_container_and_selection(
                                    viewer=self.viewer,
                                    layer_type=Image,
                                    container_name='bad correction image',
                                    )
        self.bad_layer_select.changed.connect(self.set_bad_layer)

        (dark_image_selection_container,
         self.dark_layer_select) = layer_container_and_selection(
                                        viewer=self.viewer,
                                        layer_type=Image,
                                        container_name='Dark correction image',
                                        )
        self.dark_layer_select.changed.connect(self.set_dark_layer)

        (bright_image_selection_container,
         self.bright_layer_select) = layer_container_and_selection(
                            viewer=self.viewer,
                            layer_type=Image,
                            container_name='Bright correction image',
                            )
        self.bright_layer_select.changed.connect(self.set_bright_layer)

        (points_selection_container,
         self.points_layer_select) = layer_container_and_selection(
                            viewer=self.viewer,
                            layer_type=Points,
                            container_name='Points layer for ROI selection',
                            )

        # layers selection layout
        image_layout = QVBoxLayout()
        layout.addLayout(image_layout)
        self.layout().addWidget(image_selection_container)
        self.layout().addWidget(bad_image_selection_container)
        self.layout().addWidget(dark_image_selection_container)
        self.layout().addWidget(bright_image_selection_container)
        self.layout().addWidget(points_selection_container)

        # inplace and track options
        settings_layout = QVBoxLayout()
        layout.addLayout(settings_layout)

        self.createSettings(settings_layout, logger)

    def selectProcessingMode(self, slayout: QVBoxLayout) -> None:
        """
        Selects the processing mode and sets up the corresponding UI elements.

        Args:
            slayout (QVBoxLayout): The layout to which the UI elements will
                be added.

        Returns:
            None
        """
        groupbox = QGroupBox('Global settings')
        box = QVBoxLayout()
        groupbox.setLayout(box)

        # layout for radio buttons
        boxSetting = QHBoxLayout()
        self.inplace = Settings('Inplace operations',
                                dtype=bool,
                                initial=True,
                                layout=boxSetting,
                                write_function=self.set_preprocessing)

        self.track = Settings('Track',
                              dtype=bool,
                              initial=False,
                              layout=boxSetting,
                              write_function=self.set_preprocessing)

        box.addLayout(boxSetting)
        # undo button
        self.undoBtn = QPushButton('Undo')
        self.undoBtn.clicked.connect(self.undoHistory)
        box.addWidget(self.undoBtn)

        # make self.track visible only when self.inplace is True
        slayout.addWidget(groupbox)
        self.setTrackVisibility()
        self.inplace.sbox.stateChanged.connect(self.updateHistoryFlags)
        self.track.sbox.stateChanged.connect(self.updateHistoryFlags)

    def setTrackVisibility(self) -> None:
        """
        Sets the visibility of track-related widgets based on
        the values of other widgets.

        This method updates the visibility of the track-related widgets based
        on the values of the `inplace` and `track` widgets. It sets the
        visibility of the track spin box and label based on the value of the
        inplace spin box. It also sets the visibility of the undo button based
        on the value of the track spin box.

        Returns:
            None
        """
        # initial setting
        self.track.sbox.setVisible(self.inplace.val)
        self.track.lab.setVisible(self.inplace.val)
        self.undoBtn.setVisible(self.track.val)

        # update visibility of track box when inplace box is changed
        self.inplace.sbox.stateChanged.connect(
            lambda: self.track.sbox.setVisible(self.inplace.val))
        # update the label visibility of track box when inplace box is changed
        self.inplace.sbox.stateChanged.connect(
            lambda: self.track.lab.setVisible(self.inplace.val))

        # update visibility of undoBtn when track box is changed
        self.track.sbox.stateChanged.connect(
            lambda: self.undoBtn.setVisible(self.track.val))

    def updateExperimentMode(self, button: QRadioButton):
        """
        Updates the experiment mode based on the selected radio button.

        Args:
            button (QRadioButton): The selected radio button.

        Returns:
            None
        """
        self.flagExp = button.text()
        logging.info(f'Experiment modality: {self.flagExp}')

    def updateBadPixelsMode(self, button: QRadioButton) -> None:
        """
        Updates the bad pixels mode based on the selected radio button.

        Args:
            button (QRadioButton): The selected radio button.

        Returns:
            None
        """
        self.flagBad = badPxDict[button.text()]
        logging.info(f'Bad pixels option: {self.flagBad}')

    def init_logger(self):
        """
        Initialize the logger for the widget.

        Returns:
            QTextEditLogger: The logger object for logging messages.
        """
        logTextBox = QTextEditLogger()
        logTextBox.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log = logging.getLogger()
        self.log.addHandler(logTextBox)

        # You can control the logging level
        logging.getLogger().setLevel(logging.INFO)

        return logTextBox

    def createSettings(self, slayout: QWidget,
                       logger: QTextEditLogger) -> None:
        """
        Create the settings for the widget.

        Args:
            slayout (QWidget): The layout for the settings.
            logger (QTextEditLogger): The logger object for logging messages.

        Returns:
            None
        """
        # Linear structure to impose correct order
        tabsCorr = QTabWidget()

        groupboxDark = QGroupBox('Dark/Bright Correction')
        boxAll = QVBoxLayout()
        boxExp = QHBoxLayout()
        groupboxDark.setLayout(boxAll)

        # create QradioButtons
        groupExpMode = QButtonGroup(self)
        groupExpMode.buttonClicked.connect(self.updateExperimentMode)

        transExp = QRadioButton('Transmission')
        transExp.setChecked(True)
        groupExpMode.addButton(transExp)
        boxExp.addWidget(transExp)

        emlExp = QRadioButton('Emission')
        groupExpMode.addButton(emlExp)
        boxExp.addWidget(emlExp)
        boxAll.addLayout(boxExp)

        # update flag
        self.updateExperimentMode(groupExpMode.checkedButton())

        boxInclude = QHBoxLayout()
        # checkboxes
        self.flagDark = Settings('Include Dark',
                                 dtype=bool,
                                 initial=False,
                                 layout=boxInclude,
                                 write_function=self.set_preprocessing)
        self.flagBright = Settings('Include Bright',
                                   dtype=bool,
                                   initial=False,
                                   layout=boxInclude,
                                   write_function=self.set_preprocessing)
        boxAll.addLayout(boxInclude)
        # Now comes button
        self.addButton(boxAll, 'Correct Dark + Bright', self.correctDarkBright)
        tabsCorr.addTab(groupboxDark, 'Dark/Bright')

        # create a qtab widget for the bad pixel correction
        # create a groupbox for bad pixel correction
        groupboxBad = QGroupBox('Bad pixels correction')
        boxBad = QVBoxLayout()
        groupboxBad.setLayout(boxBad)

        # bad pixel correction
        self.neigh_mode = Combo_box(name='Mode',
                                    choices=neighbours_choice_modes,
                                    layout=boxBad,
                                    write_function=self.set_preprocessing,
                                    )
        self.std_cutoff = Settings('bad STD cutoff',
                                   dtype=int,
                                   initial=5,
                                   vmin=1,
                                   vmax=20,
                                   layout=boxBad,
                                   write_function=self.set_preprocessing)

        groupBadPxMode = QButtonGroup(self)
        groupBadPxMode.buttonClicked.connect(self.updateBadPixelsMode)

        flagGetBad = QRadioButton('Identify Hot pxs')
        flagGetBad.setChecked(True)
        groupBadPxMode.addButton(flagGetBad)
        boxBad.addWidget(flagGetBad)

        flagGetDead = QRadioButton('Identify Dead pxs')
        groupBadPxMode.addButton(flagGetDead)
        boxBad.addWidget(flagGetDead)

        flagGetBoth = QRadioButton('Identify Both')
        groupBadPxMode.addButton(flagGetBoth)
        boxBad.addWidget(flagGetBoth)

        self.updateBadPixelsMode(groupBadPxMode.checkedButton())

        self.addButton(boxBad, 'Bad pixel correction', self.correctBadPixels)
        tabsCorr.addTab(groupboxBad, 'Bad pixels')

        # create a groupbox for Intensity correction
        groupboxInt = QGroupBox('Intensity correction')
        boxInt = QVBoxLayout()
        groupboxInt.setLayout(boxInt)

        # rectangle size of the corner of the image used for int calc
        self.rectSize = Settings('Rectangle size',
                                 dtype=int,
                                 initial=50,
                                 vmin=10,
                                 vmax=500,
                                 layout=boxInt,
                                 write_function=self.set_preprocessing)
        self.addButton(boxInt, 'Intensity correction', self.correctIntensity)
        tabsCorr.addTab(groupboxInt, 'Intensity Corr')
        slayout.addWidget(tabsCorr)

        # select ROI
        tabProcess = QTabWidget()
        groupboxRoi = QGroupBox('ROI selection')
        boxRoi = QVBoxLayout()
        groupboxRoi.setLayout(boxRoi)
        self.roi_height = Settings('ROI height',
                                   dtype=int,
                                   initial=200,
                                   vmin=1,
                                   vmax=5000,
                                   layout=boxRoi,
                                   write_function=self.set_preprocessing)

        self.roi_width = Settings('ROI width',
                                  dtype=int,
                                  initial=200,
                                  vmin=1,
                                  vmax=5000,
                                  layout=boxRoi,
                                  write_function=self.set_preprocessing)
        self.addButton(boxRoi, 'Select ROI', self.select_ROIs)
        tabProcess.addTab(groupboxRoi, 'ROI')

        # binning
        groupboxBin = QGroupBox('Binning')
        boxBin = QVBoxLayout()
        groupboxBin.setLayout(boxBin)

        self.bin_factor = Settings('Bin factor',
                                   dtype=int,
                                   initial=2,
                                   vmin=1,
                                   vmax=500,
                                   layout=boxBin,
                                   write_function=self.set_preprocessing)
        self.addButton(boxBin, 'Bin Stack', self.bin_stack)
        tabProcess.addTab(groupboxBin, 'Binning')

        # -Log
        groupboxLog = QGroupBox('Log')
        boxLog = QVBoxLayout()
        groupboxLog.setLayout(boxLog)

        self.addButton(boxLog, '-Log', self.calcLog)
        tabProcess.addTab(groupboxLog, 'Log')

        # Adding tabs to the layout
        slayout.addWidget(tabProcess)
        slayout.addWidget(logger.widget)

    def addButton(self, layout: QWidget,
                  button_name: str,
                  call_function: callable) -> None:
        """
        Add a button to the layout.

        Args:
            layout (QWidget): The layout to which the button will be added.
            button_name (str): The name of the button.
            call_function (callable): The function to be called when the button
                is clicked.

        Returns:
            None
        """
        button = QPushButton(button_name)
        button.clicked.connect(call_function)
        layout.addWidget(button)


if __name__ == '__main__':
    import napari

    viewer = napari.Viewer()
    optWidget = PreprocessingnWidget(viewer)
    viewer.window.add_dock_widget(optWidget, name='OPT Preprocessing')
    # warnings.filterwarnings('ignore')
    if DEBUG:
        import glob
        # load example data from data folder
        viewer.open('src_all/sample_data/corr_hot.tiff', name='bad')
        viewer.open('src_all/sample_data/dark_field.tiff', name='dark')
        viewer.open('src_all/sample_data/flat_field.tiff', name='bright')

        # open OPT stack
        viewer.open(glob.glob('src_all/sample_data/16*'),
                    stack=True,
                    name='OPT data')
        # set image layer to OPT data
        optWidget.image_layer_select.value = viewer.layers['OPT data']
        optWidget.bad_layer_select.value = viewer.layers['bad']
        optWidget.dark_layer_select.value = viewer.layers['dark']
        optWidget.bright_layer_select.value = viewer.layers['bright']
    napari.run()
