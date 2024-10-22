import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qtpy.QtWidgets import (
    QDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class PlotDialog(QDialog):
    """
    Create a pop-up widget with the OPT time execution
    statistical plots. Timings are collected during the last run
    OPT scan. The plots show the relevant statistics spent
    on particular tasks during the overall experiment,
    as well as per OPT step.
    """

    def __init__(self, parent, report: dict, canvas: str) -> None:
        # super(PlotDialog, self).__init__(parent)
        super().__init__(parent)
        self.mainWidget = QWidget()
        self.canvas = canvas
        layout = QVBoxLayout(self.mainWidget)
        if self.canvas == "IntCorrCanvas":
            self.canvas = IntCorrCanvas(
                report, self.mainWidget, width=300, height=300
            )
        elif self.canvas == "FlBleachCanvas":
            self.canvas = FlBleachCanvas(
                report, self.mainWidget, width=300, height=300
            )
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class IntCorrCanvas(FigureCanvas):
    def __init__(self, data_dict, parent=None, width=300, height=300):
        """Plot of the report

        Args:
            corr_dict (dict): report data dictionary
            parent (_type_, optional): parent class. Defaults to None.
            width (int, optional): width of the plot in pixels.
                Defaults to 300.
            height (int, optional): height of the plot in pixels.
                Defaults to 300.
        """
        fig = Figure(figsize=(width, height))
        self.ax1 = fig.add_subplot()

        self.createFigure(data_dict)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)

    def createFigure(self, data_dict: dict) -> None:
        """Create report plot.

        Args:
            data_dict (dict): Intensity correction dictionary.
        """
        my_cmap = mpl.colormaps.get_cmap("viridis")
        colors = my_cmap(np.linspace(0, 1, 2))
        self.ax1.plot(
            data_dict["stack_orig_int"], label="Original", color=colors[0]
        )

        self.ax1.plot(
            data_dict["stack_corr_int"], label="Corrected", color=colors[1]
        )

        self.ax1.set_xlabel("OPT step number")
        self.ax1.set_ylabel("Intensity")
        self.ax1.legend()


class FlBleachCanvas(FigureCanvas):
    def __init__(self, data_dict, parent=None, width=300, height=300):
        """Plot of the report

        Args:
            corr_dict (dict): report data dictionary
            parent (_type_, optional): parent class. Defaults to None.
            width (int, optional): width of the plot in pixels.
                Defaults to 300.
            height (int, optional): height of the plot in pixels.
                Defaults to 300.
        """
        self.fig = Figure(figsize=(width, height))
        self.ax1 = self.fig.add_subplot()

        self.createFigure(data_dict)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)

    def createFigure(self, data_dict: dict) -> None:
        """Create report plot.

        Args:
            data_dict (dict): Intensity correction dictionary.
        """
        my_cmap = mpl.colormaps.get_cmap("Greys")
        im = self.ax1.imshow(
            data_dict["corr_factors"].T,
            cmap=my_cmap,
            aspect="auto",
        )

        # add colorbar
        divider = make_axes_locatable(self.ax1)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        self.fig.colorbar(im, cax=cax, orientation="vertical")

        self.ax1.set_title("Normalized bleaching correction factors")
        self.ax1.set_ylabel("Camera row index")
        self.ax1.set_xlabel("Angle step index")
