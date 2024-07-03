import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)

from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget, QDialog, QSizePolicy,
)


class PlotDialog(QDialog):
    """
    Create a pop-up widget with the OPT time execution
    statistical plots. Timings are collected during the last run
    OPT scan. The plots show the relevant statistics spent
    on particular tasks during the overall experiment,
    as well as per OPT step.
    """
    def __init__(self, parent, report: dict) -> None:
        super(PlotDialog, self).__init__(parent)
        self.mainWidget = QWidget()
        layout = QVBoxLayout(self.mainWidget)
        self.canvas = IntCorrCanvas(report, self.mainWidget,
                                    width=300, height=300)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class IntCorrCanvas(FigureCanvas):
    def __init__(self, data_dict, parent=None, width=300, height=300):
        """ Plot of the report

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

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def createFigure(self, data_dict: dict) -> None:
        """Create report plot.

        Args:
            data_dict (dict): Intensity correction dictionary.
        """
        my_cmap = mpl.colormaps.get_cmap("viridis")
        colors = my_cmap(np.linspace(0, 1, 2))
        self.ax1.plot(data_dict['stack_orig_int'],
                      label='Original',
                      color=colors[0])

        self.ax1.plot(data_dict['stack_corr_int'],
                      label='Corrected',
                      color=colors[1])

        self.ax1.set_xlabel('OPT step number')
        self.ax1.set_ylabel('Intensity')
        self.ax1.legend()
