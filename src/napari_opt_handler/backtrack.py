import numpy as np
from napari.layers import Image
from napari.utils import notifications
import warnings
# from enum import Enum

from dataclasses import dataclass, field

@dataclass
class Backtrack:
    """
    Supports one undo operation. Only for inplace operations.
    Can break if:
        You operate on more datasets in parallel
    """

    raw_data: np.ndarray = None

    # indices refer always to raw data
    roi_def: tuple = field(default_factory=tuple)

    # operation, data, roi_def, bin_factor
    history_item: dict = field(default_factory=dict)
    inplace: bool = True
    track: bool = False

    def set_settings(self, inplace_value: bool, track_value: bool):
        """Global settings for inplace operations and tracking

        Args:
            inplace_value (bool): if operation are inplace (saving RAM)
            track_value (bool): Track enables reverting the inplace operations
        """

        # If values are not boolean, warn and return, no change to default
        if not isinstance(inplace_value, bool) or not isinstance(track_value, bool):
            warnings.warn('Boolean values expected.')
            return
        self.inplace = inplace_value
        self.track = track_value

    def update_history(self, image: Image, data_dict: dict) -> np.ndarray:
        """Updates history if tracking and inplace operations are selected.

        Args:
            image (Image): napari image .data attribute is the np.ndarray image
            data_dict (dict): metadata and data for the operation to register

        Raises:
            AttributeError: In case unknown operation passed

        Returns:
            np.ndarray: to be displayed in viewer.Image layer.
        """
        # for the first operation, store original as raw data
        if self.raw_data is None:
            self.raw_data = image.data

        # if no tracking, no update of history and return new image
        if self._update_compatible() is False:
            return data_dict['data']

        # DP: not that necessary check, can be removed I think
        if data_dict['operation'] not in [
            'roi', 'bin', 'corrInt', 'corrBP', 'corrDB', 'log',
             ]:
            raise AttributeError('Unknown operation to update history.')

        # compatible with update, I put old data to history item
        # and update current parameters in
        # old keys may be hanging, so delete them first
        if self.history_item != dict():
            self.history_item = dict()
        self.history_item['operation'] = data_dict['operation']
        self.history_item['data'] = image.data

        # for ROI selection
        if self.history_item['operation'] == 'roi':
            # set current roi_def to history item (first time they are ()),
            # and update roi_def after
            self.history_item['roi_def'] = self.roi_def
            self.update_roi_pars(data_dict['roi_def'])

        # for binning operation
        elif self.history_item['operation'] == 'bin':
            self.history_item['bin_factor'] = data_dict['bin_factor']
            # I need to update the roi pars too.
            # Not sure about this now
            # self.update_roi_pars(data_dict['roi_def'])

        # for intensity correction
        elif self.history_item['operation'] == 'corrInt':
            self.history_item['rect_dim'] = data_dict['rect_dim']

        # for bad pixel correction
        elif self.history_item['operation'] == 'corrBP':
            self.history_item['mode'] = data_dict['mode']
            self.history_item['hot_pxs'] = data_dict['hot_pxs']
            self.history_item['dead_pxs'] = data_dict['dead_pxs']

        # for log and dark/bright operation nothing to do
        else:
            pass

        # return new image data
        return data_dict['data']

    # DP, this should be checked upon Qt widget values
    def _update_compatible(self) -> bool:
        """Will proceed to update only if inplace and tracking
        are True.

        Returns:
            bool: if history update is going to run.
        """
        if self.inplace and self.track:
            return True
        else:
            return False

    def undo(self) -> np.ndarray:
        """Performs the actual undo operation. If history item
        exists, it identifies, which operation needs to be reverted
        to update the parameters. Image data are updated from the
        history dictionary too.

        Raises:
            ValueError: No history item to revert to.
            ValueError: Unsupported operation in the history

        Returns:
            np.ndarray: reverted image data
        """
        if self.history_item == dict():
            raise ValueError('No State to revert to.')

        # this is completely useless I think
        if self.history_item['operation'] == 'roi':
            notifications.show_info('Reverting ROI selection')
            self.roi_def = self.history_item['roi_def']

        elif self.history_item['operation'] == 'bin':
            # what about binning of the correction files?
            # TODO: corrections need to run before binning. Tha is necessary for the bad pixels
            # but not for dark and bright field. Could be fixed, or need to be imposed or at least
            # raised as warnings
            notifications.show_info('Reverting binning.')
            self.bin_factor = self.history_item['bin_factor']

        elif self.history_item['operation'] == 'corrInt':
            notifications.show_info('Reverting intensity correction.')
            self.rect_dim = self.history_item['rect_dim']

        elif self.history_item['operation'] == 'corrBP':
            notifications.show_info('Reverting Bad Pixel correction.')
            self.mode = self.history_item['mode']
            self.hot_pxs = self.history_item['hot_pxs']
            self.dead_pxs = self.history_item['dead_pxs']

        elif self.history_item['operation'] == 'log':
            notifications.show_info('Reverting log.')

        elif self.history_item['operation'] == 'corrDB':
            notifications.show_info('Reverting dark/bright correction.')

        else:
            raise ValueError('Unsupported operation')

        # resetting history dictionary, because only 1 operation can be tracked
        data = self.history_item.pop('data')
        operation = self.history_item.pop('operation')
        self.history_item = dict()
        return data, operation

    def revert_to_raw(self):
        self.history_item = dict()
        return self.raw_data

    # TODO: Do I really need this since I do always only one undo?
    def update_roi_pars(self, roi_pars):
        print('updating roi_params, should run only once.')
        if self.roi_def == ():
            self.roi_def = roi_pars
        else:
            # TODO: DP This indexing is awful
            # ULy, height, ULx, width
            i1, i2, i3, i4 = self.roi_def
            j1, j2, j3, j4 = roi_pars
            self.roi_def = (
                i1 + j1, i1 + j1 + j2,
                i3 + j3, i3 + j3 + j4,
            )
