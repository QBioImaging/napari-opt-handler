#!/usr/bin/env python

"""
Corrections Class for the microscopy acquisition

corrections available:
#. Dark-field correction
    * Always applicable and should be always performed
#. Flat-field, ie bright-field correction
    * Important for transmission measurements. Performs badly
        on the floating elements in the FOV
    * Harder to do for the Fluorescence, because of the excitation scatter
        leaking through filters.
    * Flat field can be used to identify dead pixels too
#. bad pixel correction
    * bad pixels obtained from long exposure on the blocked camera
    * Dead pixels form flat field acquisition.
    * Possible to correct as mean of 4 or 8 neighbors
More crucial for the shorter exposure times (less averaging of the source
intensity variations), which depends on which time-scale the light-source
drifts.

Notes:
* Exposure on the dark and bright field corrections must be
the same as the experimental exposure.
"""

import numpy as np
from .utils import img_to_int_type, is_positive, rescale_img
import logging


class Correct():
    """Correcting raw data from 2D array acquisitions. Currently implemented
    corrections are:

    #. Dark-field
    #. Bright-field
    #. bad-pixel correction
    #. Intensity correction.
    """
    def __init__(self, bad: np.ndarray = None, std_mult: float = None,
                 dark: np.ndarray = None, bright: np.ndarray = None,
                 ) -> None:
        """ Initialize the correction class with the correction arrays.

        Args:
            bad (np.array, optional): Bad pixel acquisition. Defaults to None.
            std_mult (float, optional): STD cutoff for outliers (bad pixels).
                Defaults to 7.0 via self.set_std_mult().
            dark (np.array, optional): Dark counts camera acquisition.
                Defaults to None.
            bright (np.array, optional): Bright field correction acquisition.
                Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.bad = bad
        self.hot_pxs = None
        self.dead_pxs = None
        try:
            self.set_std_mult(std_mult)
        except ValueError as e:
            self.logger.warning(e)
            self.std_mult = 7.0

        self.dark = dark
        self.dark_corr = None

        self.bright = bright
        self.bright_corr = None

    def correctBadPxs(self, img: np.array, mode: str = 'n4') -> np.array:
        """Correct hot pixels from its neighbor pixel values. It ignores the
        neighbor pixel if it was identified as hot pixel itself.

        Args:
            img (np.array): image to be corrected.
            mode (str, optional): How to pick neighbors. Defaults to 'n4',
                up, bottom left, right. Other option is n8, which takes the
                diagonal neighbors too.

        Raises:
            IndexError: Raised if the hot_corr array does not match the shape
                of the input img.
            ValueError: invalid mode option

        Returns:
            np.array: Corrected img array
        """
        self.badPxs = set(self.hot_pxs + self.dead_pxs)

        if self.badPxs == []:
            self.logger.info('No hot pixels identified, nothing to correct')
            return img

        # check if the shapes of the correction and image match
        if self.bad.shape != img.shape:
            print(self.bad.shape, img.shape)
            raise IndexError('images do not have the same shape')

        # define neighbors
        if mode == 'n4':
            neighs = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # U, L, R, D
        elif mode == 'n8':
            neighs = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1),
                      ]
        else:
            raise ValueError('Unknown mode option, valid is n4 and n8.')

        ans = img.copy()

        # loop over identified bad pixels and correct
        for badPx in self.badPxs:
            neigh_vals = []
            for neigh in neighs:
                px = np.add(np.array(badPx), np.array(neigh))
                # I can do this because I checked shapes above
                # check if neighbor is out of the image ranges.
                if (0 > px[0] or px[0] >= img.shape[0] or
                        0 > px[1] or px[1] >= img.shape[1]):
                    continue

                # ignore if neighbor is bad pixel
                if tuple(px) in self.badPxs:
                    continue

                neigh_vals.append(img[px[0], px[1]])

            # replace hot pixel with the mean of the neighbors
            self.logger.debug(neigh_vals, neighs, badPx)
            ans[badPx] = int(np.mean(neigh_vals))

        # test for negative values
        is_positive(ans, 'Bad-pixel')

        # cast it on correct dtype
        ans = img_to_int_type(ans, dtype=ans.dtype)
        return ans

    def correct_dark_bright(self, stack: np.ndarray,
                            useDark: bool = True,
                            useBright: bool = False,
                            modality: str = 'Transmission') -> np.ndarray:
        """
        Correct stack of images with dark and bright field images

        Args:
            stack (np.array): Stack of images to be corrected
            useDark (bool, optional): Use dark field correction.
                Defaults to True.
            useBright (bool, optional): Use bright field correction.
                Defaults to False.
            modality (str, optional): Modality of the images.
                Defaults to Transmission.

        Returns:
            np.array: Corrected stack of images
        """
        if modality not in ['Transmission', 'Emission']:
            raise ValueError(
                'Unknown modality option, valid is Transmission and Emission.',
                )

        if useDark is False and useBright is False:
            self.logger.warning('No correction selected.')
            return stack.astype(np.uint16)

        # I do not like not keeping the dtype the same, but float
        # operations do not work otherwise
        data_corr = np.empty(stack.shape,
                             #  dtype=stack.dtype
                             )
        if modality == 'Transmission':
            if useBright and useDark:
                data_corr = ((stack - self.dark) /
                             (self.bright - self.dark))
                # because it is float between 0-1, needs to be rescaled
                # and casted to uint16 check if the image has element
                # greater than 1 or less than 0
                data_corr = self.clip_and_convert_data(data_corr)

            elif useBright and not useDark:
                data_corr = stack / self.bright
                # this is float between 0-1, rescaled and cast to uint16
                # make sure that the image is between 0-1 first
                data_corr = self.clip_and_convert_data(data_corr)

            elif useDark and not useBright:
                data_corr = self.subtract_images(stack, self.dark)
            else:
                raise ValueError('Unknown combination of corrections')

        elif modality == 'Emission':
            # this assumes that dark is contained in the bright
            # because here I am subtracting
            # (stack - dark) - (bright - dark) = stack - bright
            if useBright:
                # make sure accidental floats are handled
                data_corr = self.subtract_images(stack, self.bright)

            elif not useBright:
                data_corr = self.subtract_images(stack, self.dark)
            else:
                raise ValueError('Unknown combination of corrections')

            # get rid of potential negative values
            data_corr = np.clip(data_corr, 0, None).astype(np.uint16)

        return data_corr

    def correct_int(self, img_stack: np.ndarray, mode: str = 'integral',
                    use_bright: bool = True, rect_dim: int = 50,
                    cast_to_int: bool = True) -> np.ndarray:
        """
        Intensity correction over the stack of images which are expected
        to have the same background intensity. It is preferable to
        corrected for dark, bright, and bad pixels first

        Args:
            img_stack (np.array): 3D array of images, third dimension is
                along angles
            mode (str, optional): correction mode, only available is integral
                and integral_bottom. Defaults to 'integral'.
            use_bright (bool, optional): if bright field acquisition is a ref
                to scale images. Defaults to True.
            rect_dim (int, optional): size of rectangles in the corners in
                pixels. Defaults to 50.

        Raises:
            NotImplementedError: Checking available correction modes

        Returns:
            np.ndarray: 3D array of the same shape as the img_stack,
                but intensity corrected
        """
        # check if stack is 3D array
        if img_stack.ndim != 3:
            raise IndexError('Stack has to have three dimensions.')
        # do I want to correct in respect to the bright field
        # basic idea is four corners, integrated
        # second idea, fit a correction plane into the four corners.
        if use_bright is True and self.bright is not None:
            # four corners of the bright
            # this is useless option!!!
            ref = ((self.bright[:rect_dim, :rect_dim]),
                   (self.bright[:rect_dim, -rect_dim:]),
                   (self.bright[-rect_dim:, :rect_dim]),
                   (self.bright[-rect_dim:, -rect_dim:]),
                   )
        else:
            print('Using avg of the corners in the img stack as ref')
            # assuming the stacks 3rd dimension is the right one.
            # mean over steps in the acquisition
            ref = ((np.mean(img_stack[:, :rect_dim, :rect_dim], axis=0)),
                   (np.mean(img_stack[:, :rect_dim, -rect_dim:], axis=0)),
                   (np.mean(img_stack[:, -rect_dim:, :rect_dim], axis=0)),
                   (np.mean(img_stack[:, -rect_dim:, -rect_dim:], axis=0)),
                   )

        print('shape ref:', [k.shape for k in ref])

        # integral takes sum over pixels of interest
        # TODO: This if else structure is cumbersome
        if mode == 'integral':
            # sum of all pixels over all four squares
            # this is one number
            self.ref = np.mean([np.mean(k) for k in ref])
        elif mode == 'integral_bottom':
            self.ref = np.mean([np.mean(ref[2]), np.mean(ref[3])])
        else:
            raise NotImplementedError

        # correct the stack
        corr_stack = np.empty(img_stack.shape, dtype=img_stack.dtype)

        # intensity numbers for img in the stack (sum over ROIs)
        intOrig, intCorr = [], []
        for i, img in enumerate(img_stack):
            # two means are not a clean solution
            # as long as the rectangles ar the same, it is equivalent
            if mode == 'integral':
                img_int = np.mean((np.mean(img[:rect_dim, :rect_dim]),
                                   np.mean(img[:rect_dim, -rect_dim:]),
                                   np.mean(img[-rect_dim:, :rect_dim]),
                                   np.mean(img[-rect_dim:, -rect_dim:]),
                                   ))
            elif mode == 'integral_bottom':
                img_int = np.mean((
                                np.mean(img[-rect_dim:, :rect_dim]),
                                np.mean(img[-rect_dim:, -rect_dim:]),
                ))
            intOrig.append(img_int)
            corr_stack[i] = (img / img_int) * self.ref

            # intensity after correction
            intCorr.append(
                np.mean((np.mean(corr_stack[i][:rect_dim, :rect_dim]),
                         np.mean(corr_stack[i][:rect_dim, -rect_dim:]),
                         np.mean(corr_stack[i][-rect_dim:, :rect_dim]),
                         np.mean(corr_stack[i][-rect_dim:, -rect_dim:]),
                         ))
            )
            print(i, end='\r')

        # stored in order to tract the stability fluctuations.
        intOrig = np.array(intOrig)
        intCorr = np.array(intCorr)

        # test for negative values
        is_positive(corr_stack, 'Intensity')

        # cast it on correct dtype
        if cast_to_int:
            corr_stack = img_to_int_type(corr_stack, dtype=corr_stack.dtype)

        # int correction dictionary report
        intCorrReport = {'mode': mode,
                         'use_bright': use_bright,
                         'rect_dim': rect_dim,
                         'ref': self.ref,
                         'stack_orig_int': intOrig,
                         'stack_corr_int': intCorr,
                         }

        return corr_stack, intCorrReport

    def get_bad_pxs(self, mode: str = 'hot') -> list[tuple[int, int]]:
        """
        Identify bad pixels from the bad array based on the bad
        std_mult factor threshold. Hot pixel has intensity greater than

        mean(img) + std_mult * std(img). Dead pixel has intensity less than
        mean(img) - std_mult * std(img).

        TODO: This method should also also have and option for median method
        Args:
            mode (str, optional): Mode of the bad pixel identification.
                Defaults to 'hot'. Options are 'hot', 'dead' and 'both'.
        """
        if self.bad is None:
            raise ValueError('No bad pixel array provided')

        if mode not in ['hot', 'dead', 'both']:
            raise ValueError(
                'Unknown mode option, valid is hot, dead and both.')

        self.hot_pxs, self.dead_pxs = [], []
        self.mean = np.mean(self.bad, dtype=np.float64)
        self.std = np.std(self.bad, dtype=np.float64)

        if mode == 'hot' or mode == 'both':
            self.maskAbove = np.ma.masked_greater(
                            self.bad,
                            self.mean + self.std_mult * self.std,
                            )

            # if mask did not get any bad pixels, return empty list
            if np.all(self.maskAbove.mask is False):
                self.logger.info('No hot pixels identified')
                self.logger.info(f'{self.mean} + {self.std_mult} * {self.std}')
                self.logger.info(f'{self.bad.max()}, {self.bad.min()}')
            else:
                # iterate over the mask, and append hot pixels to the list
                for row, col in zip(*np.where(self.maskAbove.mask)):
                    self.hot_pxs.append((row, col))

        if mode == 'dead' or mode == 'both':
            self.maskBelow = np.ma.masked_less(
                            self.bad,
                            self.mean - self.std_mult * self.std,
                            )

            # if mask did not get any dead pixels, return empty list
            if np.all(self.maskBelow.mask is False):
                self.logger.info('No dead pixels identified')
            else:
                # iterate over the mask and append dead pixels to the list
                for row, col in zip(*np.where(self.maskBelow.mask)):
                    self.dead_pxs.append((row, col))

    def clip_and_convert_data(self, data_corr: np.ndarray) -> np.ndarray:
        """Clip the data to 0-1 and convert it to uint16

        Args:
            data_corr (np.ndarray): Data to be clipped and converted

        Returns:
            np.ndarray: Clipped and converted data
        """
        if np.amax(data_corr) > 1 or np.amin(data_corr) < 0:
            self.logger.warning('Overflows %.2f, %.2f, clipping to 0-1.',
                                data_corr.min(), data_corr.max(),
                                )
            data_corr = np.clip(data_corr, 0, 1)

        data_corr = (data_corr * 65535).astype(np.uint16)
        return data_corr

    def subtract_images(self, image: np.ndarray,
                        corr: np.ndarray) -> np.ndarray:
        """Subtract correction image from the data image

        Args:
            image (np.ndarray): Data image
            corr (np.ndarray): Correction image

        Returns:
            np.ndarray: Corrected image
        """
        if (not np.issubdtype(image.dtype, np.integer) or
                not np.issubdtype(corr.dtype, np.integer)):
            self.logger.warning('Either data or corr is not np.integer type.')
            data_corr = np.round(
                            (image - corr.clip(None, image)),
                            ).astype(np.uint16)

        else:
            data_corr = image - corr

        return data_corr

    # helper methods
    def set_dark(self, dark: np.ndarray) -> None:
        """Update dark field correction image

        Args:
            dark (np.array): Dark field correction image
        """
        # rescale if necessary
        if np.amax(dark) > 2**16-1 or np.amin(dark) < 0:
            self.logger.warning('Dark field correction image is out of range.')
            self.dark = rescale_img(dark, np.uint16)
        else:
            self.dark = dark

    def set_bright(self, bright: np.ndarray) -> None:
        """Update bright field correction image

        Args:
            bright (np.array): Bright field correction image
        """
        # rescale if necessary
        if np.amax(bright) > 2**16-1 or np.amin(bright) < 0:
            self.logger.warning(
                'Bright field correction image is out of range.')
            self.bright = rescale_img(bright, np.uint16)
        else:
            self.bright = bright

    def set_bad(self, bad: np.ndarray) -> None:
        """Update bad pixel correction image

        Args:
            bad (np.array): Bad pixel correction image
        """
        # rescale if necessary
        if np.amax(bad) > 2**16-1 or np.amin(bad) < 0:
            self.logger.warning('Bad pixel correction image is out of range.')
            self.bad = rescale_img(bad, np.uint16)
        else:
            self.bad = bad

    def set_std_mult(self, std_mult: float = 7.0) -> None:
        """Update std_mult factor for bad pixel correction

        Args:
            std_mult (float): STD cutoff for outliers
        """
        if not np.issubdtype(type(std_mult), np.number):
            std_mult = 7.0
        elif std_mult <= 0:
            raise ValueError('STD multiplier should be positive')

        self.std_mult = float(std_mult)
