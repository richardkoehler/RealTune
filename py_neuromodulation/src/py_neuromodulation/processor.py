"""This module contains the class to process a given batch of data."""
import math
import os
from enum import Enum
from time import time
from typing import Protocol, Type

import numpy as np
import pandas as pd

from . import features, filter, io, normalization, rereference, resample

_PathLike = str | os.PathLike


class Preprocessor(Protocol):
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    def test_settings(self, settings: dict):
        ...


_PREPROCESSING_CONSTRUCTORS = [
    "notch_filter",
    "re_referencing",
    "raw_normalization",
    "raw_resample",
]


class GRIDS(Enum):
    """Definition of possible projection grid types"""

    CORTEX = "cortex"
    SUBCORTEX = "subcortex"


class DataProcessor:
    def __init__(
        self,
        sfreq: int | float,
        settings: dict | _PathLike,
        nm_channels: pd.DataFrame | _PathLike,
        line_noise: int | float | None = None,
        path_grids: _PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize data processing class.

        Parameters
        ----------
        settings : dict
            dictionary of settings such as "seglengths" or "frequencyranges"
        verbose : boolean
            if True, print out signal processed and computation time
        """
        self.settings = self._load_settings(settings)
        self.nm_channels = self._load_nm_channels(nm_channels)

        self.sfreq_features = self.settings["sampling_rate_features_hz"]
        self._sfreq_raw_orig = sfreq
        self.sfreq_raw = math.floor(sfreq)
        self.line_noise = line_noise
        self.path_grids = path_grids
        self.verbose = verbose

        self.features_previous = None

        (self.ch_names_used, _, self.feature_idx, _) = self._get_ch_info()
        self.preprocessors: list[Preprocessor] = []
        for preprocessing_method in self.settings["preprocessing"]:
            settings_str = f"{preprocessing_method}_settings"
            match preprocessing_method:
                case "raw_resampling":
                    preprocessor = resample.Resampler(
                        sfreq=self.sfreq_raw, **self.settings[settings_str]
                    )
                    self.sfreq_raw = preprocessor.sfreq_new
                    self.preprocessors.append(preprocessor)
                case "notch_filter":
                    preprocessor = filter.NotchFilter(
                        sfreq=self.sfreq_raw,
                        line_noise=self.line_noise,
                        **self.settings.get(settings_str, {}),
                    )
                    self.preprocessors.append(preprocessor)
                case "re_referencing":
                    preprocessor = rereference.ReReferencer(
                        sfreq=self.sfreq_raw,
                        nm_channels=self.nm_channels,
                    )
                    self.preprocessors.append(preprocessor)
                case "raw_normalization":
                    preprocessor = normalization.RawNormalizer(
                        sfreq=self.sfreq_raw,
                        sampling_rate_features_hz=self.sfreq_features,
                        **self.settings.get(settings_str, {}),
                    )
                    self.preprocessors.append(preprocessor)
                case _:
                    raise ValueError(
                        "Invalid preprocessing method. Must be one of"
                        f" {_PREPROCESSING_CONSTRUCTORS}. Got"
                        f" {preprocessing_method}"
                    )

        if self.settings["postprocessing"]["feature_normalization"]:
            settings_str = "feature_normalization_settings"
            self.feature_normalizer = normalization.FeatureNormalizer(
                sampling_rate_features_hz=self.sfreq_features,
                **self.settings.get(settings_str, {}),
            )

        self.features = features.Features(
            s=self.settings,
            ch_names=self.ch_names_used,
            sfreq=self.sfreq_raw,
        )

        self.cnt_samples = 0

    @staticmethod
    def _add_coordinates(coord_names: list[str], coord_list: list) -> dict:
        """Write cortical and subcortical coordinate information in joint dictionary

        Parameters
        ----------
        coord_names : list[str]
            list of coordinate names
        coord_list : list
            list of list of 3D coordinates

        Returns
        -------
        dict with (sub)cortex_left and (sub)cortex_right ch_names and positions
        """

        def is_left_coord(val: int | float, coord_region: str) -> bool:
            if coord_region.split("_")[1] == "left":
                return val < 0
            return val > 0

        coords = {}

        for coord_region in [
            coord_loc + "_" + lat
            for coord_loc in ["cortex", "subcortex"]
            for lat in ["left", "right"]
        ]:
            coords[coord_region] = {}

            ch_type = (
                "ECOG" if "cortex" == coord_region.split("_")[0] else "LFP"
            )

            coords[coord_region]["ch_names"] = [
                coord_name
                for coord_name, ch in zip(coord_names, coord_list)
                if is_left_coord(ch[0], coord_region)
                and (ch_type in coord_name)
            ]

            # multiply by 1000 to get m instead of mm
            positions = []
            for coord, coord_name in zip(coord_list, coord_names):
                if is_left_coord(coord[0], coord_region) and (
                    ch_type in coord_name
                ):
                    positions.append(coord)
            positions = np.array(positions, dtype=np.float64) * 1000
            coords[coord_region]["positions"] = positions

        return coords

    def _get_ch_info(
        self,
    ) -> tuple[list[str], list[str], list[int], np.ndarray]:
        """Get used feature and label info from nm_channels"""
        nm_channels = self.nm_channels
        ch_names_used = nm_channels[nm_channels["used"] == 1][
            "new_name"
        ].tolist()
        ch_types_used = nm_channels[nm_channels["used"] == 1]["type"].tolist()

        # used channels for feature estimation
        feature_idx = np.where(nm_channels["used"] & ~nm_channels["target"])[
            0
        ].tolist()

        # If multiple targets exist, select only the first
        label_idx = np.where(nm_channels["target"] == 1)[0]

        return ch_names_used, ch_types_used, feature_idx, label_idx

    @staticmethod
    def _load_nm_channels(
        nm_channels: pd.DataFrame | _PathLike,
    ) -> pd.DataFrame:
        if not isinstance(nm_channels, pd.DataFrame):
            return io.load_nm_channels(nm_channels)
        return nm_channels

    @staticmethod
    def _load_settings(settings: dict | _PathLike) -> dict:
        if not isinstance(settings, dict):
            return io.read_settings(str(settings))
        return settings

    def process(self, data: np.ndarray) -> pd.Series:
        """Given a new data batch, calculate and return features.

        Parameters
        ----------
        data : np.ndarray
            Current batch of raw data

        Returns
        -------
        pandas Series
            Features calculated from current data
        """
        start_time = time()

        for processor in self.preprocessors:
            data = processor.process(data)

        data = data[self.feature_idx, :]

        # calculate features
        features_dict = self.features.estimate_features(data)
        features_values = np.array(
            list(features_dict.values()), dtype=np.float64
        )

        # normalize features
        if self.settings["postprocessing"]["feature_normalization"]:
            features_values = self.feature_normalizer.process(features_values)

        features_current = pd.Series(
            data=features_values,
            index=list(features_dict.keys()),
            dtype=np.float64,
        )

        if self.verbose is True:
            print(
                "Last batch took: "
                + str(np.round(time() - start_time, 2))
                + " seconds"
            )

        return features_current

    def save_sidecar(
        self,
        out_path_root: _PathLike,
        folder_name: str,
        additional_args: dict | None = None,
    ) -> None:
        """Save sidecar incuding fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'.
        """
        sidecar = {
            "original_fs": self._sfreq_raw_orig,
            "final_fs": self.sfreq_raw,
            "sfreq": self.sfreq_features,
            "ch_names": self.ch_names_used,
        }
        if additional_args is not None:
            sidecar = sidecar | additional_args

        io.save_sidecar(sidecar, out_path_root, folder_name)

    def save_settings(
        self, out_path_root: _PathLike, folder_name: str
    ) -> None:
        io.save_settings(self.settings, out_path_root, folder_name)

    def save_nm_channels(
        self, out_path_root: _PathLike, folder_name: str
    ) -> None:
        io.save_nm_channels(self.nm_channels, out_path_root, folder_name)

    def save_features(
        self,
        out_path_root: _PathLike,
        folder_name: str,
        feature_arr: pd.DataFrame,
    ) -> None:
        io.save_features(feature_arr, out_path_root, folder_name)
