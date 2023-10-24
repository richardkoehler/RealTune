"""Module for offline data streams."""
import math
import os

import numpy as np
import pandas as pd

from . import generator, io, stream_abc

_PathLike = str | os.PathLike


class _OfflineStream(stream_abc.PNStream):
    def _add_labels(
        self, features: pd.DataFrame, data: np.ndarray
    ) -> pd.DataFrame:
        """Add resampled labels to features if there are target channels."""
        if self.nm_channels["target"].sum() > 0:
            features = io.add_labels(
                features=features,
                settings=self.settings,
                nm_channels=self.nm_channels,
                raw_arr_data=data,
                fs=self.sfreq,
            )
        return features

    def _add_timestamp(
        self, feature_series: pd.Series, cnt_samples: int
    ) -> pd.Series:
        """Add time stamp in ms.

        Due to normalization run_analysis needs to keep track of the counted
        samples. These are accessed here for time conversion.
        """
        feature_series["time"] = cnt_samples * 1000 / self.sfreq

        if self.verbose:
            print(
                str(np.round(feature_series["time"] / 1000, 2))
                + " seconds of data processed"
            )

        return feature_series

    def _handle_data(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        names_expected = self.nm_channels["name"].to_list()

        if isinstance(data, np.ndarray):
            if not len(names_expected) == data.shape[0]:
                raise ValueError(
                    "If data is passed as an array, the first dimension must"
                    " match the number of channel names in `nm_channels`. Got:"
                    f" Data columns: {data.shape[0]}, nm_channels.name:"
                    f" {len(names_expected)}."
                )
            return data
        names_data = data.columns.to_list()
        if not (
            len(names_expected) == len(names_data)
            and sorted(names_expected) == sorted(names_data)
        ):
            raise ValueError(
                "If data is passed as a DataFrame, the"
                "columns must match the channel names in `nm_channels`. Got:"
                f"Data columns: {names_data}, nm_channels.name: {names_data}."
            )
        return data.to_numpy()

    def _run_offline(
        self,
        data: np.ndarray,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
    ) -> pd.DataFrame:
        sfreq = math.floor(self.sfreq)
        sfreq_feat = self.settings["sampling_rate_features_hz"]
        batch_window = self.settings["segment_length_features_ms"]
        batch_size = np.ceil(batch_window / 1000 * sfreq).astype(int)
        sample_steps = np.ceil(sfreq / sfreq_feat).astype(int)

        gen = generator.raw_data_generator(data, batch_size, sample_steps)
        features = []
        for data_batch, cnt_samples in gen:
            feature_series = self.run_analysis.process(data_batch)
            feature_series = self._add_timestamp(feature_series, cnt_samples)
            features.append(feature_series)

        feature_df = pd.DataFrame(features)
        feature_df = self._add_labels(features=feature_df, data=data)

        if out_path_root is not None:
            self.save_after_stream(out_path_root, folder_name, feature_df)

        return feature_df


class Stream(_OfflineStream):
    def run(
        self,
        data: np.ndarray | pd.DataFrame,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
    ) -> pd.DataFrame:
        """Call run function for offline stream.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame
            shape (n_channels, n_time)
        out_path_root : _PathLike | None, optional
            Full path to store estimated features, by default None
            If None, data is simply returned and not saved
        folder_name : str, optional
             folder output name, commonly subject or run name, by default "sub"

        Returns
        -------
        pd.DataFrame
            feature DataFrame
        """

        data = self._handle_data(data)

        return self._run_offline(data, out_path_root, folder_name)
