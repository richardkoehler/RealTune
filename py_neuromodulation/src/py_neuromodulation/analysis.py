from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import io

TARGET_FILTER_STR = {
    "CLEAN",
    "SQUARED_EMG",
    "SQUARED_INTERPOLATED_EMG",
    "SQUARED_ROTAWHEEL",
    "SQUARED_ROTATION",
    "rota_squared",
}


class Feature_Reader:
    feature_dir: str
    feature_list: list[str]
    settings: dict
    sidecar: dict
    sfreq: int
    line_noise: int
    nm_channels: pd.DataFrame
    feature_arr: pd.DataFrame
    ch_names: list[str]
    ch_names_ECOG: list[str]

    def __init__(
        self, feature_dir: str, feature_file: str, binarize_label: bool = True
    ) -> None:
        """Feature_Reader enables analysis methods on top of NM_reader and NM_Decoder

        Parameters
        ----------
        feature_dir : str, optional
            Path to py_neuromodulation estimated feature runs, where each feature is a folder,
        feature_file : str, optional
            specific feature run, if None it is set to the first feature folder in feature_dir
        """
        self.feature_dir = feature_dir
        self.feature_list = io.get_run_list_indir(self.feature_dir)
        if feature_file is None:
            self.feature_file = self.feature_list[0]
        else:
            self.feature_file = feature_file

        FILE_BASENAME = Path(self.feature_file).name  # stem
        PATH_READ_FILE = str(
            Path(self.feature_dir, FILE_BASENAME, FILE_BASENAME)
        )

        self.settings = io.read_settings(PATH_READ_FILE)
        self.sidecar = io.read_sidecar(PATH_READ_FILE)
        self.sfreq = 10  # self.sidecar["sfreq"]
        self.nm_channels = io.read_nm_channels(PATH_READ_FILE)
        self.feature_arr = io.read_features(PATH_READ_FILE)

        self.ch_names = self.nm_channels.new_name
        self.used_chs = list(
            self.nm_channels[
                (self.nm_channels["target"] == 0)
                & (self.nm_channels["used"] == 1)
            ]["new_name"]
        )
        self.ch_names_ECOG = self.nm_channels.query(
            '(type=="ecog") and (used == 1) and (status=="good")'
        ).new_name.to_list()

        if self.nm_channels["target"].sum() > 0:
            self.label_name = self._get_target_ch()
            self.label = self.read_target_ch(
                self.feature_arr,
                self.label_name,
                binarize=binarize_label,
                binarize_th=0.3,
            )

    def _get_target_ch(self) -> str:
        target_names = list(
            self.nm_channels[self.nm_channels["target"] == 1]["name"]
        )
        target_clean = [
            target_name
            for target_name in target_names
            for filter_str in TARGET_FILTER_STR
            if filter_str.lower() in target_name.lower()
        ]

        if len(target_clean) == 0:
            if "ARTIFACT" not in target_names[0]:
                target = target_names[0]
            elif len(target_names) > 1:
                target = target_names[1]
            else:
                target = target_names[0]
        else:
            for target_ in target_clean:
                # try to select contralateral label
                if self.sidecar["sess_right"] is True and "LEFT" in target_:
                    target = target_
                    continue
                elif (
                    self.sidecar["sess_right"] is False and "RIGHT" in target_
                ):
                    target = target_
                    continue
                if target_ == target_clean[-1]:
                    target = target_clean[0]  # set label to last element
        return target

    @staticmethod
    def read_target_ch(
        feature_arr: pd.DataFrame,
        label_name: str,
        binarize: bool = True,
        binarize_th: float = 0.3,
    ) -> None:
        label = np.nan_to_num(np.array(feature_arr[label_name]))
        if binarize:
            label = label > binarize_th
        return label

    @staticmethod
    def filter_features(
        feature_columns: list,
        ch_name: str = None,
        list_feature_keywords: list[str] = None,
    ) -> list:
        """filters read features by ch_name and/or modality

        Parameters
        ----------
        feature_columns : list
            [description]
        ch_name : str, optional
            [description], by default None
        list_feature_keywords : list[str], optional
            list of feature strings that need to be in the columns, by default None

        Returns
        -------
        list
            column list that suffice the ch_name and list_feature_keywords
        """
        features_reverse_order_plotting = {"stft", "fft", "bandpass"}
        if ch_name is not None:
            feature_select = [i for i in list(feature_columns) if ch_name in i]
        else:
            feature_select = feature_columns

        if list_feature_keywords is not None:
            feature_select = [
                f
                for f in feature_select
                if any(x in f for x in list_feature_keywords)
            ]

            if (
                len(
                    [
                        mod
                        for mod in features_reverse_order_plotting
                        if mod in list_feature_keywords
                    ]
                )
                > 0
            ):
                # flip list s.t. theta band is lowest in subsequent plot
                feature_select = feature_select[::-1]

        return feature_select

    def set_target_ch(self, ch_name: str) -> None:
        self.label = ch_name

    def normalize_features(
        self,
    ) -> pd.DataFrame:
        """Normalize feature_arr feature columns

        Returns:
            pd.DataFrame: z-scored feature_arr
        """
        cols_norm = [c for c in self.feature_arr.columns if "time" not in c]
        feature_arr_norm = stats.zscore(self.feature_arr[cols_norm])
        feature_arr_norm["time"] = self.feature_arr["time"]
        return feature_arr_norm


    @staticmethod
    def get_epochs(data, y_, epoch_len, sfreq, threshold=0):
        """Return epoched data.

        Parameters
        ----------
        data : np.ndarray
            array of extracted features of shape (n_samples, n_channels, n_features)
        y_ : np.ndarray
            array of labels e.g. ones for movement and zeros for
            no movement or baseline corr. rotameter data
        sfreq : int/float
            sampling frequency of data
        epoch_len : int
            length of epoch in seconds
        threshold : int/float
            (Optional) threshold to be used for identifying events
            (default=0 for y_tr with only ones
            and zeros)

        Returns
        -------
        epoch_ np.ndarray
            array of epoched ieeg data with shape (epochs,samples,channels,features)
        y_arr np.ndarray
            array of epoched event label data with shape (epochs,samples)
        """

        epoch_lim = int(epoch_len * sfreq)

        ind_mov = np.where(np.diff(np.array(y_ > threshold) * 1) == 1)[0]

        low_limit = ind_mov > epoch_lim / 2
        up_limit = ind_mov < y_.shape[0] - epoch_lim / 2

        ind_mov = ind_mov[low_limit & up_limit]

        epoch_ = np.zeros(
            [ind_mov.shape[0], epoch_lim, data.shape[1], data.shape[2]]
        )

        y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])

        for idx, i in enumerate(ind_mov):
            epoch_[idx, :, :, :] = data[
                i - epoch_lim // 2 : i + epoch_lim // 2, :, :
            ]

            y_arr[idx, :] = y_[i - epoch_lim // 2 : i + epoch_lim // 2]

        return epoch_, y_arr
