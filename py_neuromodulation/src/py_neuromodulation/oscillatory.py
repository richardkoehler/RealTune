import math
from typing import Iterable

import numpy as np
from scipy import fft

from . import kalmanfilter, features_abc


class OscillatoryFeature(features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.KF_dict = {}

        self.f_ranges_dict = settings["frequency_ranges_hz"]
        self.fband_names = list(settings["frequency_ranges_hz"].keys())
        self.f_ranges = list(settings["frequency_ranges_hz"].values())

    @staticmethod
    def test_settings_osc(
        s: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
        osc_feature_name: str,
    ):
        assert (
            fb[0] < sfreq / 2 and fb[1] < sfreq / 2
            for fb in s["frequency_ranges_hz"].values()
        ), (
            "the frequency band ranges need to be smaller than the nyquist frequency"
            f"got sfreq = {sfreq} and fband ranges {s['frequency_ranges_hz']}"
        )

        if osc_feature_name != "bandpass_filter_settings":
            assert isinstance(
                s[osc_feature_name]["windowlength_ms"], int
            ), f"windowlength_ms needs to be type int, got {s[osc_feature_name]['windowlength_ms']}"
        else:
            for seg_length in s[osc_feature_name][
                "segment_lengths_ms"
            ].values():
                assert isinstance(
                    seg_length, int
                ), f"segment length has to be type int, got {seg_length}"
        assert isinstance(
            s[osc_feature_name]["log_transform"], bool
        ), f"log_transform needs to be type bool, got {s[osc_feature_name]['log_transform']}"
        assert isinstance(
            s[osc_feature_name]["kalman_filter"], bool
        ), f"kalman_filter needs to be type bool, got {s[osc_feature_name]['kalman_filter']}"

        if s[osc_feature_name]["kalman_filter"] is True:
            kalmanfilter.test_kf_settings(s, ch_names, sfreq)

        assert isinstance(s["frequency_ranges_hz"], dict)

        assert (
            isinstance(value, list)
            for value in s["frequency_ranges_hz"].values()
        )
        assert (len(value) == 2 for value in s["frequency_ranges_hz"].values())

        assert (
            isinstance(value[0], list)
            for value in s["frequency_ranges_hz"].values()
        )

        assert (
            len(value[0]) == 2 for value in s["frequency_ranges_hz"].values()
        )

        assert (
            isinstance(value[1], (float, int))
            for value in s["frequency_ranges_hz"].values()
        )

    def init_KF(self, feature: str) -> None:
        for f_band in self.s["kalman_filter_settings"]["frequency_bands"]:
            for channel in self.ch_names:
                self.KF_dict[
                    "_".join([channel, feature, f_band])
                ] = kalmanfilter.define_KF(
                    self.s["kalman_filter_settings"]["Tp"],
                    self.s["kalman_filter_settings"]["sigma_w"],
                    self.s["kalman_filter_settings"]["sigma_v"],
                )

    def update_KF(self, feature_calc: float, KF_name: str) -> float:
        if KF_name in self.KF_dict:
            self.KF_dict[KF_name].predict()
            self.KF_dict[KF_name].update(feature_calc)
            feature_calc = self.KF_dict[KF_name].x[0]
        return feature_calc


class FFT(OscillatoryFeature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        super().__init__(settings, ch_names, sfreq)
        if self.s["fft_settings"]["kalman_filter"]:
            self.init_KF("fft")

        if self.s["fft_settings"]["log_transform"]:
            self.log_transform = True
        else:
            self.log_transform = False

        window_ms = self.s["fft_settings"]["windowlength_ms"]
        self.window_samples = int(-np.floor(window_ms / 1000 * sfreq))
        self.fft_size = (
            int(np.floor(self.sfreq))
            if window_ms < 1000
            else self.window_samples
        )
        freqs = fft.rfftfreq(self.fft_size, 1 / self.sfreq)

        self.padding_length = None
        self.pad_a = 0
        self.pad_b = 0

        self.feature_params = []
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband, f_range in self.f_ranges_dict.items():
                idx_range = np.where(
                    (freqs >= f_range[0]) & (freqs <= f_range[1])
                )[0]
                feature_name = "_".join([ch_name, "fft", fband])
                self.feature_params.append((ch_idx, feature_name, idx_range))

    @staticmethod
    def test_settings(s: dict, ch_names: Iterable[str], sfreq: int | float):
        OscillatoryFeature.test_settings_osc(
            s, ch_names, sfreq, "fft_settings"
        )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        data = data[:, self.window_samples :]
        if not self.padding_length:
            self.padding_length = int(self.sfreq) - data.shape[1]
            if self.padding_length % 2:
                self.pad_a = math.ceil(self.padding_length / 2)
                self.pad_b = math.floor(self.padding_length / 2)
            else:
                self.pad_a = self.pad_b = self.padding_length // 2
        data = np.pad(data, ((0, 0), (self.pad_a, self.pad_b)), "reflect")
        Z = np.abs(fft.rfft(data))
        for ch_idx, feature_name, idx_range in self.feature_params:
            Z_ch = Z[ch_idx, idx_range]
            feature_calc = np.mean(Z_ch)

            if self.log_transform:
                feature_calc = np.log(feature_calc)

            if self.KF_dict:
                feature_calc = self.update_KF(feature_calc, feature_name)

            features_compute[feature_name] = feature_calc
        return features_compute
