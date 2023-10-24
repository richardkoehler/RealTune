"""Module for calculating features."""
import numpy as np

from . import features_abc, oscillatory


class Features:
    features: list[features_abc.Feature] = []

    def __init__(
        self, s: dict, ch_names: list[str], sfreq: int | float
    ) -> None:
        """Class for calculating features."""
        self.features = []

        for feature in s["features"]:
            if s["features"][feature] is False:
                continue
            match feature:
                case "fft":
                    FeatureClass = oscillatory.FFT
                case _:
                    raise ValueError(f"Unknown feature found. Got: {feature}.")

            FeatureClass.test_settings(s, ch_names, sfreq)
            f_obj = FeatureClass(s, ch_names, sfreq)
            self.features.append(f_obj)

        self.s = s
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.fband_names = list(self.s["frequency_ranges_hz"].keys())
        self.f_ranges = list(self.s["frequency_ranges_hz"].values())

    def estimate_features(self, data: np.ndarray) -> dict:
        """Calculate features, as defined in settings.json
        Features are based on bandpower, raw Hjorth parameters and sharp wave
        characteristics.

        Parameters
        ----------
        data (np array) : (channels, time)

        Returns
        -------
        dat (dict) with naming convention:
            channel_method_feature_(f_band)
        """

        # features_compute = {}

        # for feature in self.features:
        #     features_compute = feature.calc_feature(
        #         data,
        #         features_compute,
        #     )

        # return features_compute

        features_ = dict()

        # sequential approach
        for ch, data_ in zip(self.ch_names, data, strict=True):
            features_ = get_fft_features(
                features_=features_,
                s=self.s,
                fs=self.sfreq,
                data=data_,
                ch=ch,
                f_ranges=self.f_ranges,
                f_band_names=self.fband_names,
            )
        return features_


import math
from scipy import fft


def get_fft_features(features_, s, fs, data, ch, f_ranges, f_band_names):
    """Get FFT features for different f_ranges. Data needs to be a batch of 1s length

    Parameters
    ----------
    features_ : dict
        feature dictionary
    s : dict
        settings dict
    fs : int/float
        sampling frequency
    data : np.array
        data for single channel, assumed to be one second
    KF_dict : dict
        Kalmanfilter dictionaries, channel, bandpower and frequency
        band specific
    ch : string
        channel name
    f_ranges : list
        list of list with respective frequency band ranges
    f_band_names : list
        list of frequency band names
    """
    windowlen_sec = (
        s["fft_settings"]["windowlength_ms"] / 1000
    )  # convert from ms to s
    data = data[-int(fs * windowlen_sec) :]

    padding_length = int(fs) - data.shape[0]
    if padding_length % 2:
        pad_a = math.ceil(
            padding_length / 2,
        )
        pad_b = math.floor(padding_length / 2)
    else:
        pad_a = pad_b = padding_length // 2
    data = np.pad(data, (pad_a, pad_b), "reflect")
    # data = np.hstack(
    #     (data, np.zeros(shape=(int(fs) - data.shape[0],)))
    # )  # zero-pad

    Z = np.abs(fft.rfft(data))
    if s["fft_settings"]["log_transform"]:
        Z = np.log(Z)

    f = np.linspace(0, int(fs) // 2, data.shape[0] // 2 + 1)

    for idx_fband, f_range in enumerate(f_ranges):
        fband = f_band_names[idx_fband]
        idx_range = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
        feature_calc = np.mean(Z[idx_range])

        feature_name = "_".join([ch, "fft", fband])
        features_[feature_name] = feature_calc
    return features_
