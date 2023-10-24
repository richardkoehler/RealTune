from typing import Iterator

import numpy as np


def raw_data_generator_old(
    data: np.ndarray,
    settings: dict,
    sfreq: int,
) -> Iterator[tuple[np.ndarray, int]]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with sfreq_new.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
        sfreq: int
        sfreq_new: int
        offset_time: int | float
    Returns
    -------
        np.array: new batch for run function of full segment length shape
        int: sample count
    """
    sfreq_new = settings["sampling_rate_features_hz"]
    offset_time = settings["segment_length_features_ms"]
    offset_start = np.ceil(offset_time / 1000 * sfreq).astype(int)
    min_batch_size = sfreq / sfreq_new

    cnt_fsnew = 0
    for cnt in range(data.shape[1]):
        if cnt < offset_start:
            cnt_fsnew += 1
            continue

        cnt_fsnew += 1
        if cnt_fsnew >= min_batch_size:
            cnt_fsnew = 0
            yield data[:, cnt - offset_start : cnt], cnt


def raw_data_generator_intermed(
    data: np.ndarray,
    settings: dict,
    sfreq: int,
) -> Iterator[tuple[np.ndarray, int]]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with sfreq_new.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
        sfreq: int
        sfreq_new: int
        offset_time: int | float
    Returns
    -------
        np.array: new batch for run function of full segment length shape
        int: sample count
    """
    sfreq_new = settings["sampling_rate_features_hz"]
    offset_time = settings["segment_length_features_ms"]
    offset_start = np.ceil(offset_time / 1000 * sfreq).astype(int)
    batch_size = np.ceil(sfreq / sfreq_new).astype(int)

    for cnt in range(offset_start, data.shape[1], batch_size):
        yield data[:, cnt - offset_start : cnt], cnt


def raw_data_generator(
    data: np.ndarray,
    batch_size: int,
    sample_steps: int,
) -> Iterator[tuple[np.ndarray, int]]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with sfreq_new.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
        sfreq: int
        sfreq_new: int
        offset_time: int | float
    Returns
    -------
        np.array: new batch for run function of full segment length shape
        int: sample count
    """
    for cnt in range(batch_size, data.shape[1], sample_steps):
        yield data[:, cnt - batch_size : cnt], cnt


if __name__ == "__main__":
    import time

    settings = {
        "sampling_rate_features_hz": 10,
        "segment_length_features_ms": 1000,
    }
    sfreq = 4000
    duration = 600
    n_channels = 4

    data = np.random.rand(n_channels, duration * sfreq)
    print(f"{data.shape=}")
    gen = raw_data_generator_old(data, settings, sfreq)
    start = time.perf_counter()
    cnt_1 = 0
    for batch_1 in gen:
        if cnt_1 == 0:
            print(f"{batch_1[0].shape = }")
        cnt_1 += 1
    print(f"{time.perf_counter() - start:.4f} seconds elapsed. {cnt_1 = }")

    # sfreq = 40000
    # data = np.random.rand(n_channels, duration * sfreq)
    gen = raw_data_generator_intermed(data, settings, sfreq)
    start = time.perf_counter()
    cnt_2 = 0
    for batch_2 in gen:
        if cnt_2 == 0:
            print(f"{batch_2[0].shape = }")
        cnt_2 += 1
    print(f"{time.perf_counter() - start:.4f} seconds elapsed. {cnt_2 = }")

    assert cnt_1 == cnt_2
    assert np.array_equal(batch_1[0], batch_2[0])

    sfreq_feat = settings["sampling_rate_features_hz"]
    batch_window = settings["segment_length_features_ms"]
    batch_size = np.ceil(batch_window / 1000 * sfreq).astype(int)
    sample_steps = np.ceil(sfreq / sfreq_feat).astype(int)
    gen = raw_data_generator(data, batch_size, sample_steps)
    start = time.perf_counter()
    cnt_3 = 0
    for batch_3 in gen:
        if cnt_3 == 0:
            print(f"{batch_3[0].shape = }")
        cnt_3 += 1
    print(f"{time.perf_counter() - start:.4f} seconds elapsed. {cnt_3 = }")

    assert cnt_1 == cnt_3
    assert np.array_equal(batch_1[0], batch_3[0])
