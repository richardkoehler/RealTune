"""Test decoding gait from GPi LFP data."""
import pathlib
import time

import mne
import mne_bids
import numpy as np
import pandas as pd
import py_neuromodulation as pn
from py_neuromodulation.generator import raw_data_generator
import realtune
import scipy.io
import sklearn.linear_model
import sklearn.model_selection


class Decoder:
    def __init__(self) -> None:
        pass

    def add_features(self) -> None:
        pass

    def get_features(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        pass


def cross_validate(model, features, labels, groups, cv) -> None:
    sklearn.model_selection.cross_validate(
        estimator=model,
        X=features,
        y=labels,
        groups=groups,
        scoring=None,
        cv=cv,
        verbose=0,
        fit_params=None,
    )
    pass


def test_decode_gait() -> None:
    mne.viz.set_browser_backend("qt")
    root = pathlib.Path(__file__).parents[1] / "data" / "gait_dystonia"
    source_dir = root / "sourcedata"
    raw_dir = root / "rawdata"
    file = source_dir / "DW02_walking_standing_cleaned.mat"
    struct = scipy.io.loadmat(file)
    data = struct["data"]
    times = struct["times"]
    sfreq = 256
    event_list = []
    event_desc = {1: "start_standing", 2: "end_standing"}
    for ind, item in event_desc.items():
        event = struct[item].squeeze() * sfreq
        event = np.expand_dims(event, axis=(1))
        ev_id = np.zeros((event.shape[0], 2), dtype=float)
        ev_id[:, 1] = ind
        event = np.concatenate([event, ev_id], axis=1)
        event_list.append(event)
    events = np.concatenate(event_list, axis=0)
    ch_names = [entry[0] for entry in struct["ch_names"].squeeze()]
    print(*ch_names)
    ch_idx = {i: ch for i, ch in enumerate(ch_names) if ch.endswith("seeg")}
    data_dbs = data[np.array(list(ch_idx.keys()))]
    info = mne.create_info(
        ch_names=list(ch_idx.values()),
        sfreq=256,
        ch_types="dbs",
    )
    raw = mne.io.RawArray(data_dbs, info, verbose=None)
    raw.reorder_channels(sorted(raw.ch_names))
    annotations = mne.annotations_from_events(
        events=events, sfreq=sfreq, event_desc=event_desc
    )
    raw.set_annotations(annotations)
    raw.plot(block=True)
    bids_path = mne_bids.BIDSPath(
        subject=file.name.split("_")[0],
        session="walking",
        task="walking",
        acquisition="StimOff",
        run="01",
        datatype="ieeg",
        extension=".vhdr",
    )
    mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        format="brainvision",
        overwrite=True,
        allow_preload=True,
        verbose="ERROR",
    )
    sfreq_feat = 10  # Hz
    batch_window = 1000  # ms
    batch_size = np.ceil(batch_window / 1000 * sfreq).astype(int)
    sample_steps = np.ceil(sfreq / sfreq_feat).astype(int)
    gen = raw_data_generator(data_dbs, batch_size, sample_steps)
    for data_batch in gen:
        print(data_batch[0].shape)
        break
    #     ...
    # for event in events:
    #     print(event[0])
    # model = sklearn.linear_model.LogisticRegression()
    # processor = pn.DataProcessor(
    #     sfreq=sfreq,
    #     settings=settings,
    #     nm_channels=nm_channels,
    # )
    # decoder = Decoder(model=model)
    # for i in range(12):
    #     timestamp = time.time()
    #     features = processor.process(data=np.random.rand(sfreq, 1))
    #     label = 0 if i < 5 else 1
    #     group = i % 2
    #     decoder.add_features(
    #         features=features,
    #         timestamp=timestamp,
    #         label=label,
    #         group=group,
    #     )
    # features, labels, groups = decoder.get_features()
    # cross_validate(
    #     model=model,
    #     features=features,
    #     labels=labels,
    #     groups=groups,
    #     cv=sklearn.model_selection.LeaveOneGroupOut(),
    # )

    # model.fit(features, labels)
    # features = processor.process(data=np.random.rand(sfreq, 1))
    # prediction = model.predict(features)
    ...


if __name__ == "__main__":
    test_decode_gait()
    print("Done.")
