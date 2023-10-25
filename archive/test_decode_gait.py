"""Test decoding gait from GPi LFP data."""
import pathlib

import mne
import mne_bids
import numpy as np
import pandas as pd
import plotting_settings

# import py_neuromodulation as pn
# from py_neuromodulation.generator import raw_data_generator
# import realtune
import scipy.io
import sklearn.linear_model
import sklearn.model_selection

# import time
from matplotlib import pyplot as plt


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


def add_squared_channel(
    raw: mne.io.BaseRaw, event_id: dict, ch_name: str, inplace: bool = False
) -> mne.io.BaseRaw:
    """Create squared data (0s and 1s) from events and add to Raw object.

    Parameters
    ----------
    raw : MNE Raw object
        The MNE Raw object for this function to modify.
    event_id : dict | callable() | None | ‘auto’
        event_id (see MNE documentation) 'defining the annotations to be chosen
        from your Raw object. ONLY pass annotation names that should be used to
        generate the squared data'.
    ch_name : str
        Name for the squared channel to be added.
    inplace : bool. Default: False
        Set to True if Raw object should be modified in place.

    Returns
    -------
    raw : MNE Raw object
        The Raw object containing the added squared channel.
    """
    events, event_id = mne.events_from_annotations(raw, event_id)
    events_ids = events[:, 0]
    data_squared = np.zeros((1, raw.n_times))
    for i in np.arange(0, len(events_ids), 2):
        data_squared[0, events_ids[i] : events_ids[i + 1]] = 1.0  # * 1e-6

    info = mne.create_info(
        ch_names=[ch_name], ch_types=["misc"], sfreq=raw.info["sfreq"]
    )
    raw_squared = mne.io.RawArray(data_squared, info)
    raw_squared.set_meas_date(raw.info["meas_date"])
    raw_squared.info["line_freq"] = 50

    if not inplace:
        raw = raw.copy()
    if not raw.preload:
        raw.load_data()
    raw.add_channels([raw_squared], force_update_info=True)
    return raw


def preprocess(in_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    files = list(in_dir.glob("*.mat"))
    for file in files:
        # file = in_dir / f"DW{sub}.mat"
        sub = file.name[:4]
        struct = scipy.io.loadmat(file)
        data = struct["data"]
        bad_segments = struct["bad_segments"]
        tmin = bad_segments[1, 0]
        tmax = bad_segments[0, 1]
        sfreq = struct["sfreq"][0, 0]
        # event_desc = {1: "start_standing", 2: "start_walking"}
        start_standing: np.ndarray = struct["start_standing"].squeeze()
        end_standing: np.ndarray = struct["end_standing"].squeeze()
        duration_stand = end_standing - start_standing
        if sub in ["DW04", "DW09"]:
            duration_walk = np.array([tmax - end_standing])
        else:
            end_standing = end_standing[:-1]
            duration_walk = start_standing[1:] - end_standing
        annot_start = mne.Annotations(
            onset=start_standing,
            duration=duration_stand,
            description="start_standing",
        )
        annot_walk = mne.Annotations(
            onset=end_standing,
            duration=duration_walk,
            description="start_walking",
        )
        annotations = annot_start.__add__(annot_walk)
        ch_names = [entry[0] for entry in struct["ch_names"].squeeze()]
        ch_idx = {
            i: ch.split("_seeg")[0]
            for i, ch in enumerate(ch_names)
            if ch.endswith("seeg")
        }
        print(*ch_idx.items())

        data_dbs = data[np.array(list(ch_idx.keys()))]
        info = mne.create_info(
            ch_names=list(ch_idx.values()),
            sfreq=sfreq,
            ch_types="dbs",
        )
        raw = mne.io.RawArray(data_dbs, info, verbose="ERROR")
        raw.reorder_channels(sorted(raw.ch_names))
        raw.set_annotations(annotations)
        # add_squared_channel(
        #     raw=raw,
        #     event_id={ev: i for i, ev in event_desc.items()},
        #     ch_name="SQUARED_STANDING",
        #     inplace=True,
        # )
        raw.crop(tmin=tmin, tmax=tmax)
        # raw.plot(scalings="auto", block=True)
        bids_path = mne_bids.BIDSPath(
            subject=sub,
            session="LfpFollowUp02",
            task="Walking",
            acquisition="StimOff",
            run="01",
            datatype="ieeg",
            extension=".vhdr",
            root=out_dir,
        )
        mne_bids.write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            format="BrainVision",
            overwrite=True,
            allow_preload=True,
            verbose="ERROR",
        )
        # for ind, item in event_desc.items():
        #     event: np.ndarray = struct[item]
        #     event = event.squeeze().astype(np.int64) * sfreq
        #     event = np.expand_dims(event, axis=(1))
        #     ev_id = np.zeros((event.shape[0], 2), dtype=float)
        #     ev_id[:, 1] = ind
        #     event = np.concatenate([event, ev_id], axis=1)
        #     event_list.append(event)
        # duration = event_list[1][:, 0] - event_list[0][:, 0]
        # event_list[0][:, 1] = duration
        # events = np.concatenate(event_list, axis=0)
        # annotations = mne.annotations_from_events(
        #     events=events, sfreq=sfreq, event_desc=event_desc
        # )


def get_power_spectra(in_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    sides = ("Left", "Right")
    for file in in_dir.rglob("*.vhdr"):
        bids_path = mne_bids.get_bids_path_from_fname(file, verbose="ERROR")
        raw = mne_bids.read_raw_bids(bids_path, verbose="ERROR")
        # raw.plot(block=True, scalings="auto")
        chs_by_side = {}
        for side in sides:
            match = side[0]
            chs = [ch for ch in raw.ch_names if ch.endswith(match)]
            chs_by_side[side] = chs
        idx_stand, idx_walk = [], []
        for i, descr in enumerate(raw.annotations.description):
            if descr == "start_standing":
                idx_stand.append(i)
            elif descr == "start_walking":
                idx_walk.append(i)
            else:
                pass
        annot_stand = raw.annotations.copy()
        annot_stand.delete(idx_walk)
        annot_walk = raw.annotations.copy()
        annot_walk.delete(idx_stand)
        raws_stand = raw.crop_by_annotations(
            annotations=annot_stand, verbose="ERROR"
        )
        raws_walk = raw.crop_by_annotations(
            annotations=annot_walk, verbose=None
        )
        psds: dict[str, mne.time_frequency.EpochsSpectrum] = {}
        for descr, raws in (("Standing", raws_stand), ("Walking", raws_walk)):
            psd = psd_from_raw(raws, raw.info)
            psds[descr] = psd
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        for side, axs in zip(sides, axes, strict=True):
            for ax, (descr, psd) in zip(axs, psds.items(), strict=True):
                psd.plot(
                    picks=chs_by_side[side],
                    average=False,
                    dB=True,
                    amplitude="auto",
                    xscale="linear",
                    ci="sd",
                    ci_alpha=0.3,
                    color="black",
                    alpha=None,
                    spatial_colors=True,
                    sphere=None,
                    exclude=(),
                    axes=ax,
                    show=False,
                )
                ax.set_title(f"{side} Hem. - {descr}")
        fig.suptitle(f"Subject {bids_path.subject}")
        fig.tight_layout()
        basename = f"{file.stem}_psd.png"
        plotting_settings.save_fig(fig, out_dir / basename)
        # plt.show(block=True)

    # mne.time_frequency.tfr_morlet(
    #     inst,
    #     freqs,
    #     n_cycles,
    #     use_fft=False,
    #     return_itc=True,
    #     decim=1,
    #     n_jobs=None,
    #     picks=None,
    #     zero_mean=True,
    #     average=True,
    #     output="power",
    #     verbose=None,
    # )
    # raw.plot(block=False, scalings="auto")
    # sfreq_feat = 10  # Hz
    # batch_window = 1000  # ms
    # batch_size = np.ceil(batch_window / 1000 * sfreq).astype(int)
    # sample_steps = np.ceil(sfreq / sfreq_feat).astype(int)
    # data = raw.get_data()
    # gen = raw_data_generator(data, batch_size, sample_steps)
    # for data_batch in gen:
    #     print(data_batch[0].shape)
    #     break
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


def psd_from_raw(
    raws: list[mne.io.BaseRaw], info: mne.Info
) -> mne.time_frequency.EpochsSpectrum:
    epochs_list = []
    for raw_single in raws:
        # raw_single.plot(scalings="auto", block=True)
        epochs_single = mne.make_fixed_length_epochs(
            raw_single,
            duration=1.0,
            overlap=0.5,
            id=1,
            verbose="ERROR",
        )
        epochs_list.append(epochs_single.get_data())
        # epochs.plot(n_epochs=1, block=True)
    epochs_data = np.concatenate(epochs_list)
    epochs = mne.EpochsArray(epochs_data, info, events=None, verbose="ERROR")
    psd = epochs.compute_psd(
        method="welch",
        fmin=3,
        fmax=50,
        picks="dbs",
        proj=False,
        n_jobs=1,
        verbose=None,
    )
    return psd


if __name__ == "__main__":
    mne.viz.set_browser_backend("qt")
    plotting_settings.activate()
    root = pathlib.Path(__file__).parents[1] / "data" / "gait_dystonia"
    source_dir = root / "sourcedata"
    raw_dir = root / "rawdata"
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    # preprocess(source_dir, raw_dir)
    get_power_spectra(raw_dir, plot_dir)
    print("Done.")
