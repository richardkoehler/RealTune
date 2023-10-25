"""Test decoding gait from GPi LFP data."""
import pathlib

import fooof
import fooof.analysis
import mne
import mne_bids
import numpy as np
import pandas as pd
import plotting
import plotting_settings
import scipy.io
from matplotlib import figure
from matplotlib import pyplot as plt
from statannotations.stats.StatTest import StatTest


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


def preprocess_walking(in_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    files = list(in_dir.glob("*.mat"))
    for file in files:
        sub = file.name[:4]
        struct = scipy.io.loadmat(file)
        data = struct["data"]
        bad_segments = struct["bad_segments"]
        tmin = bad_segments[1, 0]
        tmax = bad_segments[0, 1]
        sfreq = struct["sfreq"][0, 0]
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


def preprocess_rest(in_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    files = list(in_dir.glob("*.mat"))
    for file in files:
        sub = file.name[:4]
        struct = scipy.io.loadmat(file)
        data = struct["data_rest"]
        sfreq = struct["sfreq"][0, 0]
        ch_names = [entry[0] for entry in struct["ch_names_rest"].squeeze()]
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
        # raw.plot(scalings="auto", block=True)
        bids_path = mne_bids.BIDSPath(
            subject=sub,
            session="LfpFollowUp02",
            task="Rest",
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


def fit_model_iterative(
    freqs: np.ndarray, power_spectrum: np.ndarray, fig, ax
) -> tuple[fooof.FOOOF, figure.Figure]:
    """Fit FOOOF model iteratively."""
    fit_knee: bool = True
    while True:
        if fit_knee:
            print("Fitting model WITH knee component.")
        else:
            print("Fitting model WITHOUT knee component.")
        model = fit_model(freqs, power_spectrum, fit_knee, ax)
        fig.tight_layout()
        fig.canvas.draw()
        redo_fit = get_input_y_n("Try new fit w or w/o knee")
        ax.clear()
        if redo_fit.lower() == "n":
            break
        fit_knee = not fit_knee
    return model, fig


def fit_model(
    freqs: np.ndarray, power_spectrum: np.ndarray, fit_knee: bool, ax
) -> fooof.FOOOF:
    """Fit fooof model."""
    aperiodic_mode = "knee" if fit_knee else "fixed"
    model = fooof.FOOOF(
        peak_width_limits=(2, 20.0),
        max_n_peaks=4,
        min_peak_height=0.0,
        peak_threshold=1.0,
        aperiodic_mode=aperiodic_mode,
        verbose=True,
    )
    model.fit(freqs=freqs, power_spectrum=power_spectrum)
    model.print_results()
    model.plot(ax=ax)
    return model


def get_input_y_n(message: str) -> str:
    """Get ´y` or `n` user input."""
    while True:
        user_input = input(f"{message} (y/n)? ")
        if user_input.lower() in ["y", "n"]:
            break
        print(
            f"Input must be `y` or `n`. Got: {user_input}."
            " Please provide a valid input."
        )
    return user_input


def plot_psd(
    psds: dict[str, mne.time_frequency.EpochsSpectrum],
    subject: str,
    chs_by_side: dict[str, list[str]],
    out_path: pathlib.Path,
) -> None:
    """Plot PSD for hemisphere and condition separately."""
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for (side, chs), axs in zip(chs_by_side.items(), axes, strict=True):
        for ax, (descr, psd) in zip(axs, psds.items(), strict=True):
            psd.plot(
                picks=chs,
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
    fig.suptitle(f"Subject {subject}")
    fig.tight_layout()
    plotting_settings.save_fig(fig, out_path)
    plt.show(block=True)


def plot_fooof(
    psds: dict[str, mne.time_frequency.EpochsSpectrum],
    bids_path: mne_bids.BIDSPath,
    out_dir: pathlib.Path,
) -> list:
    """"""
    fname = bids_path.fpath.stem
    sub = bids_path.subject  # type: ignore
    fig, ax = plt.subplots(1, 1)
    fig.show()
    peak_fits = []
    for descr, psd in psds.items():
        power, freqs = psd.get_data(picks="data", return_freqs=True)
        power = power.mean(axis=0)
        for ch, power_spectrum in zip(psd.ch_names, power, strict=True):
            fig.suptitle(ch)
            model, fig = fit_model_iterative(
                freqs, power_spectrum, fig=fig, ax=ax
            )
            basename = f"{fname}_{ch}_{descr}"
            model.save_report(
                file_name=f"{basename}_model.pdf",
                file_path=str(out_dir),
            )
            peak_fits.append([sub, descr, ch, *model._peak_fit])  # type: ignore
    power_final = pd.DataFrame(
        peak_fits,
        columns=[
            "Subject",
            "Condition",
            "Channel",
            *(freq for freq in freqs),
        ],
    )
    power_final.to_csv(
        out_dir / f"fooof_walking_standing_sub-{sub}.csv", index=False
    )
    plt.close()
    return peak_fits
    # fig.savefig(str(PLOT_DIR / (basename + ".png")))

    # beta_peaks = fooof.analysis.get_band_peak_fm(
    #     model,
    #     band=(13.0, 35.0),
    #     select_highest=False,
    #     attribute="peak_params",
    # )
    # if beta_peaks.ndim == 1:
    #     beta_peaks = np.expand_dims(beta_peaks, axis=0)
    # results.extend(
    #     (
    #         [
    #             sub,
    #             med,
    #             stim,
    #             ch,
    #             peak[0],
    #             peak[1],
    #             peak[2],
    #         ]
    #         for peak in beta_peaks
    #     )
    # )


def get_power_walking(in_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    sides = ("Left", "Right")
    fmin = 3
    fmax = 90
    freqs = np.arange(fmin, fmax + 1, dtype=int)
    for file in in_dir.rglob("*Walking*.vhdr"):
        bids_path = mne_bids.get_bids_path_from_fname(file, verbose="ERROR")
        raw = mne_bids.read_raw_bids(bids_path, verbose="ERROR")
        # raw.plot(block=True, scalings="auto")
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
        for descr, raws in (
            ("Standing", raws_stand),
            ("Walking", raws_walk),
        ):
            psd = psd_from_raw(raws, raw.info, fmin=fmin, fmax=fmax)
            psds[descr] = psd

        power_single = plot_fooof(psds, bids_path=bids_path, out_dir=out_dir)

    power_all = pd.DataFrame(
        power_single,
        columns=[
            "Subject",
            "Condition",
            "Channel",
            *(freq for freq in freqs),
        ],
    )
    power_all.to_csv(out_dir / "fooof_walking_standing.csv", index=False)


def get_power_rest(in_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    fmin = 3
    fmax = 90
    freqs = np.arange(fmin, fmax + 1, dtype=int)
    for file in in_dir.rglob("*Rest*.vhdr"):
        bids_path = mne_bids.get_bids_path_from_fname(file, verbose="ERROR")
        print(bids_path.basename)
        raw = mne_bids.read_raw_bids(bids_path, verbose="ERROR")
        # raw.plot(block=True, scalings="auto")
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=1.0,
            overlap=0.5,
            id=1,
            verbose="ERROR",
        )
        psd = epochs.compute_psd(
            method="multitaper",
            fmin=fmin,
            fmax=fmax,
            picks="dbs",
            proj=False,
            n_jobs=1,
            verbose="ERROR",
        )
        fname = bids_path.fpath.stem
        sub = bids_path.subject  # type: ignore
        fig, ax = plt.subplots(1, 1)
        fig.show()
        peak_fits = []
        descr = "Rest"
        power, freqs = psd.get_data(picks="data", return_freqs=True)
        power = power.mean(axis=0)
        for ch, power_spectrum in zip(psd.ch_names, power, strict=True):
            fig.suptitle(ch)
            model, fig = fit_model_iterative(
                freqs, power_spectrum, fig=fig, ax=ax
            )
            basename = f"{fname}_{ch}_{descr}"
            model.save_report(
                file_name=f"{basename}_model.pdf",
                file_path=str(out_dir),
            )
            peak_fits.append([sub, descr, ch, *model._peak_fit])  # type: ignore
        power_single = pd.DataFrame(
            peak_fits,
            columns=[
                "Subject",
                "Condition",
                "Channel",
                *(freq for freq in freqs),
            ],
        )
        power_single.to_csv(out_dir / f"fooof_rest_sub-{sub}.csv", index=False)


def psd_from_raw(
    raws: list[mne.io.BaseRaw], info: mne.Info, fmin: int = 3, fmax: int = 50
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
        method="multitaper",
        fmin=fmin,
        fmax=fmax,
        picks="dbs",
        proj=False,
        n_jobs=1,
        verbose="ERROR",
    )
    return psd


def lineplot_power(
    plot_dir: pathlib.Path, power: dict[str, np.ndarray], freqs: np.ndarray
) -> None:
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(4, 3),
        squeeze=True,
    )
    _, _ = plotting.lineplot_compare(
        ax=ax,
        x_1=power["Standing"],
        x_2=power["Walking"],
        times=freqs,
        y_lims=None,
        data_labels=list(power.keys()),
        x_label="Frequency [Hz]",
        y_label="Periodic Power [AU]",
        alpha=0.05,
        n_perm=10000,
        correction_method="cluster_pvals",
        two_tailed=True,
        paired_x1x2=True,
        print_n=True,
        outpath=None,
        colors=plt.rcParams["axes.prop_cycle"].by_key()["color"][0:2],
        show=False,
    )
    plotting_settings.save_fig(
        fig, plot_dir / "psd_walking_standing_lineplot.svg"
    )
    plt.show(block=True)


def load_power(
    plot_dir,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    powers = {"Standing": [], "Walking": []}
    freqs = np.arange(3, 91, dtype=int)
    for file in plot_dir.rglob("fooof_walking_standing*.csv"):
        data = pd.read_csv(file)
        for condition, values in powers.items():
            data_cond = data.query(f"Condition == '{condition}'")
            data_cond = data_cond.drop(
                columns=["Subject", "Channel", "Condition"]
            )
            power = data_cond.to_numpy().mean(axis=0)
            values.append(power)
    for file in plot_dir.rglob("fooof_rest*.csv"):
        data = pd.read_csv(file)
        powers["Rest"] = values = []
        data_cond = data.drop(columns=["Subject", "Channel", "Condition"])
        power = data_cond.to_numpy().mean(axis=0)
        values.append(power)
    results: dict[str, np.ndarray] = {}
    for condition, values in powers.items():
        results[condition] = np.stack(values).T
    return results, freqs


def permutation_onesample() -> StatTest:
    """Wrapper for StatTest with permutation one-sample test."""

    def _stat_test(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray):
        # if isinstance(x, pd.Series):
        #     x = x.to_numpy()
        # if isinstance(y, pd.Series):
        #     y = y.to_numpy()
        # diff = x - y
        # return pte_stats.permutation_onesample(
        #     data_a=diff, data_b=0, n_perm=100000, two_tailed=True
        # )
        res = scipy.stats.permutation_test(
            (x - y,),
            np.mean,
            vectorized=True,
            n_resamples=int(1e6),
            permutation_type="samples",
        )
        return res.statistic, res.pvalue

    return StatTest(
        func=_stat_test,
        alpha=0.05,
        test_long_name="Permutation Test",
        test_short_name="Perm. Test",
    )


def boxplot_power(
    plot_dir: pathlib.Path, power: dict[str, np.ndarray], freqs: np.ndarray
) -> None:
    f_bands = {
        "α": [8, 12],
        "Low β": [13, 20],
        "High β": [21, 35],
        "β": [13, 35],
        "γ": [40, 90],
    }
    for fband, flims in f_bands.items():
        fband_str = f"{fband} [{flims[0]}-{flims[1]}Hz]"
        idx = (freqs > flims[0]) & (freqs < flims[1])
        y = "Periodic Power [AU]"
        data_list = []
        for cond, pow in power.items():
            pow_avg = pow[idx].mean(axis=0)
            pow_cond = pd.DataFrame(
                np.stack(
                    [pow_avg, np.arange(len(pow_avg), dtype=int)], axis=0
                ).T,
                columns=[y, "Subject"],
            )
            pow_cond["Condition"] = cond
            data_list.append(pow_cond)
        data = pd.concat(data_list)
        fig = plotting.boxplot_results(
            data=data,
            x="Condition",
            y=y,
            order=["Standing", "Walking"],
            stat_test=permutation_onesample(),
            add_lines="Subject",
            figsize=(1.6, 2.4),
            show=False,
        )
        fig.axes[0].set_title(fband_str, y=1.15)
        fig.tight_layout()
        plotting_settings.save_fig(
            fig, plot_dir / f"psd_walking_standing_{flims[0]}-{flims[1]}Hz.svg"
        )
        plt.show(block=True)


if __name__ == "__main__":
    mne.viz.set_browser_backend("qt")
    plotting_settings.activate()
    plotting_settings.stimoffvson()
    root = pathlib.Path(__file__).parents[1] / "data" / "gait_dystonia"
    source_dir = root / "sourcedata"
    raw_dir = root / "rawdata"
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    # preprocess_rest(source_dir / "rest", raw_dir)
    # get_power_rest(raw_dir, plot_dir)
    power, freqs = load_power(plot_dir)

    # preprocess_walking(source_dir / "walking", raw_dir)
    # get_power_walking(raw_dir, plot_dir)
    # lineplot_power(plot_dir, power, freqs)
    # boxplot_power(plot_dir, power, freqs)
    ...
