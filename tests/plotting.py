"""Module for plotting functions."""
import pathlib
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import cluster
import numpy as np
import pandas as pd
import permutation
import scipy.stats
import seaborn as sns
import timeseries
from matplotlib import axes, collections, figure
from matplotlib import pyplot as plt
from statannotations import Annotator
from statannotations.stats import StatTest


def boxplot_all_conds(
    data: pd.DataFrame,
    x: str,
    y: str,
    outpath: str | Path | None = None,
    order: Sequence | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> figure.Figure:
    """Plot performance as violinplot."""
    if order is None:
        order = list(data[x].unique())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax = sns.swarmplot(
        orient="h",
        x=y,
        y=x,
        order=order,
        data=data,
        color="black",
        alpha=0.9,
        dodge=True,
        s=3,
        ax=ax,
    )

    ax = sns.boxplot(
        orient="h",
        x=y,
        y=x,
        order=order,
        data=data,
        width=0.5,
        ax=ax,
        showfliers=False,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.yaxis.set_tick_params(length=0)
    ax.set_ylabel("")
    if title is not None:
        y_coord = 1.02
        ax.set_title(title, y=y_coord)

    if outpath is not None:
        fig.savefig(str(outpath), bbox_inches="tight")
    if show:
        plt.show(block=True)
    return fig


def boxplot_results(
    data: pd.DataFrame,
    x: str,
    y: str,
    outpath: str | Path | None = None,
    hue: str | None = None,
    order: Sequence | None = None,
    hue_order: Sequence | None = None,
    stat_test: str | Callable | None = "Permutation",
    alpha: float = 0.05,
    add_lines: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | Literal["auto"] = "auto",
    show: bool = True,
) -> figure.Figure:
    """Plot performance as violinplot."""
    if order is None:
        order = list(data[x].unique())

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figsize == "auto":
        if hue:
            if hue_order is None:
                hue_order = list(data[hue].unique())
            hue_factor = len(hue_order)
        else:
            hue_factor = 1
        figsize = (1 + (1.1 * len(order) * hue_factor), 4.8)

    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    ax = sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        color="black",
        alpha=0.9,
        dodge=True,
        s=3,
        ax=ax,
    )

    # ax = sns.violinplot(
    #     x=x,
    #     y=y,
    #     hue=hue,
    #     order=order,
    #     hue_order=hue_order,
    #     data=data,
    #     inner="box",
    #     width=0.9,
    #     alpha=1.0,
    #     cut=0.2,
    #     ax=ax,
    # )
    ax = sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        width=0.5,
        ax=ax,
        showfliers=False,
    )

    if stat_test:
        _add_stats(
            ax=ax,
            data=data,
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            stat_test=stat_test,
            alpha=alpha,
            location="outside",
        )

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            label.replace(" ", "\n") for label in labels[: len(labels) // 2]
        ]
        _ = plt.legend(
            handles[: len(handles) // 2],
            new_labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            title=hue,
            labelspacing=0.7,
        )

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_xlabels = [xtick.replace(" ", "\n") for xtick in xlabels]
    ax.set_xticklabels(new_xlabels)
    if title is not None:
        if stat_test:
            y_coord = 1.15
        else:
            y_coord = 1.02
        ax.set_title(title, y=y_coord)

    if add_lines:
        _add_lines(
            ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines
        )

    if outpath is not None:
        fig.savefig(str(outpath), bbox_inches="tight")
    if show:
        plt.show(block=True)
    return fig


def _add_lines(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable[str],
    add_lines: str,
) -> None:
    """Add lines connecting single dots"""
    data = data.sort_values(  # type: ignore
        by=x, key=lambda k: k.map({item: i for i, item in enumerate(order)})
    )
    lines = [
        [[i, n] for i, n in enumerate(group)]
        for _, group in data.groupby([add_lines], sort=False)[y]
    ]
    ax.add_collection(
        collections.LineCollection(
            lines, colors="grey", linewidths=0.5  # 1.0, alpha=0.5
        )  # type: ignore
    )


def _add_stats(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable,
    hue: str | None,
    hue_order: Iterable | None,
    stat_test: str | StatTest.StatTest,
    alpha: float,
    location: str = "inside",
) -> None:
    """Perform statistical test and annotate graph."""
    if not hue:
        pairs = list(combinations(order, 2))
    else:
        pairs = [
            list(combinations(list(product([item], hue_order)), 2))
            for item in order
        ]
        pairs = [item for sublist in pairs for item in sublist]

    if stat_test == "Permutation":
        stat_test = StatTest.StatTest(
            func=_permutation_wrapper,
            n_perm=10000,
            alpha=alpha,
            test_long_name="Permutation Test",
            test_short_name="Perm.",
            stat_name="Effect Size",
        )
    annotator = Annotator.Annotator(
        ax=ax,
        pairs=pairs,
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
        plot="violinplot",
    )
    annotator.configure(
        alpha=alpha,
        test=stat_test,
        text_format="simple",
        loc=location,
        color="black",
        line_width=1,
        pvalue_format={
            "pvalue_thresholds": [
                [1e-6, "0.000001"],
                [1e-5, "0.00001"],
                [1e-4, "0.0001"],
                [1e-3, "0.001"],
                [1e-2, "0.01"],
                [5e-2, "0.05"],
            ],
            "fontsize": 7,
            "text_format": "simple",
            "pvalue_format_string": "{:.3f}",
            "show_test_name": False,
        },
    )
    annotator.apply_and_annotate()


def _permutation_wrapper(x, y, n_perm) -> tuple:
    """Wrapper for statannotations to convert pandas series to numpy array."""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    return permutation.permutation_twosample(data_a=x, data_b=y, n_perm=n_perm)


def lineplot_compare(
    x_1: np.ndarray,
    x_2: np.ndarray,
    times: np.ndarray,
    data_labels: Sequence,
    alpha: float = 0.05,
    n_perm: int = 1000,
    correction_method: str = "cluster",
    two_tailed: bool = False,
    paired_x1x2: bool = False,
    ax: axes.Axes | None = None,
    y_lims: Sequence | None = None,
    colors: Sequence[tuple] | None = None,
    x_label: str = "Time (s)",
    y_label: str | None = None,
    print_n: bool = True,
    legend: bool = True,
    show: bool = True,
    outpath: pathlib.Path | str | None = None,
) -> tuple[figure.Figure | None, list[tuple[float, float]]]:
    """Plot comparison of continuous prediction arrays."""
    fig = None
    if not ax:
        fig, axis = plt.subplots(1, 1)
    else:
        axis = ax
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if colors is None:
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        colors = (color, (125 / 255, 125 / 255, 125 / 255))
    for i, data in enumerate([x_1, x_2]):
        axis.plot(
            times,
            data.mean(axis=1),
            label=data_labels[i],
            color=colors[i],
        )
        axis.fill_between(
            times,
            data.mean(axis=1) - scipy.stats.sem(data, axis=1),
            data.mean(axis=1) + scipy.stats.sem(data, axis=1),
            alpha=0.5,
            color=colors[i],
        )
    cluster_times = _pval_correction_lineplot(
        ax=axis,
        x=x_1,
        y=x_2,
        times=times,
        alpha=alpha,
        n_perm=n_perm,
        correction_method=correction_method,
        two_tailed=two_tailed,
        min_cluster_size=2,
        onesample_xy=paired_x1x2,
    )
    axis.set_title(f"{data_labels[0]} vs. {data_labels[1]}")
    axis.set_xlabel(x_label)
    if legend:
        axis.legend(frameon=False)
    if y_label:
        axis.set_ylabel(y_label)
    if y_lims:
        axis.set_ylim(y_lims[0], y_lims[1])
    if print_n:
        axis.text(
            0.9,
            1.0,
            f"N = {x_1.shape[1]}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
            weight="bold",
            color=tuple(np.array([1 / 255, 1 / 255, 1 / 255]) * 50),
        )
    if fig is not None:
        fig.tight_layout()
        if outpath:
            fig.savefig(str(outpath), bbox_inches="tight")
    if show:
        plt.show(block=True)
    return fig, cluster_times


def _pval_correction_lineplot(
    ax: axes.Axes,
    x: np.ndarray,
    y: int | float | np.ndarray,
    times: np.ndarray,
    alpha: float,
    correction_method: str,
    n_perm: int,
    two_tailed: bool,
    onesample_xy: bool,
    one_tailed_test: Literal["larger"] | Literal["smaller"] = "larger",
    min_cluster_size: int = 2,
) -> list[tuple[float, float]]:
    """Perform p-value correction for singe lineplot."""
    cluster_times = []
    if onesample_xy:
        data_a = x - y
        data_b = 0.0
    else:
        data_a = x
        data_b = y

    if not two_tailed and one_tailed_test == "smaller":
        data_a_stat = data_a * -1
    else:
        data_a_stat = data_a

    if correction_method == "cluster":
        _, clusters_ind = cluster.cluster_analysis_1d(
            data_a=data_a_stat.T,
            data_b=data_b,
            alpha=alpha,
            n_perm=n_perm,
            only_max_cluster=False,
            two_tailed=two_tailed,
            min_cluster_size=min_cluster_size,
        )
        if len(clusters_ind) == 0:
            return cluster_times
        cluster_count = len(clusters_ind)
        clusters = np.zeros(data_a_stat.shape[0], dtype=np.int32)
        for ind in clusters_ind:
            clusters[ind] = 1
    elif correction_method in ["cluster_pvals", "fdr"]:
        p_vals = timeseries.timeseries_pvals(
            x=data_a_stat, y=data_b, n_perm=n_perm, two_tailed=two_tailed
        )
        clusters, cluster_count = cluster.clusters_from_pvals(
            p_vals=p_vals,
            alpha=alpha,
            correction_method=correction_method,
            n_perm=n_perm,
            min_cluster_size=min_cluster_size,
        )
    else:
        raise ValueError(
            f"Unknown cluster correction method: {correction_method}."
        )

    if cluster_count <= 0:
        print("No clusters found.")
        return cluster_times
    if isinstance(y, (int, float)):
        y_arr = np.ones((x.shape[0], 1))
        y_arr[:, 0] = y
    else:
        y_arr = y
    if onesample_xy:
        x_arr = x
    else:
        x_arr = data_a
    label = f"p ≤ {alpha}"
    x_labels = times.round(2)
    text_annotated = False
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            print("No clusters found.")
            continue
        lims = np.arange(index[0], index[-1] + 1)
        y1 = x_arr.mean(axis=1)[lims]
        y2 = y_arr.mean(axis=1)[lims]
        ax.fill_between(
            x=times[lims],
            y1=y1,
            y2=y2,
            alpha=0.5,
            color="yellow",
            label=label,
        )
        time_0 = x_labels[lims[0]]
        time_1 = x_labels[lims[-1]]
        print(f"Cluster found between {time_0} Hz and" f" {time_1} Hz.")
        label = None  # Avoid printing label multiple times
        if not text_annotated:
            ax.text(
                time_0,
                1.0,
                f"{time_0} Hz",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.get_xaxis_transform(),
                weight="bold",
                color=tuple(np.array([1 / 255, 1 / 255, 1 / 255]) * 50),
            )
            ax.text(
                time_1,
                0.9,
                f"{time_1} Hz",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.get_xaxis_transform(),
                weight="bold",
                color=tuple(np.array([1 / 255, 1 / 255, 1 / 255]) * 50),
            )
            text_annotated = True
        cluster_times.append((time_0, time_1))
    return cluster_times
