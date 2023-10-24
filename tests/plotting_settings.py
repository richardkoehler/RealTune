"""Initialize settings for plotting."""
import os
import re
from typing import Union

import cycler
from enum import Enum
import matplotlib as mpl
from matplotlib import figure
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

PALETTE_WJN_2022 = np.array(
    (
        (55 / 255, 110 / 255, 180 / 255),  # blue - Med. OFF Stim. OFF
        (113 / 255, 188 / 255, 173 / 255),  # green - Med. ON Stim. OFF
        (223 / 255, 74 / 255, 74 / 255),  # red - Med. OFF Stim. ON
        (125 / 255, 125 / 255, 125 / 255),  # medium grey - STN
        (150 / 255, 150 / 255, 150 / 255),  # medium grey - Parietal Cortex
        (215 / 255, 178 / 255, 66 / 255),  # orange - Sensory Cortex
        (100 / 255, 100 / 255, 100 / 255),  # dark grey - Motor Cortex
        (215 / 255, 213 / 255, 203 / 255),  # medium/light grey
        (234 / 255, 234 / 255, 234 / 255),  # light grey
        # (64 / 255, 64 / 255, 64 / 255),
    )
)


class Color(Enum):
    ECOG_MEDOFF = (55 / 255, 110 / 255, 180 / 255)  # blue - Med. OFF Stim. OFF
    ECOG_MEDON = (113 / 255, 188 / 255, 173 / 255)  # green - Med. ON Stim. OFF
    ECOG_STIMON = (223 / 255, 74 / 255, 74 / 255)  # red - Med. OFF Stim. ON
    STN = (125 / 255, 125 / 255, 125 / 255)  # medium grey - STN
    PARIETAL = (
        150 / 255,
        150 / 255,
        150 / 255,
    )  # medium grey - Parietal Cortex
    SENSORY = (215 / 255, 178 / 255, 66 / 255)  # orange - Sensory Cortex
    MOTOR = (
        215 / 255,
        213 / 255,
        203 / 255,
    )  # medium/light grey - Motor Cortex


def activate() -> None:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 7

    mpl.rcParams["pdf.fonttype"] = "TrueType"
    mpl.rcParams["svg.fonttype"] = "none"

    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0
    mpl.rcParams["savefig.transparent"] = False

    mpl.rcParams["figure.figsize"] = [6.4, 4.8]

    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=PALETTE_WJN_2022)
    mpl.rcParams["axes.xmargin"] = 0

    mpl.rcParams["legend.title_fontsize"] = mpl.rcParams["legend.fontsize"]

    mpl.rcParams["lines.linewidth"] = 1


def medoffvson(show: bool = False) -> None:
    colors = [Color.ECOG_MEDOFF.value, Color.ECOG_MEDON.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def stimoffvson(show: bool = False) -> None:
    colors = [Color.ECOG_MEDOFF.value, Color.ECOG_STIMON.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def save_fig(fig: figure.Figure, outpath: Union[str, os.PathLike]) -> None:
    outpath = str(outpath)
    fig.savefig(outpath)  # , bbox_inches="tight")
    if outpath.endswith(".svg"):
        with open(outpath, "r", encoding="utf-8") as f:
            svg_text = f.read()
        patched_svg = patch_affinity_svg(svg_text)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(patched_svg)


def patch_affinity_svg(svg_text: str) -> str:
    """Patch Matplotlib SVG so that it can be read by Affinity Designer."""
    matches = [
        x
        for x in re.finditer(
            'font:( [0-9][0-9]?[0-9]?[0-9]?)? ([0-9.]+)px ([^;"]+)[";]',
            svg_text,
        )
    ]
    if not matches:
        return svg_text
    svg_pieces = [svg_text[: matches[0].start()]]
    for i, match in enumerate(matches):
        # Change "font" style property to separate "font-size" and
        # "font-family" properties because Affinity ignores "font".
        group = match.groups()
        if len(group) == 2:
            font_weight, font_size_px, font_family = match.groups()
            new_font_style = (
                f"font-size: {float(font_size_px):.1f}px; "
                f"font-family: {font_family}"
            )
        else:
            font_weight, font_size_px, font_family = match.groups()
            if font_weight is not None:
                new_font_style = (
                    f"font-weight: {font_weight}; "
                    f"font-size: {float(font_size_px):.1f}px; "
                    f"font-family: {font_family}"
                )
            else:
                new_font_style = (
                    f"font-size: {float(font_size_px):.1f}px; "
                    f"font-family: {font_family}"
                )
        svg_pieces.append(new_font_style)
        if i < len(matches) - 1:
            svg_pieces.append(
                svg_text[match.end() - 1 : matches[i + 1].start()]
            )
        else:
            svg_pieces.append(svg_text[match.end() - 1 :])
    return "".join(svg_pieces)


if __name__ == "__main__":
    activate()
    sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.show(block=True)
