"""Presentation and deterministic child-worker thread configuration."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from vgi_notebooks import finish


def configure(size):
    if type(size) is not int or not 1 <= size <= 4:
        raise ValueError("SLATE_SIZE must be an integer from 1 through 4")
    keys = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = "1"
    return previous


def cleanup(previous):
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def interaction_ranks(matrix_frame, SLATE_SIZE):
    observed = matrix_frame.pivot(index="learner", columns="item", values="interactions").astype(float)
    ranks = matrix_frame.pivot(index="learner", columns="item", values="rank").astype(float)
    titles = matrix_frame.drop_duplicates("item").sort_values("item").title.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.8))
    sns.heatmap(
        observed,
        mask=observed.isna(),
        annot=True,
        fmt=".0f",
        cmap="Greens",
        vmin=0,
        vmax=3,
        linewidths=0.7,
        linecolor="#E6EBE8",
        cbar_kws={"label": "Interaction count", "ticks": [1, 2, 3]},
        ax=axes[0],
    )
    sns.heatmap(
        ranks,
        mask=ranks.isna(),
        annot=True,
        fmt=".0f",
        cmap="YlGn_r",
        vmin=1,
        vmax=4,
        linewidths=0.7,
        linecolor="#E6EBE8",
        cbar_kws={"label": "Recommendation rank (1 best)", "ticks": [1, 2, 3, 4]},
        ax=axes[1],
    )
    for y, x in zip(*np.where(observed.notna()), strict=True):
        axes[1].text(x + 0.5, y + 0.5, "×", ha="center", va="center", color="#A3ADA6", fontsize=13)
    for ax, title in zip(axes, ["Authored interactions", "Returned recommendations"], strict=True):
        ax.set(title=title, xlabel="", ylabel="Learner")
        ax.set_xticklabels(titles, rotation=48, ha="right", fontsize=9)
    finish(
        fig,
        "Observed interactions and ranked recommendations",
        f"Slate size: {SLATE_SIZE} • 3 latent factors • gray × = already seen • blanks = no entry",
        "Synthetic 8-learner × 8-module corpus • VGI Recommender v0.4.0 · implicit ALS.\n"
        "Counts and ranks have separate scales; unobserved is unknown, not disliked.",
        name="01-interactions-and-ranks",
    )


def slate_exposure(exposure_frame, SLATE_SIZE):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    bars = ax.barh(exposure_frame.title, exposure_frame.appearances, color="#327659", height=0.6)
    ax.bar_label(bars, padding=5)
    ax.invert_yaxis()
    ax.set(
        xlabel="Returned recommendation slates containing the module (count)",
        xlim=(0, max(1, exposure_frame.appearances.max()) + 1.1),
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    finish(
        fig,
        "Returned slates allocate exposure unevenly",
        f"Slate size: {SLATE_SIZE} • {int(exposure_frame.appearances.sum())} total slots "
        "• fixed model and input",
        "Authored interaction matrix • model-generated lists, not observed engagement.\n"
        "Every module is shown, including any with zero appearances · VGI Recommender v0.4.0.",
        name="02-slate-exposure",
    )
