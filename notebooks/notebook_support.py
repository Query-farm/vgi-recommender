"""Validate the slate size and configure deterministic worker thread counts."""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display


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


def prose(text: str) -> None:
    display(Markdown(text))


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.dpi": 145,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8EDE9",
            "text.color": "#243B2F",
            "axes.labelcolor": "#405047",
            "text.parse_math": False,
            "legend.frameon": False,
        },
    )


def chart(fig, title: str, subtitle: str, source: str) -> None:
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=17, fontweight="semibold")
    fig.text(0.08, 0.90, subtitle, ha="left", va="top", fontsize=10, color="#58685F")
    fig.tight_layout(rect=(0.02, 0.11, 1, 0.82))
    fig.text(0.08, 0.03, source, fontsize=8, color="#6C7D73")
    plt.show()
