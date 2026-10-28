import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    from typing import Final

    import numpy as np


@app.cell
def _():
    SEED: Final[int] = 0
    INSPECTION_PROB: Final[float] = 0.05
    """Inspection probability per day."""

    rng = np.random.default_rng(SEED)

    rng.uniform()
    return


if __name__ == "__main__":
    app.run()
