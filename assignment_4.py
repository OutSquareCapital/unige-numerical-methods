import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    from typing import Final
    from pyochain import Iter, Some
    import numpy as np


@app.cell
def _():
    SEED: Final[int] = 0
    INSPECTION_PROB: Final[float] = 0.05
    """Inspection probability per day."""

    RNG = np.random.default_rng(SEED)

    def _single_inspection(draw: float) -> int:
        """Exercise `1.1`."""
        inspected = may_inspect(draw)
        branched = 1 if inspected else 0
        assert branched == int(inspected)
        return branched

    def may_inspect(draw: float) -> bool:
        return draw < INSPECTION_PROB

    _single_inspection(RNG.uniform())
    return INSPECTION_PROB, RNG, may_inspect


@app.cell
def _(RNG, may_inspect):
    def many_inspections_looped(days: int = 70) -> int:
        """Exercise `1.2`."""
        return np.array([may_inspect(RNG.uniform()) for _ in range(days)])

    s = many_inspections_looped()
    s.sum()
    return


@app.cell
def _(INSPECTION_PROB: Final[float], RNG):
    def _many_inspections_np(days: int = 70) -> int:
        """Exercise `1.3`."""
        draws = RNG.uniform(size=days)
        return np.sum(draws < INSPECTION_PROB)

    return


if __name__ == "__main__":
    app.run()
