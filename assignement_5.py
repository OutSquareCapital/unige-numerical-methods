import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import itertools
    import operator
    from collections.abc import Iterable

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from numpy.typing import NDArray


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `1.1`
    """)
    return


@app.cell
def _():
    def polynomial_p(x: NDArray) -> NDArray:
        return (x - 2) ** 5

    def expanded(x: NDArray) -> NDArray:
        return x**5 - 10 * x**4 + 40 * x**3 - 80 * x**2 + 80 * x - 32

    x1 = np.linspace(1.998, 2.002, 41)
    p1 = polynomial_p(x1)
    f1 = expanded(x1)
    return expanded, f1, p1, polynomial_p, x1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `1.2`
    """)
    return


@app.cell
def _(f1, p1, x1):
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(x1, p1, color="tab:blue", label=r"$p(x)=(x-2)^5$")
    ax1.plot(x1, f1, color="tab:red", label=r"$f(x)$ (expanded)")
    ax1.set_title("p(x) vs f(x) near x = 2")
    ax1.legend()
    fig1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Observations

    - The expanded form has a much higher variance than the "simple" form.
    - At the extremes, `p` tend to  go down the lower the `x` input is, and inversely the higher it is. This is not really observable on `f`, at least visually.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `1.3`
    """)
    return


@app.cell(hide_code=True)
def _(expanded, polynomial_p):
    x2 = np.linspace(1000.998, 1001.002, 41)
    diff = expanded(x2) - polynomial_p(x2)
    diff
    return diff, x2


@app.cell
def _(diff, x2):
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    ax2.plot(x2, diff, "-", marker="o", color="tab:green")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x) - p(x)")
    fig2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `2`
    """)
    return


@app.cell
def _():
    START: int = 1
    STOP: int = 31

    def to_array(it: Iterable[float]) -> NDArray:
        return np.fromiter(it, dtype=float, count=STOP)

    # u_n = u_{n-1} / 4, starting from u_0 = 1
    _r = range(START, STOP)
    _u_n = itertools.accumulate(_r, lambda prev, _: prev / 4, initial=1.0)
    u_n = to_array(_u_n)

    # Sn = S_{n-1} + u_n, S_0 = 0
    _arr = u_n[START:]
    _accumulated_sum = itertools.accumulate(_arr, operator.add, initial=0.0)
    s_n = to_array(_accumulated_sum)

    # Rn = (1 - 3 S_n) / u_n
    r_n = (1 - 3 * s_n) / u_n
    return START, r_n, s_n, u_n


@app.cell
def _(START: int, r_n, s_n, u_n):
    # n = 1..30: skip index 0 (initial values)
    n = np.arange(START, len(u_n))

    fig3, ax3 = plt.subplots(figsize=(16, 6))

    ax3.plot(n, r_n[START:], "-", marker="o", color="tab:red", label=r"$R_n$")
    ax3.plot(n, s_n[START:], "-", marker="s", color="tab:blue", label=r"$S_n$")
    ax3.plot(
        n, u_n[START:], "-", marker="^", color="tab:green", label=r"$u_n$"
    )
    ax3.set_xlabel("n")
    ax3.legend()
    fig3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `3.1`
    """)
    return


@app.cell
def _():
    N = 8000
    THEORETICAL_VAR = 4
    RNG = np.random.default_rng(1)
    y = 2 * RNG.standard_normal(N) + 1e9

    def one_pass_variance(a: NDArray) -> float:
        n = len(a)
        return 1 / (n - 1) * (np.sum(a**2) - 1 / n * np.sum(a) ** 2)

    def relative_deviation(variance: float) -> float:
        return abs(variance - THEORETICAL_VAR) / THEORETICAL_VAR

    var_one_pass = one_pass_variance(y)
    e_rel_one_pass = relative_deviation(var_one_pass)
    print(f"One-pass variance: {var_one_pass}")
    print(f"Relative deviation: {e_rel_one_pass}")
    return one_pass_variance, relative_deviation, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `3.2`
    """)
    return


@app.cell
def _(one_pass_variance, relative_deviation, y):
    K = np.arange(11)
    shifts = 10.0**K
    _it = (one_pass_variance(y - d) for d in shifts)
    shifted_variances = np.fromiter(_it, dtype=float, count=len(K))
    # Using polars for convenient table display purposes.
    pl.DataFrame(
        {
            "k": K,
            "d": shifts,
            "variance": shifted_variances,
            "e_rel": relative_deviation(shifted_variances),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `3.3`
    """)
    return


@app.cell
def _(relative_deviation, y):
    def two_pass_variance(a: NDArray) -> float:
        n = len(a)
        mean = 1 / n * np.sum(a)
        return 1 / (n - 1) * np.sum((a - mean) ** 2)

    var_two_pass = two_pass_variance(y)
    print(f"Two-pass variance: {var_two_pass}")
    print(f"Relative deviation: {relative_deviation(var_two_pass)}")
    return


if __name__ == "__main__":
    app.run()
