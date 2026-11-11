import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `1.1`
    """)
    return


@app.cell
def _():
    def polynomial_p(x: float) -> float:
        return (x - 2) ** 5

    def expanded(x: float) -> float:
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
    - The tendency of `p` to , at the extremes, to go down the lower the `x` input is, and inversely the higher it is, is not really observable on `f`, at least visually.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercise `1.3`
    """)
    return


@app.cell
def _(expanded, polynomial_p):
    x2 = np.linspace(1000.998, 1001.002, 41)
    diff = expanded(x2) - polynomial_p(x2)

    fig2, ax2 = plt.subplots(figsize=(16, 6))
    ax2.plot(x2, diff, "o-", color="tab:green")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x) - p(x)")
    fig2
    return


if __name__ == "__main__":
    app.run()
