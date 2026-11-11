import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import numpy as np
    import marimo as mo


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
