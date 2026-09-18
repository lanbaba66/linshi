import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import sklearn
import streamlit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    x = np.arange(10).reshape(-1, 1)
    y = 2 * x.ravel() + 1
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)

    df = pd.DataFrame({"x": x.ravel(), "y": y, "pred": pred})
    _ = px.line(df, x="x", y=["y", "pred"])

    fig, ax = plt.subplots()
    ax.plot(df["x"], df["y"])
    plt.close(fig)

    print("Python:", sys.version.split()[0])
    print("streamlit:", streamlit.__version__)
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("plotly:", plotly.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("MAE:", mean_absolute_error(y, pred))
    print("R2:", r2_score(y, pred))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
