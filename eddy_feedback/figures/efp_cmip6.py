"""
Plot of Eddy-feedback parameter calculated for each ensemble member in CMIP6 historical
simulations, grouped by model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eddy_feedback import get_files_by_model, plotdir


def main():
    months = "DJF"
    years = "1850-2014"

    files_by_model = get_files_by_model("eddy_feedback", months=months, years=years)
    make_plot(files_by_model)

    plt.title("Eddy Feedback Parameter in CMIP6 {} {}".format(years, months))
    plt.savefig(plotdir / "eddy-feedback_cmip6_{}_{}.png".format(months, years))
    plt.show()


def make_plot(files_by_model):
    x = 0
    plt.figure(figsize=(12, 6.75))

    models = []
    model_current = None
    for n, row in files_by_model.iterrows():
        print(row["model"])
        # Increment the x point for each new model
        if row["model"] != model_current:
            model_current = row["model"]
            models.append(model_current)
            x += 1

        data = pd.read_csv(row["filename"], header=None)

        # Plot the data as a cloud of points around the x point
        xpoints = np.linspace(x - 0.2, x + 0.2, len(data[0]))
        plt.scatter(xpoints, data[0], alpha=0.5)

    # Label the data by model
    plt.xticks(range(1, x + 1), models, rotation=90)
    plt.xlabel("Model")
    plt.ylabel("Eddy-Feedback Parameter")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
