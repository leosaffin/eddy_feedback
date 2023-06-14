"""
Plot of NAO variance calculated for each ensemble member in CMIP6 historical simulations
grouped by model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import eddy_feedback
from eddy_feedback.nao_variance import formatted_filename


def main():
    diag = "north_atlantic_oscillation"
    filter_length = 20

    for months in ["DJF", "DJFM"]:
        for detrend_nao in [True, False]:
            for normalise_variance in [True, False]:
                filename = formatted_filename(
                    diag=diag,
                    filter_length=filter_length,
                    months=months,
                    detrend_nao=detrend_nao,
                    normalise_variance=normalise_variance
                )

                nao_variance_df = pd.read_csv(filename)

                plt.figure(figsize=(8, 5))
                make_plot(nao_variance_df, normalise_variance)
                plt.title("CMIP6 {} NAO {}-year running mean".format(
                    months, filter_length
                ))
                filename_plot = (
                        eddy_feedback.plotdir /
                        filename.split("/")[-1].replace(".csv", ".png")
                )
                plt.savefig(filename_plot)


def make_plot(data, normalise_variance=False):
    reanalyses = data[data.variant == "reanalysis"]
    linestyles = ["-", "--", "-.", ":"]
    for m, (n, reanalysis) in enumerate(reanalyses.iterrows()):
        plt.axhline(
            reanalysis.nao_variance,
            color="k",
            linestyle=linestyles[m],
            label=reanalysis.model
        )
    data.drop(reanalyses.index, inplace=True)

    models = sorted(set(data["model"]), key=lambda x: x.lower())
    for n, model in enumerate(models):
        data_model = data.loc[data["model"] == model]
        ensmean = data_model[data_model["variant"] == "ensmean"]
        data_model.drop(ensmean.index, inplace=True)

        # Plot the data as a cloud of points around the x point
        xpoints = [n] * len(data_model)
        plt.scatter(xpoints, data_model["nao_variance"], alpha=0.5)
        plt.plot(n, np.mean(data_model["nao_variance"]), f"DC{n % 10}", mec="k")
        plt.plot(n, np.median(data_model["nao_variance"]), f"oC{n % 10}", mec="k")

    # Label the data by model
    plt.xticks(range(len(models)), models, rotation=45, ha="right")
    if normalise_variance:
        ylabel = r"NAO Variance [$\sigma_\mathrm{1-year}$]"
    else:
        ylabel = "NAO Variance [hPa$^2$]"

    plt.ylabel(ylabel)
    plt.legend()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
