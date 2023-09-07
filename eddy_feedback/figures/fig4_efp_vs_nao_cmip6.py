"""
Plot of eddy-feedback parameter vs NAO for CMIP6 models and ERA5
1. CMIP6 ensemble means
2. CMIP6 ensemble members
3. ERA5 bootstrap eddy-feedback parameter vs NAO
4. Same as 3, but only using 1979-2022

Adds a linear regression line for each figure. A second regression added to 2 showing
the weighted average regression calculated individually for each model ensemble
"""

from collections import namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import eddy_feedback
from eddy_feedback.figures import markers


def main():
    months = ["Dec", "Jan", "Feb"]
    years = "1850-2014"
    filter_size = 1

    months_str = "".join([m[0] for m in months])
    efp = eddy_feedback.get_files_by_model(
        "eddy_feedback", months=months_str, years=years
    )
    nao = pd.read_csv(
        eddy_feedback.datadir /
        f"NAO_index_data/nao_{filter_size}-year-variance_{months_str}_detrended.csv"
    )
    models = sorted(list(efp.model), key=lambda x: x.lower())

    # Holds the linear regression from each set of points to be plotted on each
    # subfigure
    results = [None] * 4

    # Load in model eddy-feedback and NAO variance
    efp_by_model = {}
    nao_variance_by_model = {}
    for n, row in efp.iterrows():
        model = row["model"]

        efp_by_model[model] = []
        nao_variance_by_model[model] = []

        fname = str(row["filename"])
        data = pd.read_csv(fname, header=None)
        catalogue_fname = fname.replace("EFP", "catalogue").replace(
            f"{model}_{months_str}_{years}", f"daily_ua_va_Spirit_{years}_{model}"
        )
        catalogue = pd.read_csv(catalogue_fname, header=None)

        if len(catalogue) != len(data):
            raise ValueError(f"Catalogue for {model} does not match data")

        for m, catalogue_row in catalogue.iterrows():
            variant = catalogue_row[2]
            nao_variance = nao.loc[
                (nao["model"] == model) &
                (nao["variant"] == variant)
            ].iloc[0]["nao_variance"]
            efp = data[0][m]

            efp_by_model[model].append(efp)
            nao_variance_by_model[model].append(nao_variance)

    # Plot the cloud of points for each model and calculate a linear regression
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex="all", sharey="all")
    weighted_average_r = 0.0
    weighted_average_slope = 0.0
    weighted_average_intercept = 0.0
    n_runs = 0

    for n, model in enumerate(models):
        result = linregress(efp_by_model[model], nao_variance_by_model[model])

        weighted_average_r += result.rvalue ** 2 * len(efp_by_model[model])
        weighted_average_slope += result.slope * len(efp_by_model[model])
        weighted_average_intercept += result.intercept * len(efp_by_model[model])
        n_runs += len(efp_by_model[model])

        # Subfig 1 shows ensemble mean
        plt.axes(axes[0, 0])
        plt.plot(
            np.mean(efp_by_model[model]),
            np.mean(nao_variance_by_model[model]),
            markers[model],
            label=f"{model}"
        )

        # Subfig 2 shows all ensemble members
        plt.axes(axes[0, 1])
        plt.plot(
            efp_by_model[model],
            nao_variance_by_model[model],
            markers[model],
            alpha=0.5
        )
        print(model, result)

    # Linear regressions
    # Ensemble mean of each model
    efp_mean = [np.mean(efp_by_model[model]) for model in efp_by_model]
    nao_mean = [np.mean(nao_variance_by_model[model]) for model in efp_by_model]
    results[0] = linregress(efp_mean, nao_mean)

    # All simulations from all models
    efp_all = [x for model in efp_by_model for x in efp_by_model[model]]
    nao_all = [x for model in nao_variance_by_model for x in nao_variance_by_model[model]]
    results[1] = linregress(efp_all, nao_all)

    # Weighted average of each model's regression
    res = namedtuple("result", ["rvalue", "slope", "intercept"])
    results_weighted = res(
        slope=weighted_average_slope / n_runs,
        intercept=weighted_average_intercept / n_runs,
        rvalue=(weighted_average_r / n_runs) ** 0.5,
    )

    # ADD ERA5
    for n, reanalysis in enumerate(["era5", "era5_1979-2020"]):
        efp_era5 = np.load(f"efp_{reanalysis}_bootstrap.npy")
        nao_era5 = np.load(f"nao_variance_{reanalysis}_bootstrap.npy")
        results[n + 2] = linregress(efp_era5, nao_era5)

        plt.axes(axes[1, n])
        plt.scatter(efp_era5, nao_era5, color="k", marker=".", alpha=0.1)

    # Plot linear regressions
    xp = np.arange(0, 1.0, 0.01)
    for n, ax in enumerate(axes.flatten()):
        for m, result in enumerate(results):
            if n == m:
                alpha = 1.0
            else:
                alpha = 0.5
            ax.plot(xp, result.slope * xp + result.intercept, "-k", alpha=alpha)
        print(results[n])

    # Add weighted average regression separately
    for ax in axes.flatten():
        ax.plot(xp, results_weighted.slope * xp + results_weighted.intercept, "--k", alpha=0.5)
    axes[0, 1].plot(xp, results_weighted.slope * xp + results_weighted.intercept, "--k")
    print("Weighted Average", results_weighted)

    axes[0, 0].set_xlim(0.05, 0.6)
    axes[0, 0].set_ylim(5, 34)
    fig.text(0.5, 0.05, "Eddy-Feedback Parameter", ha="center")
    fig.text(0.05, 0.5, f"NAO Variability ({filter_size}-year)", va="center", rotation="vertical")

    axes[0, 0].set_title("Ensemble Mean")
    axes[0, 1].set_title("All Simulations")
    axes[1, 0].set_title("ERA5 (1940-2022)")
    axes[1, 1].set_title("ERA5 (1979-2022)")

    plt.subplots_adjust(hspace=0.5)
    fig.legend(ncol=4, loc="center", bbox_to_anchor=(0.5, 0.5))

    plt.savefig(
        eddy_feedback.plotdir /
        f"fig4_efp_nao_{filter_size}-year-variability_correlation_cmip6_{months_str}_{years}.png"
    )
    plt.show()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
