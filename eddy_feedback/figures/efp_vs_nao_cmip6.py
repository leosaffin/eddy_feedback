"""
Plot of eddy-feedback parameter vs NAO for CMIP6 models

Shows all individual points for each model and calculate regression slops for
    1. Ensemble Means
    2. All ensemble members from all models
    3. A weighted average of the regression for each individual model

Add reference lines for values calculated from ERA5 (1940-2020)
"""

import iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

import eddy_feedback
from eddy_feedback.nao_variance import season_mean

markers = ["ok", "oC0", "oC1", "oC2", "oC3", "oC4", "oC5",
           "Xk", "XC0", "XC1", "XC2", "XC3", "XC4", "XC5"]


def main():
    months = ["Dec", "Jan", "Feb"]
    years = "1850-2014"
    filter_size = 1

    months_str = "".join([m[0] for m in months])
    efp = eddy_feedback.get_files_by_model("eddy_feedback", months=months_str, years=years)
    nao = pd.read_csv(
        eddy_feedback.datadir /
        f"NAO_index_data/nao_{filter_size}-year-variance_{months_str}_detrended.csv"
    )

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
    plt.figure(figsize=(8, 5))
    weighted_average_r = 0.0
    weighted_average_slope = 0.0
    weighted_average_intercept = 0.0
    n_runs = 0
    for n, model in enumerate(efp_by_model):
        result = linregress(efp_by_model[model], nao_variance_by_model[model])

        weighted_average_r += result.rvalue ** 2 * len(efp_by_model[model])
        weighted_average_slope += result.slope * len(efp_by_model[model])
        weighted_average_intercept += result.intercept * len(efp_by_model[model])
        n_runs += len(efp_by_model[model])

        plt.plot(efp_by_model[model], nao_variance_by_model[model], markers[n], label=f"{model}")
        print(f"{model}", result)

    # ADD ERA5
    data_path = eddy_feedback.datadir / "constrain/eddy_feedback/daily_mean"
    ep_flux = iris.load_cube(data_path / "era5_daily_EP-flux-divergence_NDJFM.nc")
    ep_flux = season_mean(ep_flux, months=months, seasons=["ndjfma", "mjjaso"])
    u_zm = iris.load_cube(data_path / "era5_daily_zonal-mean-zonal-wind_NDJFM.nc")
    u_zm = season_mean(u_zm, months=months, seasons=["ndjfma", "mjjaso"])

    efp = eddy_feedback_parameter(ep_flux, u_zm)
    nao_era5 = nao[nao["model"] == "ERA5"]["nao_variance"].iloc[0]

    plt.axvline(efp.data, color="k")
    plt.axhline(nao_era5, color="k")

    # Linear regressions
    # Ensemble mean of each model
    efp_mean = [np.mean(efp_by_model[model]) for model in efp_by_model]
    nao_mean = [np.mean(nao_variance_by_model[model]) for model in efp_by_model]
    result = linregress(efp_mean, nao_mean)
    xmin, xmax = 0.1, 0.4
    x = np.arange(xmin, xmax, 0.01)
    plt.plot(x, result.slope * x + result.intercept, "--k", label="Ensemble mean")
    print("Ensemble Mean", result)

    # All simulations from all models
    efp_all = [x for model in efp_by_model for x in efp_by_model[model]]
    nao_all = [x for model in nao_variance_by_model for x in nao_variance_by_model[model]]
    result = linregress(efp_all, nao_all)
    x = np.arange(xmin, xmax, 0.01)
    plt.plot(x, result.slope * x + result.intercept, ":k", label="All Simulation")
    print("All Simulation", result)

    # Weighted average of each model's regression
    slope = weighted_average_slope / n_runs
    intercept = weighted_average_intercept / n_runs
    plt.plot(x, slope * x + intercept, "-.k", label="Weighted average")
    print(f"Weighted average r={(weighted_average_r / n_runs) ** 0.5:.3f}, slope={slope}, intercept={intercept}")

    plt.legend(ncol=1, bbox_to_anchor=(1.01, 0.95))
    plt.xlim(xmin, xmax)
    plt.xlabel("Eddy-Feedback Parameter")
    plt.ylabel(f"NAO Variability ({filter_size}-year)")

    plt.savefig(
        eddy_feedback.plotdir /
        f"efp_nao_{filter_size}-year-variability_correlation_cmip6_{months_str}_{years}.png"
    )
    plt.show()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
