"""
Plot of eddy-feedback parameter vs NAO for CMIP6 models and ERA5
1. ERA5 bootstrap eddy-feedback parameter vs NAO
2. Same as 1, but only using 1979-2022
3. CMIP6 ensemble means
4. CMIP6 ensemble members
5/6. Same as 3/4 but for multidecadal NAO variability

Adds a linear regression line for each figure. A second regression added to 2 showing
the weighted average regression calculated individually for each model ensemble
"""
from string import ascii_lowercase

from collections import namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import eddy_feedback
from eddy_feedback import bootstrapping
from eddy_feedback.figures import markers


def main():
    months = ["Dec", "Jan", "Feb"]
    years = "1850-2014"

    n_samples = 1000
    plevs = "500hPa"

    months_str = "".join([m[0] for m in months])

    # Holds the linear regression from each set of points to be plotted on each
    # subfigure
    results = []
    results_weighted = []

    fig, axes = plt.subplots(3, 2, figsize=(8, 12), sharex="all", sharey="row")
    xp = np.arange(0, 1.0, 0.01)

    # Add ERA5
    for n, (start_year, end_year) in enumerate([(1941, 2022), (1980, 2022)]):
        efp_era5 = bootstrapping.bootstrap_eddy_feedback_parameter(
            start_year=start_year,
            end_year=end_year,
            n_samples=n_samples,
            plevs=plevs,
        )
        nao_era5 = bootstrapping.bootstrap_nao(
            start_year=start_year,
            end_year=end_year,
            n_samples=n_samples,
            months=months,
            months_str=months_str,
            detrend=False,
        )
        axes[0, n].plot(efp_era5, nao_era5, ".k", alpha=0.1)
        results.append(linregress(efp_era5, nao_era5))

    # Save model data to dictionaries to do model mean regression
    data = get_data(months_str, years)
    models = sorted(set(data.model))

    for n, nao_type in enumerate(["nao_variance", "nao_variance_multidecadal"]):
        efp_mean = []
        nao_mean = []
        efp_all = []
        nao_all = []

        # Plot the cloud of points for each model and calculate a linear regression
        weighted_average_r = 0.0
        weighted_average_slope = 0.0
        weighted_average_intercept = 0.0
        n_runs = 0

        for m, model in enumerate(models):
            data_model = data.loc[data.model == model]
            if n == 0:
                label = model
            else:
                label = None
            # Subfig 1 shows ensemble mean
            efp_mean.append(np.mean(data_model.efp))
            nao_mean.append(np.mean(data_model[nao_type]))
            axes[n+1, 0].plot(
                efp_mean[-1],
                nao_mean[-1],
                markers[model],
                label=label,
            )

            # Subfig 2 shows all ensemble members
            efp_all.extend(data_model.efp)
            nao_all.extend(data_model[nao_type])
            axes[n+1, 1].plot(
                data_model.efp,
                data_model[nao_type],
                markers[model],
                alpha=0.5,
            )

            # Linear regression for individual model
            result = linregress(data_model.efp, data_model[nao_type])
            print(model, result)
            nvariants = len(data_model.variant)
            weighted_average_r += result.rvalue * nvariants
            weighted_average_slope += result.slope * nvariants
            weighted_average_intercept += result.intercept * nvariants
            n_runs += nvariants

        # Linear regressions
        # Ensemble mean of each model
        results.append(linregress(efp_mean, nao_mean))

        # All simulations from all models
        results.append(linregress(efp_all, nao_all))

        # Weighted average of each model's regression
        res = namedtuple("result", ["rvalue", "slope", "intercept"])
        results_weighted.append(res(
            slope=weighted_average_slope / n_runs,
            intercept=weighted_average_intercept / n_runs,
            rvalue=weighted_average_r / n_runs,
        ))

    # Plot linear regressions
    add_linear_regressions(axes.flatten()[:4], results[:4], results_weighted[0], xp)
    add_linear_regressions(axes.flatten()[4:], results[4:], results_weighted[1], xp)

    axes[0, 0].set_xlim(0.05, 0.6)
    axes[0, 0].set_ylim(5, 34)
    axes[1, 0].set_ylim(5, 34)
    fig.text(0.5, 0.12, "Eddy-Feedback Parameter", ha="center")
    fig.text(0.06, 0.65, "Total NAO Variance (hPa)", va="center", rotation="vertical")
    axes[2, 0].set_ylabel("Multidecadal NAO Variance (hPa)")

    axes[0, 0].set_title("ERA5 (1940-2022)")
    axes[0, 1].set_title("ERA5 (1979-2022)")

    axes[1, 0].set_title("CMIP6 Ensemble Mean")
    axes[1, 1].set_title("CMIP6 All Simulations")

    # Add legend of labels for each CMIP6 model at bottom of figure
    plt.subplots_adjust(bottom=0.15)
    fig.legend(ncol=4, loc="center", bbox_to_anchor=(0.5, 0.05))

    for n, ax in enumerate(axes.flatten()):
        ax.text(0.01, 1.02, f"({ascii_lowercase[n]})", transform=ax.transAxes)

    plt.savefig(
        eddy_feedback.plotdir /
        f"fig4_efp_nao_correlation_cmip6_{months_str}_{years}.png"
    )
    plt.show()


def get_data(months_str, years):
    # Load in model eddy-feedback and NAO variance
    efp_files = eddy_feedback.get_files_by_model(
        "eddy_feedback", months=months_str, years=years
    )
    data = pd.DataFrame(dict(
        model=[], variant=[], efp=[], nao_variance=[], nao_variance_multidecadal=[]
    ))

    nao_1yr = pd.read_csv(
        eddy_feedback.datadir /
        f"NAO_index_data/nao_1-year-variance_{months_str}.csv"
    )
    nao_20yr = pd.read_csv(
        eddy_feedback.datadir /
        f"NAO_index_data/nao_20-year-variance_{months_str}.csv"
    )

    for n, row in efp_files.iterrows():
        model = row["model"]
        fname = str(row["filename"])

        # EFP data gives the actual values of the eddy-feedback parameter and
        # catalogue fname gives the matching variant labels
        efp_data = pd.read_csv(fname, header=None)
        catalogue_fname = fname.replace("EFP", "catalogue").replace(
            f"{model}_{months_str}_{years}", f"daily_ua_va_Spirit_{years}_{model}"
        )
        catalogue = pd.read_csv(catalogue_fname, header=None)

        if len(catalogue) != len(efp_data):
            raise ValueError(f"Catalogue for {model} does not match data")

        # Add EFP and NAO variance to rows of pandas.DataSet for each model/variant
        for m, catalogue_row in catalogue.iterrows():
            variant = catalogue_row[2]
            data = data.append(dict(
                model=model,
                variant=variant,
                efp=efp_data[0][m],
                nao_variance=nao_1yr.loc[np.logical_and(
                    nao_1yr["model"] == model, nao_1yr['variant'] == variant
                )].iloc[0]["nao_variance"],
                nao_variance_multidecadal=nao_20yr.loc[np.logical_and(
                    nao_20yr["model"] == model, nao_20yr['variant'] == variant
                )].iloc[0]["nao_variance"],
            ), ignore_index=True)

    return data


def add_linear_regressions(axes, results, results_weighted, xp):
    # Plot linear regressions
    for n, ax in enumerate(axes):
        for m, result in enumerate(results):
            if n == m:
                alpha = 1.0
            else:
                alpha = 0.5
            ax.plot(xp, result.slope * xp + result.intercept, "-k", alpha=alpha)
        print(results[n])

    # Add weighted average regression separately
    for ax in axes:
        ax.plot(xp, results_weighted.slope * xp + results_weighted.intercept, "--k", alpha=0.5)
    axes[-1].plot(xp, results_weighted.slope * xp + results_weighted.intercept, "--k")
    print("Weighted Average", results_weighted)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
