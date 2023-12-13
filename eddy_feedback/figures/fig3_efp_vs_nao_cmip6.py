"""
Plot of eddy-feedback parameter vs NAO for CMIP6 models and ERA5
1. CMIP6 EFP vs total NAO variance
2. ERA5 EFP vs total NAO variance
3. CMIP6 EFP vs multidecadal NAO variance

Adds a linear regression line for each figure. A second regression added to 2 showing
the weighted average regression calculated individually for each model ensemble
"""

from collections import namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import eddy_feedback
from eddy_feedback import bootstrapping
from eddy_feedback.figures import markers, label_axes


def main():
    months = ["Dec", "Jan", "Feb"]
    years = "1850-2014"
    n_samples = 1000
    plevs = "500hPa"

    months_str = "".join([m[0] for m in months])

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey="row")
    xp = np.arange(0, 1.0, 0.01)
    # Add ERA5
    for start_year, end_year, color in [(1941, 2022, "C0"), (1980, 2022, "C1")]:
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
        print(np.min(efp_era5), np.min(nao_era5))
        print(np.max(efp_era5), np.max(nao_era5))
        axes[0, 1].plot(efp_era5, nao_era5, f".{color}", alpha=0.1)
        result = linregress(efp_era5, nao_era5)
        axes[0, 1].plot(xp, result.slope * xp + result.intercept, f"-{color}", zorder=30, label=f"{start_year-1}-{end_year}")

    # Save model data to dictionaries to do model mean regression
    data = get_data(months_str, years)
    models = sorted(set(data.model), key=lambda x: x.lower())

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
            if n == 1:
                label = model
            else:
                label = None

            data_model = data.loc[data.model == model]

            efp_all.extend(data_model.efp)
            nao_all.extend(data_model[nao_type])
            axes[n, 0].plot(
                data_model.efp,
                data_model[nao_type],
                markers[model],
                alpha=0.5,
            )

            # Overlay ensemble mean
            efp_mean.append(np.mean(data_model.efp))
            nao_mean.append(np.mean(data_model[nao_type]))
            axes[n, 0].plot(
                efp_mean[-1],
                nao_mean[-1],
                markers[model],
                mec="k",
                label=label,
                zorder=20,
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
        results_mean = linregress(efp_mean, nao_mean)

        # All simulations from all models
        results_all = linregress(efp_all, nao_all)

        # Weighted average of each model's regression
        res = namedtuple("result", ["rvalue", "slope", "intercept"])
        results_weighted = res(
            slope=weighted_average_slope / n_runs,
            intercept=weighted_average_intercept / n_runs,
            rvalue=weighted_average_r / n_runs,
        )

        # Plot linear regressions
        for result, linestyle, label in [
            (results_mean, "-k", "Ensemble Mean"),
            (results_all, "--k", "All Simulations"),
            (results_weighted, ":k", "Weighted Average")
        ]:
            if n == 0:
                axes[n, 1].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75)
            else:
                label = None
            axes[n, 0].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75, label=label, zorder=30)

    for ax in axes.flatten():
        ax.set_xlim(0.08, 0.6)
    axes[0, 0].set_ylim(4, 32)
    axes[1, 0].set_xlabel("Eddy-Feedback Parameter")
    axes[0, 0].set_ylabel("NAO Variance (hPa)\nTotal")
    axes[1, 0].set_ylabel("NAO Variance (hPa)\n20-Year Filter")

    axes[0, 0].set_title("CMIP6")
    axes[0, 1].set_title("ERA5")
    axes[0, 0].legend()
    axes[0, 1].legend()

    # Add legend of labels for each CMIP6 model at bottom of figure
    axes[1, 1].axis("off")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), ncol=2, loc="center", bbox_to_anchor=(0.7, 0.3))

    label_axes(axes.flatten()[:3])

    plt.savefig(
        eddy_feedback.plotdir /
        f"fig3_efp_nao_correlation_cmip6_{months_str}_{years}.png"
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


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
