"""
Plot of eddy-feedback parameter vs NAO for CMIP6 models and ERA5
1. CMIP6 EFP vs total NAO variance
2. ERA5 EFP vs total NAO variance
3. CMIP6 EFP vs multidecadal NAO variance

Adds a linear regression line for each figure. A second regression added to 2 showing
the weighted average regression calculated individually for each model ensemble
"""

from collections import namedtuple

import iris
from iris.analysis import MEAN, STD_DEV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import linregress

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

import eddy_feedback
from eddy_feedback import bootstrapping, datadir, get_reanalysis_diagnostic
from eddy_feedback.nao_variance import season_mean
from eddy_feedback.figures import markers, label_axes


def main():
    months = ["Dec", "Jan", "Feb"]
    years = "1850-2014"
    length = 164
    n_samples = 1000
    plevs = "500hPa"

    months_str = "".join([m[0] for m in months])

    month_cs = iris.Constraint(month=months)
    data_path = datadir / "eddy_feedback/daily_mean/"
    ep_flux = iris.load_cube(data_path / f"era5_daily_EP-flux-divergence_NDJFM.nc", month_cs)
    ep_flux = ep_flux.aggregated_by("season_year", MEAN)[1:-1]
    u_zm = iris.load_cube(data_path / f"era5_daily_zonal-mean-zonal-wind_NDJFM.nc", month_cs)
    u_zm = u_zm.aggregated_by("season_year", MEAN)[1:-1]
    nao = get_reanalysis_diagnostic("north_atlantic_oscillation", months="DJFM")
    nao = season_mean(nao, months=months, seasons=["ndjfma", "mjjaso"])

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey="row")
    kde_ax_x = axes[0, 1].twinx()
    kde_ax_x.yaxis.set_visible(False)
    kde_ax_y = axes[0, 1].twiny()
    kde_ax_y.xaxis.set_visible(False)
    xp = np.arange(0, 0.5, 0.01)
    # Add ERA5
    for start_year, end_year, color in [(1941, 2022, "C0")]:
        efp_era5 = bootstrapping.bootstrap_eddy_feedback_parameter(
            start_year=start_year,
            end_year=end_year,
            length=length,
            n_samples=n_samples,
            plevs=plevs,
        )
        nao_era5 = bootstrapping.bootstrap_nao(
            start_year=start_year,
            end_year=end_year,
            length=length,
            n_samples=n_samples,
            months=months,
            months_str=months_str,
            detrend=False,
        )
        print(np.min(efp_era5), np.min(nao_era5))
        print(np.max(efp_era5), np.max(nao_era5))
        axes[0, 1].plot(efp_era5, nao_era5, f".{color}", alpha=0.1)
        seaborn.kdeplot(x=efp_era5, ax=kde_ax_x, color=color, fill=True, alpha=0.5)
        seaborn.kdeplot(y=nao_era5, ax=kde_ax_y, color=color, fill=True, alpha=0.5)
        result = linregress(efp_era5, nao_era5)
        axes[0, 1].plot(xp, result.slope * xp + result.intercept, f"-{color}", zorder=20)

        # Add full value
        time_cs = iris.Constraint(
            season_year=lambda cell: start_year <= cell <= end_year
        )
        ep_flux_years = ep_flux.extract(time_cs)
        u_zm_years = u_zm.extract(time_cs)
        efp_full = eddy_feedback_parameter(ep_flux_years, u_zm_years).data
        nao_full = nao.extract(time_cs).collapsed("season_year", STD_DEV).data ** 2
        axes[0, 1].plot(efp_full, nao_full, f"o{color}", mec="k", zorder=30, label=f"{start_year-1}-{end_year}")

    # Save model data to dictionaries to do model mean regression
    data = pd.read_csv(datadir / "CMIP6_diagnostics_by_model.csv")
    models = sorted(set(data.model), key=lambda x: x.lower())

    for n, nao_type in enumerate(["nao_variance", "nao_variance_multidecadal"]):
        efp_mean = []
        nao_mean = []

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
        results_all = linregress(data.efp, data[nao_type])

        # Weighted average of each model's regression
        res = namedtuple("result", ["rvalue", "slope", "intercept"])
        results_weighted = res(
            slope=weighted_average_slope / n_runs,
            intercept=weighted_average_intercept / n_runs,
            rvalue=weighted_average_r / n_runs,
        )

        # Plot linear regressions
        for result, linestyle, label in [
            (results_mean, "-k", "Mean"),
            (results_all, "--k", "All"),
            (results_weighted, ":k", "Weighted")
        ]:
            if n == 0:
                axes[n, 1].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75)
            else:
                label = None
            axes[n, 0].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75, label=label, zorder=30)

    for ax in axes.flatten():
        ax.set_xlim(0.1, 0.4)
    axes[0, 0].set_ylim(4, 29)
    kde_ax_x.set_ylim(0, 75)
    kde_ax_y.set_xlim(0.9, 0)

    axes[0, 1].set_xlabel("Eddy-Feedback Parameter")
    axes[1, 0].set_xlabel("Eddy-Feedback Parameter")
    axes[0, 0].set_ylabel("NAO Variance (hPa)\nTotal")
    axes[1, 0].set_ylabel("NAO Variance (hPa)\n20-Year Filter")

    axes[0, 0].set_title("CMIP6")
    axes[0, 1].set_title("ERA5")
    axes[0, 0].legend()

    # Add legend of labels for each CMIP6 model at bottom of figure
    axes[1, 1].axis("off")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), ncol=2, loc="center", bbox_to_anchor=(0.7, 0.3))

    label_axes(axes.flatten()[:3])

    plt.savefig(
        eddy_feedback.plotdir /
        f"fig3_efp_nao_correlation_cmip6_{months_str}_{years}.png"
    )
    plt.show()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
