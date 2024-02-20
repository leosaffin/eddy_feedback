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
from matplotlib.lines import Line2D
import seaborn
from scipy.stats import linregress

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import bootstrapping, datadir, plotdir, get_reanalysis_diagnostic
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
    for start_year, end_year, color in [(1941, 2022, "C7")]:
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
        axes[0, 1].plot(efp_era5, nao_era5, f".{color}", alpha=0.1)
        seaborn.kdeplot(x=efp_era5, ax=kde_ax_x, color=color, fill=True, alpha=0.5)
        seaborn.kdeplot(y=nao_era5, ax=kde_ax_y, color=color, fill=True, alpha=0.5)
        result = linregress(efp_era5, nao_era5)
        print_as_latex(result, "ERA5")
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

    # Add CMIP6 models
    data = pd.read_csv(datadir / "CMIP6_diagnostics_by_model.csv")
    models = sorted(set(data.model), key=lambda x: x.lower())
    for model in models:
        data_model = data.loc[data.model == model]
        efp_mean = np.mean(data_model.efp)

        for n, nao_type in enumerate(["nao_variance", "nao_variance_multidecadal"]):
            if n == 1:
                label = model
            else:
                label = None

            # Ensemble mean
            axes[n, 0].plot(
                efp_mean, np.mean(data_model[nao_type]), markers[model], mec="k", zorder=10, label=label,
            )

            # Ensemble members
            axes[n, 0].plot(
                data_model.efp, data_model[nao_type], markers[model], alpha=0.5
            )

    # Linear regressions
    efp_mean = [np.mean(data.loc[data.model == model].efp) for model in markers]
    for n, diag in enumerate(["nao_variance", "nao_variance_multidecadal"]):
        print("\n" + diag)
        diag_mean = [np.mean(data.loc[data.model == model][diag]) for model in markers]
        result_mean = linregress(efp_mean, diag_mean)
        result_all = linregress(data.efp, data[diag])
        result_weighted = weighted_average_regression(data, markers.keys(), "efp", diag)

        print_as_latex(result_mean, "Mean")
        print_as_latex(result_all, "All")
        print_as_latex(result_weighted, "Weighted")

        for result, linestyle in [
            (result_mean, "-k"),
            (result_all, "--k"),
            (result_weighted, ":k"),
        ]:
            axes[n, 0].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75)
            if "multidecadal" not in diag:
                axes[n, 1].plot(xp, result.slope * xp + result.intercept, linestyle, alpha=0.75)

    for ax in axes.flatten():
        ax.set_xlim(0.1, 0.4)
    axes[0, 0].set_ylim(4, 29)
    kde_ax_x.set_ylim(0, 75)
    kde_ax_y.set_xlim(0.9, 0)

    axes[0, 0].set_title("CMIP6")
    axes[0, 1].set_title("ERA5")
    axes[0, 1].set_xlabel("Eddy-Feedback Parameter")
    axes[1, 0].set_xlabel("Eddy-Feedback Parameter")
    axes[0, 0].set_ylabel("NAO Variance (hPa)\nTotal")
    axes[1, 0].set_ylabel("NAO Variance (hPa)\n20-Year Filter")
    axes[1, 1].set_visible(False)

    # Add legend of labels for each CMIP6 model at bottom of figure
    custom_lines = [
        Line2D([0], [0], linestyle=linestyle, color="k") for linestyle in ["-", "--", ":"]]
    labels = ["Mean", "All", "Weighted"]
    legend = axes[0, 0].legend(custom_lines, labels, framealpha=0.25)
    legend.set_zorder(11)

    custom_lines = [
        Line2D([], [], marker=markers[model][0], color=markers[model][1:], linestyle="", mec="k")
        for model in markers]
    fig.legend(custom_lines, markers.keys(), ncol=2, loc="center", bbox_to_anchor=(0.7, 0.3))

    label_axes(axes.flatten()[:3])

    plt.savefig(plotdir / f"fig3_efp_nao_correlation_cmip6_{months_str}_{years}.pdf")
    plt.show()


res = namedtuple("result", ["slope", "intercept", "rvalue", "pvalue"])
def weighted_average_regression(df, models, diag1, diag2):
    slope, intercept, rvalue, n_runs = 0.0, 0.0, 0.0, 0
    for model in models:
        data_model = df.loc[df.model == model]
        result_ = linregress(data_model[diag1], data_model[diag2])
        print_as_latex(result_, model)
        ensemble_size = len(data_model.index)
        slope += result_.slope * ensemble_size
        intercept += result_.intercept * ensemble_size
        rvalue += result_.rvalue * ensemble_size
        n_runs += ensemble_size

    result_weighted = res(
        slope=slope / n_runs,
        intercept=intercept / n_runs,
        rvalue=rvalue / n_runs,
        pvalue=np.nan,
    )
    return result_weighted


def print_as_latex(result, title):
    print(
        f"{title} & {result.slope:.2f} & {result.intercept:.2f} & {result.rvalue:.2g} &"
        f"{result.pvalue:.2g} \\\\"
    )


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
