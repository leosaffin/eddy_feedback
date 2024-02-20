import iris
from iris.analysis import MEAN, STD_DEV
import iris.plot as iplt
import matplotlib.pyplot as plt

from eddy_feedback import get_reanalysis_diagnostic, plotdir
from eddy_feedback.nao_variance import detrend, season_mean
from eddy_feedback.figures import label_axes


def main():
    months = ["Dec", "Jan", "Feb"]
    seasons = ["ndjfma", "mjjaso"]
    diag = "north_atlantic_oscillation"

    for n, (start_year, end_year, suffix) in enumerate([
        (1836, 2022, "full_period"),
        (1940, 2005, "common_period"),
    ]):
        cs = iris.Constraint(season_year=lambda x: start_year < x <= end_year)
        make_plot(diag, months, seasons, cs)
        #plt.xlim(start_year, end_year)
        plt.savefig(plotdir / f"figA{n+2}_nao_reanalysis_trends_{suffix}.pdf")

    plt.show()


def make_plot(diag, months, seasons, cs):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex="all")

    for reanalysis, linestyle, alpha in [
        ("ERA5", "-k",  1.0),
        ("ERA20C", "--k", 0.5),
        ("20CRv3", "--C0",  0.5),
        ("HadSLP2", "--C1", 0.5),
    ]:
        print(reanalysis)
        nao = get_reanalysis_diagnostic(diag, reanalysis=reanalysis, months="*")
        nao = season_mean(nao, months=months, seasons=seasons)
        nao = nao.extract(cs)

        nao_20yr = nao.rolling_window("season_year", MEAN, 20)
        nao_variance = nao.collapsed("season_year", STD_DEV).data ** 2
        nao_20yr_variance = nao_20yr.collapsed("season_year", STD_DEV).data ** 2

        plt.axes(axes[0, 0])
        label = f"{reanalysis} ({nao_variance:.2f})"
        iplt.plot(nao, linestyle, label=label, alpha=alpha)

        plt.axes(axes[0, 1])
        label = f"{reanalysis} ({nao_20yr_variance:.2f})"
        iplt.plot(nao_20yr, linestyle, alpha=alpha, label=label)

        nao_detrend = detrend(nao)
        nao_20yr = nao_detrend.rolling_window("season_year", MEAN, 20)
        nao_variance = nao_detrend.collapsed("season_year", STD_DEV).data ** 2
        nao_20yr_variance = nao_20yr.collapsed("season_year", STD_DEV).data ** 2

        plt.axes(axes[1, 0])
        label = f"{reanalysis} ({nao_variance:.2f})"
        iplt.plot(nao_detrend, linestyle, label=label, alpha=alpha)

        plt.axes(axes[1, 1])
        label = f"{reanalysis} ({nao_20yr_variance:.2f})"
        iplt.plot(nao_20yr, linestyle, alpha=alpha, label=label)

    axes[0, 0].set_title("Full")
    axes[0, 1].set_title("20-year mean")
    axes[1, 0].set_title("Detrended")
    axes[1, 1].set_title("Detrended, 20-year mean")

    fig.subplots_adjust(hspace=0.6, bottom=0.2)
    for ax in axes.flatten():
        ax.legend(title="Reanalysis (variance)", ncol=2, bbox_to_anchor=(0.1, -0.6, 1, 0.5))

    label_axes(axes.flatten())

    fig.text(0.01, 0.5, "NAO (hPa)", rotation="vertical", va="center")
    fig.text(0.5, 0.01, "Year")


if __name__ == '__main__':
    main()
