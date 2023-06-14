"""
Maps of correlation between seasonal-mean local eddy-feedback and NAO at each position
the local eddy feedback has been calculated. Produces a 2d map for each ensemble member
of each model.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri
import cartopy.crs as ccrs
import iris
from iris.analysis import STD_DEV
from iris.analysis.stats import pearsonr
from iris.util import promote_aux_coord_to_dim_coord

from matplotlib import cm

from irise import plot

from eddy_feedback import get_files_by_model, plotdir
from eddy_feedback.nao_variance import season_mean


def main():
    months = ["Dec", "Jan", "Feb"]
    seasons = ["ndjfma", "mjjaso"]
    as_amplitude = True

    lefp = get_files_by_model("local_eddy_feedback_monthly", months="DJFM")
    nao = get_files_by_model("north_atlantic_oscillation")

    cmap = cm.get_cmap("cmc.vik", 13)

    for n, row in lefp.iterrows():
        model = row["model"]
        variant = row["variant"]

        print(model, variant)

        # Load in monthly-mean NAO data and collapse to season mean
        nao_filename = nao[
            (nao["model"] == model) & (nao["variant"] == variant)
        ].iloc[0]["filename"]
        cube_nao = iris.load_cube(nao_filename)
        cube_nao = season_mean(cube_nao, months, seasons)

        # Load daily local-eddy feedback and collapse to season mean
        cube_lefp = iris.load_cube(row["filename"])
        cube_lefp = season_mean(cube_lefp, months, seasons)

        # Some models don't have wind data to calculate local eddy feedback for the full
        # historical period so match the NAO data to the local eddy feedback data
        cube_nao = cube_nao.extract(iris.Constraint(
            season_year=cube_lefp.coord("season_year").points
        ))

        # Remove non-matching time coordinates that prevent correlation coefficient
        # calculation from running
        for cube in [cube_lefp, cube_nao]:
            cube.remove_coord("time")
            cube.remove_coord("month")
            promote_aux_coord_to_dim_coord(cube, "season_year")

        corr = pearsonr(cube_lefp, cube_nao, corr_coords="season_year")

        figure_filename = str(
            plotdir / f"lefp_maps_cmip6/lefp_nao_correlation_{model}_{variant}.png"
        )

        # Multiply correlation by variability to give an idea of amplitude
        if as_amplitude:
            lefp_variability = cube_lefp.collapsed("season_year", STD_DEV)
            corr = corr * lefp_variability

            figure_filename = figure_filename.replace(
                "correlation", "correlation_amplitude"
            )
            limit = np.abs(corr.data).max()
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = -0.65, 0.65

        plt.figure(figsize=(12, 8))
        plot.pcolormesh(
            corr, vmin=vmin, vmax=vmax, cmap=cmap, projection=ccrs.NorthPolarStereo()
        )
        plt.gca().gridlines(draw_labels=True, color="w")
        plt.title(f"{model} - {variant}")
        plt.savefig(figure_filename)
        plt.close()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
