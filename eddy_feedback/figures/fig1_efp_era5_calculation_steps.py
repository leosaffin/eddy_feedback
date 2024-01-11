"""
Plot the calculation steps for the eddy feedback parameter

2x2 plot showing seasonal means of
1. Zonal-mean zonal wind
2. Horizontal EP-Flux Divergence
3. Product of the anomalies (vs time) of the (1) and (2), such that the covariance is
   the sum over time
4. (3) but normalised by the standard deviation of (1) and (2), such that the
   correlation coefficient is the sum over time
"""

import iris
from iris.analysis import MEAN, STD_DEV
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import cmcrameri

from eddy_feedback import datadir, plotdir
from eddy_feedback.figures import label_axes


def main():
    latitude = (25, 72)
    months = ["Dec", "Jan", "Feb"]

    path = datadir / "eddy_feedback/daily_mean"
    ep_flux = iris.load_cube(path / "era5_daily_EP-flux-divergence_NDJFM.nc")
    u_zm = iris.load_cube(path / "era5_daily_zonal-mean-zonal-wind_NDJFM.nc")

    ep_flux = extract_subset(
        ep_flux,
        latitude=latitude,
        months=months,
    )
    u_zm = extract_subset(
        u_zm,
        latitude=latitude,
        months=months,
    )

    ep_flux_mean, ep_flux_anom, ep_flux_std_dev = calc_stats(ep_flux)
    u_zm_mean, u_zm_anom, u_zm_std_dev = calc_stats(u_zm)

    covariance = ep_flux_anom * u_zm_anom
    correlation = covariance / (ep_flux_std_dev * u_zm_std_dev)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex="all", sharey="all")

    coords = ["season_year", "latitude"]
    plt.axes(axes[0, 0])
    make_plot(
        cube=u_zm,
        cmap="cubehelix_r",
        coords=coords,
        cbar_label="m s$^{-1}$",
        title=r"$\bar{u}$",
        vmin=0
    )

    plt.axes(axes[0, 1])
    make_plot(
        cube=ep_flux,
        cmap="cmc.vik",
        coords=coords,
        cbar_label="m s$^{-2}$",
        title=r"$\frac{\nabla . \mathbf{F}_{H}}{\rho a cos(\phi)}$",
    )

    plt.axes(axes[1, 0])
    make_plot(
        cube=covariance,
        cmap="cmc.broc",
        coords=coords,
        cbar_label="m$^{2}$ s$^{-3}$",
        title="Covariance",
    )

    plt.axes(axes[1, 1])
    make_plot(
        cube=correlation,
        cmap="cmc.broc",
        coords=coords,
        cbar_label=" ",
        title="Correlation",
    )

    label_axes(axes.flatten())
    plt.savefig(plotdir / "fig1_eddy-feedback-parameter_calculation_steps.png")
    plt.show()


def extract_subset(cube, latitude, months, pressure_level=None):
    cube = cube.extract(iris.Constraint(month=months))

    cube = cube.intersection(latitude=latitude, ignore_bounds=True)

    if pressure_level == "depth_average":
        cube = cube.collapsed("pressure_level", MEAN)
    elif pressure_level is not None:
        cube = cube.extract(iris.Constraint(pressure_level=pressure_level))

    return cube.aggregated_by(["season_year"], MEAN)[1:-1]


def calc_stats(cube):
    mean = cube.collapsed("season_year", MEAN)
    anom = cube - mean
    std_dev = cube.collapsed("season_year", STD_DEV)

    return mean, anom, std_dev


def make_plot(cube, cmap, coords, cbar_label, title, vmin=-1):
    limit = np.abs(cube.data).max()
    if vmin != 0:
        vmin = -limit
    iplt.pcolormesh(cube, vmin=vmin, vmax=limit, cmap=cmap, coords=coords)
    plt.colorbar(label=cbar_label)
    plt.title(title)


if __name__ == '__main__':
    main()
