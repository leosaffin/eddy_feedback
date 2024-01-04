from pathlib import Path

from parse import parse
from tqdm import tqdm
import numpy as np
import iris
from iris.analysis import MEAN, STD_DEV
from iris.coord_categorisation import add_season_year

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

from eddy_feedback import datadir, get_reanalysis_diagnostic, local_eddy_feedback_north_atlantic_index
from eddy_feedback.nao_variance import season_mean, detrend

cached_files = Path(__file__).parent / "bootstrap_data/"


def load_if_saved(filename):
    """Generate a decorator for the bootstrapping functions that loads the file if it
    has already been produced or does the analysis and saves the data if not

    Args:
        filename (str): A string representation of filename that can be formatted by
            passing keyword arguments
    """
    def decorator(func):
        def inner(*args, **kwargs):
            fname = filename.format(**kwargs)
            path = cached_files / fname

            if path.exists():
                print(f"Loading saved file: {path}")

                return np.load(path)
            else:
                print(f"Generating file: {path}")

                result = func(*args, **kwargs)
                np.save(path, result)

            return result

        return inner

    return decorator


@load_if_saved(filename="sample_years_{start_year}-{end_year}_l{length}_n{n_samples}.npy")
def sample_years(**kwargs):
    if "length" not in kwargs:
        length = kwargs["end_year"] - kwargs["start_year"]
    else:
        length = kwargs["length"]

    return np.random.randint(
        kwargs["start_year"],
        kwargs["end_year"] + 1,
        size=(kwargs["n_samples"], length),
    )


def extract_sample_years(cubes, **kwargs):
    years = cubes[0].coord("season_year").points.astype(int)
    assert kwargs["start_year"] >= years[0]
    assert kwargs["end_year"] <= years[-1]

    samples = sample_years(**kwargs)
    for n in tqdm(range(kwargs["n_samples"])):
        yield [cube[samples[n, :] - years[0]] for cube in cubes]


@load_if_saved(filename="efp_{start_year}-{end_year}_l{length}_n{n_samples}_{plevs}_bootstrap.npy")
def bootstrap_eddy_feedback_parameter(**kwargs):
    data_path = datadir / "eddy_feedback/daily_mean"

    if kwargs["plevs"] == "500hPa":
        suffix = "NDJFM"
    else:
        suffix = "600-200hPa_DJF"

    cs = iris.Constraint(month=["Dec", "Jan", "Feb"])
    ep_flux = iris.load_cube(data_path / f"era5_daily_EP-flux-divergence_{suffix}.nc", cs)
    u_zm = iris.load_cube(data_path / f"era5_daily_zonal-mean-zonal-wind_{suffix}.nc", cs)

    cubes = []
    for cube in [ep_flux, u_zm]:
        if "season_year" not in [c.name() for c in cube.coords()]:
            add_season_year(cube, "time", seasons=["ndjfma", "mjjaso"])
        cube_yearly = cube.aggregated_by("season_year", MEAN)[1:-1]
        cubes.append(cube_yearly)

    results = []
    for ep_flux_sub, u_zm_sub in extract_sample_years(cubes, **kwargs):
        efp = eddy_feedback_parameter(ep_flux_sub, u_zm_sub)

        if "pressure_level" in [c.name() for c in efp.coords()]:
            efp = efp.collapsed("pressure_level", MEAN)
        results.append(efp.data)

    return results


@load_if_saved(filename="lefp_{start_year}-{end_year}_l{length}_n{n_samples}_{plevs}_bootstrap.npy")
def bootstrap_local_eddy_feedback_parameter(**kwargs):
    lefp_era5 = iris.load_cube(local_eddy_feedback_north_atlantic_index.output_filename_era5)

    plev = parse("{p:d}hPa", kwargs["plevs"])["p"]
    lefp_era5 = lefp_era5.extract(iris.Constraint(pressure_level=plev))

    results = []
    for lefp_sample in extract_sample_years([lefp_era5], **kwargs):
         results.append(lefp_sample[0].collapsed("season_year", MEAN).data)

    return results


@load_if_saved(filename="nao_{start_year}-{end_year}_{months_str}_l{length}_n{n_samples}_bootstrap.npy")
def bootstrap_nao(**kwargs):
    nao = get_reanalysis_diagnostic("north_atlantic_oscillation", months="DJFM")
    nao = season_mean(nao, months=kwargs["months"], seasons=["ndjfma", "mjjaso"])

    if detrend in kwargs and kwargs["detrend"] is True:
        nao = detrend(nao)

    results = []
    for nao_sub in extract_sample_years([nao], **kwargs):
        results.append(nao_sub[0].collapsed("season_year", STD_DEV).data**2)

    return results
