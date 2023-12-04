from pathlib import Path

from tqdm import tqdm
import numpy as np
import iris
from iris.analysis import MEAN

from constrain.eddy_feedback_parameter import eddy_feedback_parameter

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


@load_if_saved(filename="sample_years_{start_year}-{end_year}_n{n_samples}.npy")
def sample_years(**kwargs):
    return np.random.randint(
        kwargs["start_year"],
        kwargs["end_year"] + 1,
        size=(kwargs["n_samples"], kwargs["end_year"] - kwargs["start_year"]),
    )


def extract_sample_years(cubes, **kwargs):
    samples = sample_years(**kwargs)
    year_at_idx0 = cubes[0].coord("season_year").points[0]
    for n in tqdm(range(kwargs["n_samples"])):
        yield [cube[samples[n, :] - year_at_idx0] for cube in cubes]


@load_if_saved(filename="efp_{start_year}-{end_year}_n{n_samples}_{plevs}_bootstrap.npy")
def bootstrap_eddy_feedback_parameter(ep_flux, u_zm, **kwargs):
    results = []

    for ep_flux_sub, u_zm_sub in extract_sample_years([ep_flux, u_zm], **kwargs):
        efp = eddy_feedback_parameter(ep_flux_sub, u_zm_sub)

        if "pressure_level" in [c.name() for c in efp.coords()]:
            efp = efp.collapsed("pressure_level", MEAN)
        results.append(efp.data)

    return results
