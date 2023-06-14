import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
from tqdm import tqdm
import iris
from iris.analysis import MEAN, STD_DEV
from iris.coord_categorisation import add_month, add_season_year
from iris.util import broadcast_to_shape
from irise.grid import get_datetime
import pandas as pd
from scipy.stats import linregress

import eddy_feedback

filename_pattern = "nao_{filter_length}-year-variance_{months}.csv"
reanalysis_datasets = ["ERA5", "ERA20C", "20CRv3", "HadSLP2"]


def main():
    diag = "north_atlantic_oscillation"
    seasons = ["ndjfma", "mjjaso"]
    filter_length = 20

    for months in [["Dec", "Jan", "Feb"], ["Dec", "Jan", "Feb", "Mar"]]:
        for detrend_nao in [True, False]:
            for normalise_variance in [True, False]:
                generate_csv(
                    diag, months, seasons, filter_length, detrend_nao, normalise_variance
                )


def generate_csv(diag, months, seasons, filter_length, detrend_nao, normalise_variance):
    # List of values of NAO variance to save as a pandas DataFrame
    rows_list = []

    # NAO filenames for all CMIP6 models and variants
    nao = eddy_feedback.get_files_by_model(diag)
    for n, row in tqdm(nao.iterrows(), total=len(nao.index)):
        nao = iris.load_cube(row["filename"])
        nao_variance = nao_variance_from_monthly_data(
            nao, months, seasons, filter_length, detrend_nao, normalise_variance
        )
        rows_list.append(dict(
            model=row["model"],
            variant=row["variant"],
            nao_variance=nao_variance.data
        ))

    # Add reanalysis datasets
    for reanalysis in reanalysis_datasets:
        nao = eddy_feedback.get_reanalysis_diagnostic(
            diag, reanalysis=reanalysis, months="*"
        )
        nao_variance = nao_variance_from_monthly_data(
            nao, months, seasons, filter_length, detrend_nao, normalise_variance
        )
        rows_list.append(dict(
            model=reanalysis,
            variant="reanalysis",
            nao_variance=nao_variance.data
        ))

    # Save Dataframe of NAO variance with specific choices of calculation in filename
    filename = formatted_filename(
        diag=diag,
        filter_length=filter_length,
        months="".join([m[0] for m in months]),
        detrend_nao=detrend_nao,
        normalise_variance=normalise_variance
    )
    nao_variance_df = pd.DataFrame(rows_list)
    nao_variance_df.to_csv(filename)


def nao_variance_from_monthly_data(
        nao,
        months,
        seasons,
        filter_length,
        detrend_nao=True,
        normalise_variance=False
):
    """

    Parameters
    ----------
    nao : iris.cube.Cube
        Monthly timeseries of North Atlantic Oscillation
    months : list(str)
        The subset of months to calculate seasonal means over
    seasons : list(str)
        A list of seasons to subset the months by
    filter_length : int
        Number of years to average over when applying a running mean to the yearly NAO
        data
    detrend_nao : logical
        Whether to remove the trend in the yearly NAO before applying the running mean
        filter and calculating the variance (default=True)
    normalise_variance : logical
        Whether to normalise the final result by the yearly variance

    Returns
    -------
    iris.cube.Cube

    """
    nao = season_mean(nao, months, seasons)

    if detrend_nao:
        nao = detrend(nao)

    if filter_length == 1:
        return nao.collapsed("season_year", STD_DEV)**2
    else:
        nao_filtered = nao.rolling_window("time", MEAN, filter_length)
        nao_variance = nao_filtered.collapsed("season_year", STD_DEV)**2

        if normalise_variance:
            nao_variance = nao_variance / nao.collapsed("season_year", STD_DEV) ** 2

        return nao_variance


def season_mean(cube, months, seasons):
    if "month" not in [c.name() for c in cube.coords()]:
        add_month(cube, "time")
    cube = cube.extract(iris.Constraint(month=months))

    if "season_year" not in [c.name() for c in cube.coords()]:
        add_season_year(cube, "time", seasons=seasons)

    # Account for different month lengths in weights
    # Easiest is to use the bounds on the time coordinate but if that isn't saved, use
    # datetimes to determine the length of each month
    if cube.coord("time").bounds is not None:
        weights = np.array([x[1] - x[0] for x in cube.coord("time").bounds])
    else:
        weights = np.array([
            (datetime.datetime(x.year, x.month, 1) + relativedelta(months=1) -
             datetime.datetime(x.year, x.month, 1)).total_seconds() / (3600 * 24)
            for x in get_datetime(cube)
        ])

    # Newer versions of iris allow the weights to be specified for aggregated_by but
    # for now this works fine
    weights = broadcast_to_shape(weights, cube.shape, cube.coord_dims("time"))
    weights_cube = cube.copy(data=weights)
    weights_sum = weights_cube.aggregated_by("season_year", MEAN)
    cube = (cube * weights).aggregated_by("season_year", MEAN) / weights_sum

    # Select only full seasons by checking that all months are present
    cube = cube.extract(iris.Constraint(
        month=lambda x: np.array([month in x.point for month in months]).all()
    ))

    return cube


def detrend(cube, time_coord="season_year"):
    """
    Calculate a linear least squares regression from the data, and return the data with
    this removed

    Parameters
    ----------
    cube : iris.cube.Cube
    time_coord : str

    Returns
    -------
    iris.cube.Cube

    """
    x = cube.coord(time_coord).points
    y = cube.data

    result = linregress(x, y)

    detrended_data = y - (result.slope * x + result.intercept)

    return cube.copy(data=detrended_data)


def formatted_filename(diag, filter_length, months, detrend_nao, normalise_variance):
    filename = str(eddy_feedback.diagnostic_path[diag] / filename_pattern.format(
        filter_length=filter_length, months=months,
    ))
    if detrend_nao:
        filename = filename.replace(".csv", "_detrended.csv")
    if normalise_variance:
        filename = filename.replace(".csv", "_normalised-variance.csv")

    return filename


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
