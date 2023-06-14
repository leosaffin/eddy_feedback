import pathlib
import re

from parse import parse
import pandas as pd
import iris

plotdir = pathlib.Path(
    "~/Documents/meteorology/output/constrain/eddy_feedback/"
).expanduser()
datadir = pathlib.Path("~/Documents/meteorology/data/constrain/").expanduser()

diagnostic_path = dict(
    arctic_oscillation=datadir / "AO_index_data",
    eddy_feedback=datadir / "eddy_feedback_parameter_data/",
    local_eddy_feedback=datadir / "local_eddy_feedback_parameter_data/ESGF_JASMIN",
    north_atlantic_oscillation=datadir / "NAO_index_data",
    jet=datadir / "Up0NA_index_data",
)
filename_pattern = dict(
    arctic_oscillation="AO-index-TW98_{model}_historical_{variant}.nc",
    eddy_feedback="EFP_CMIP6_LE_hist_{model}_{months}_{years}.csv",
    local_eddy_feedback="G_mean_lat_lon_{months}_{model}_historical_{variant}.nc",
    north_atlantic_oscillation="NAOI_monthly_{months}_{model}_historical_{variant}.nc",
    jet="U500NA_monthly_{months}_{model}_historical_{variant}.nc",
)


def get_reanalysis_diagnostic(diagnostic, reanalysis="ERA5", **other_keywords):
    filename = str(
            diagnostic_path[diagnostic] /
            filename_pattern[diagnostic].replace("_historical_{variant}", "")
    )

    filename = partial_string_format_by_name(filename, dict(model=reanalysis))
    filename = partial_string_format_by_name(filename, other_keywords)

    return iris.load_cube(filename)


def get_files_by_model(diagnostic, **kwargs):
    """Get a DataFrame listing filenames for the diagnostic requested

    The other columns in the DataFrame will list model and other parameters that have
    different files, such as months or years the diagnostic has been calculated over

    Example, if you want the North Atlantic Oscillation for winter months
    >>> get_files_by_model("north_atlantic_oscillation", months="DJF")

    Currently returns:
                 model      years           filename
    0   ACCESS-ESM1-5  1950-2014  /path/to/data/...
    1           CESM2  1850-2014  /path/to/data/...
    2           CESM2  1950-2014  /path/to/data/...
    3    CMCC-CM2-SR5  1850-2014  /path/to/data/...
    4    CMCC-CM2-SR5  1950-2014  /path/to/data/...
    5      CNRM-CM6-1  1850-2014  /path/to/data/...
    6      CNRM-CM6-1  1950-2014  /path/to/data/...
    7     CNRM-ESM2-1  1850-2014  /path/to/data/...
    8         CanESM5  1850-2014  /path/to/data/...
    9         CanESM5  1950-2014  /path/to/data/...
    10      EC-Earth3  1950-2014  /path/to/data/...
    11      EC-Earth3  1850-2014  /path/to/data/...
    12      INM-CM5-0  1850-2014  /path/to/data/...
    13      INM-CM5-0  1950-2014  /path/to/data/...
    14   IPSL-CM6A-LR  1950-2014  /path/to/data/...
    15   IPSL-CM6A-LR  1850-2014  /path/to/data/...
    16     MIROC-ES2L  1850-2014  /path/to/data/...
    17     MIROC-ES2L  1950-2014  /path/to/data/...
    18         MIROC6  1850-2014  /path/to/data/...
    19         MIROC6  1950-2014  /path/to/data/...
    20  MPI-ESM1-2-HR  1850-2014  /path/to/data/...
    21  MPI-ESM1-2-HR  1950-2014  /path/to/data/...
    22  MPI-ESM1-2-LR  1850-2014  /path/to/data/...
    23  MPI-ESM1-2-LR  1950-2014  /path/to/data/...
    24     MRI-ESM2-0  1950-2014  /path/to/data/...
    25    UKESM1-0-LL  1850-2014  /path/to/data/...
    26    UKESM1-0-LL  1950-2014  /path/to/data/...

    Parameters
    ----------
    diagnostic : str
        The name of the diagnostic to load
    kwargs
        Pass other parameters that you want the specific files for. For example,
        "years=1850-2014" to only return the diagnostic calculated on the full
        historical period

    Returns
    -------
    pandas.DataFrame

    """
    filename = partial_string_format_by_name(filename_pattern[diagnostic], kwargs)
    filename_wildcards = re.sub("{\w+}", "*", filename)

    output = pd.DataFrame()
    for n, f in enumerate(diagnostic_path[diagnostic].rglob(filename_wildcards)):
        info = parse(filename, str(f.name)).named
        info["filename"] = f
        df = pd.DataFrame({i: [info[i]] for i in info})
        if n == 0:
            output = df
        else:
            output = pd.concat([output, df], ignore_index=True)

    output.sort_values("model", inplace=True)

    # Remove duplicates where everything but the filename is the same
    output.drop_duplicates(
        subset=output.columns.drop("filename"), inplace=True, ignore_index=True
    )

    return output


def partial_string_format_by_name(string, replacement_dict):
    for key in replacement_dict:
        string = string.replace("{{{}}}".format(key), replacement_dict[key])

    return string
