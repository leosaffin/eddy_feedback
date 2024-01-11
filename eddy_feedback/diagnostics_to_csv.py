
import numpy as np
import pandas as pd
import iris
from iris.analysis import MEAN

from eddy_feedback import get_files_by_model, datadir, local_eddy_feedback_north_atlantic_index


def main():
    results = get_data("DJF", "1850-2014")
    results = results.sort_values(by=["model", "variant"])
    results.to_csv(datadir / "CMIP6_diagnostics_by_model.csv")


def get_data(months_str, years):
    # Load in model eddy-feedback and NAO variance
    efp_files = get_files_by_model("eddy_feedback", months=months_str, years=years)
    data = pd.DataFrame(dict(
        model=[], variant=[], efp=[], nao_variance=[], nao_variance_multidecadal=[], G_na=[],
    ))

    nao_1yr = pd.read_csv(
        datadir / f"NAO_index_data/nao_1-year-variance_{months_str}.csv"
    )
    nao_20yr = pd.read_csv(
        datadir / f"NAO_index_data/nao_20-year-variance_{months_str}.csv"
    )

    # More models have data for NAO than EFP so just collect data for models with EFP
    for n, row in efp_files.iterrows():
        model = row["model"]
        fname_efp = str(row["filename"])

        # EFP data gives the actual values of the eddy-feedback parameter and
        # catalogue fname gives the matching variant labels
        efp_data = pd.read_csv(fname_efp, header=None)
        catalogue_fname = fname_efp.replace("EFP", "catalogue").replace(
            f"{model}_{months_str}_{years}", f"daily_ua_va_Spirit_{years}_{model}"
        )
        catalogue = pd.read_csv(catalogue_fname, header=None)

        if len(catalogue) != len(efp_data):
            raise ValueError(f"Catalogue for {model} does not match data")

        # Add EFP and NAO variance to rows of pandas.DataSet for each model/variant
        for m, catalogue_row in catalogue.iterrows():
            variant = catalogue_row[2]
            fname_lefp = local_eddy_feedback_north_atlantic_index.output_filename.format(
                model=model, variant=variant
            )

            lefp = iris.load_cube(fname_lefp)
            lefp = lefp.collapsed("season_year", MEAN).data

            data = pd.concat([data, pd.DataFrame([dict(
                model=model,
                variant=variant,
                efp=efp_data[0][m],
                nao_variance=nao_1yr.loc[np.logical_and(
                    nao_1yr["model"] == model, nao_1yr['variant'] == variant
                )].iloc[0]["nao_variance"],
                nao_variance_multidecadal=nao_20yr.loc[np.logical_and(
                    nao_20yr["model"] == model, nao_20yr['variant'] == variant
                )].iloc[0]["nao_variance"],
                G_na=lefp,
            )])], ignore_index=True)

    return data


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
