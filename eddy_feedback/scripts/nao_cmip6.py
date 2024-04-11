"""
Calculating NAO for CMIP6 data from ESGF downloaded to Leeds
"""

from pathlib import Path

from parse import parse
import iris
from iris.coord_categorisation import add_month

from constrain import north_atlantic_oscillation


def main():
    input_path = Path("/nfs/annie/earlsa/cmip6/psl_Amon")
    filename_pattern = "psl_Amon_{model}_historical_{variant}_mergetime_NATL_rg_{months}.nc"

    output_path = Path("/nfs/annie/earlsa/nao")
    output_filename = "NAOI_monthly_{months}_{model}_historical_{variant}.nc"

    for cmip_file in input_path.rglob(filename_pattern.format(model="*", variant="*", months="*")):
        info = parse(filename_pattern, cmip_file.name).named
        print(info)

        mslp = iris.load_cube(cmip_file)
        add_month(mslp, "time")
        nao_index = north_atlantic_oscillation.from_boxes(mslp)

        iris.save(nao_index, output_path / output_filename.format(**info))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
