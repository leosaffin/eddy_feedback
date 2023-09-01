import iris

from constrain import north_atlantic_oscillation, regrid_to_coarsest

from eddy_feedback import datadir


def main():
    mslp = iris.load_cube(datadir / f"constrain/ERA20C/prmsl.mon.mean.nc")
    for coord in ["longitude", "latitude"]:
        mslp.coord(coord).guess_bounds()

    mslp = regrid_to_coarsest(mslp)
    nao = north_atlantic_oscillation.from_boxes(mslp)
    iris.save(nao, datadir / f"NAO_index_data/NAOI_monthly_all_ERA20C_CanESM5-grid.nc")


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    main()
